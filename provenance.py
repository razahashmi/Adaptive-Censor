"""Content identities for models, preprocessing, predictions and saved coverage."""
import hashlib
import importlib.metadata
from functools import lru_cache
import json
import os
from pathlib import Path
import threading

FEATURE_SCHEMA = 2
_hashes = {}
_lock = threading.RLock()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    path = Path(path)
    if not path.is_file():
        return None
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    with _lock:
        if key not in _hashes:
            h = hashlib.sha256()
            with path.open('rb') as f:
                for block in iter(lambda: f.read(1 << 20), b''):
                    h.update(block)
            _hashes[key] = h.hexdigest()
        return _hashes[key]


def head_identity(heads):
    # Selection/provenance dates and certification do not change the function.
    fields={k: heads.get(k) for k in ('kinds', 'grid', 'heads', 'enabled', 'feature_schema', 'temporal_offsets', 'projection')}
    fields.update({k:heads[k] for k in ('input_layout','attention_model') if k in heads})
    return digest(fields)


def code_identity():
    folder = Path(__file__).parent
    files = ('analysis_engine.py', 'scene_verifier.py', 'learned_runtime.py', 'features.py',
             'core.py', 'detectors.py', 'provenance.py', 'frame_cache.py', 'roi_gate.py', 'anatomy_runtime.py', 'nonlinear_heads.py',
             'player_inputs.py','learned_confirmation.py','attention_probe.py')
    return {name: file_hash(folder / name) for name in files}


@lru_cache(maxsize=1)
def dependency_identity():
    packages=('numpy','av','onnxruntime','onnxruntime-gpu','opencv-python-headless','transformers',
              'faster-whisper','ctranslate2','tokenizers','nudenet')
    versions={}
    for name in packages:
        try: versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name]=None
    return versions


def pipeline_identity(root, config, workers=1):
    config = config.model_dump() if hasattr(config, 'model_dump') else dict(config)
    settings = {k: v for k, v in config.items() if k != 'lookahead'}
    assets = {}
    if root is not None:
        root = Path(root)
        files = []
        if config.get('learned_heads'):
            files += [root / 'heads' / f"{config['learned_heads']}.json", root / 'learned-runtime' / 'manifest.json']
            head_file=root/'heads'/f"{config['learned_heads']}.json"
            head=json.loads(head_file.read_text(encoding='utf-8')) if head_file.is_file() else {}
            if head.get('attention_model'):
                files += [root/'attention-runtime'/head['attention_model']['file'],root/'attention-runtime/tokens.onnx',root/'attention-runtime/tokens.json']
            manifest = root / 'learned-runtime' / 'manifest.json'
            if manifest.is_file():
                models = json.loads(manifest.read_text(encoding='utf-8')).get('models', {})
                files += [root / 'learned-runtime' / m['file'] for m in models.values()]
            files += list((root / 'learned-runtime' / 'ast-extractor').glob('*.json'))
        if config.get('nudity'):
            if config.get('anatomy_roi'):
                files += [root/'roi-models'/f"{config['anatomy_roi']}.json",root/'clip/onnx/vision_model_quantized.onnx']
            if config.get('anatomy_model') == '640m':
                files += [root / 'scene-verifier' / '640m.onnx']
            else:
                import importlib.util
                spec = importlib.util.find_spec('nudenet')
                if spec and spec.origin:
                    files += [Path(spec.origin).parent / '320n.onnx']
        if config.get('scenes') or config.get('learned_heads'):
            files += list((root / 'clip').rglob('*.onnx')) + list((root / 'clip').glob('*.json'))
        if config.get('scenes') and (config.get('scene_mode') == 'verified' or config.get('learned_verification') in ('selective','confirm')):
            size = config.get('scene_model', '4B').lower()
            files += [p for p in (root / 'scene-verifier').rglob('*')
                      if p.is_file() and (size in p.name.lower() or p.suffix == '.exe')]
        if config.get('speech'):
            files += list((root / 'models').rglob('*' + config.get('speech_model', 'tiny.en') + '*'))
            # CTranslate2 weights/configs can live in a snapshot directory.
            files += [p for p in (root / 'models').rglob('*') if p.is_file()]
        for p in sorted(set(files)):
            if p.is_file():
                key = str(p.relative_to(root)) if p.is_relative_to(root) else str(p)
                assets[key] = (head_identity(json.loads(p.read_text(encoding='utf-8')))
                               if p.parent == root / 'heads' else file_hash(p))
    return dict(schema=FEATURE_SCHEMA, config=settings, workers=workers, code=code_identity(), assets=assets,
                dependencies=dependency_identity(),
                runtime_tuning=dict(vjepa_batch=max(1,min(2,int(os.environ.get('ADAPTIVE_CENSOR_VJEPA_BATCH','2')))),
                                    learned_gpu_jobs=max(1,int(os.environ.get('ADAPTIVE_CENSOR_LEARNED_GPU_JOBS','1')))))


def qualification_identity(pipeline):
    config = {k:v for k,v in pipeline['config'].items() if k != 'scene_filtering'}
    return digest(dict(schema=pipeline['schema'], code=pipeline['code'], assets=pipeline['assets'],
                       config=config, workers=pipeline['workers'],runtime_tuning=pipeline['runtime_tuning'],
                       dependencies=pipeline['dependencies']))


def qualified(heads, verification='off'):
    certificate = heads.get('promoted') or {}
    return (isinstance(certificate, dict) and certificate.get('qualified') is True
            and certificate.get('head_identity') == head_identity(heads)
            and certificate.get('verification') == verification)


def cached_revision(root, model):
    """Require an installed, immutable HF snapshot; never fetch 'main' at playback."""
    folder = Path(root) / 'hf' / ('models--' + model.replace('/', '--'))
    ref = folder / 'refs' / 'main'
    revision = ref.read_text().strip() if ref.is_file() else ''
    snapshots = sorted((folder / 'snapshots').glob('*'))
    if not revision and len(snapshots) == 1:
        revision = snapshots[0].name
    if not revision or not (folder / 'snapshots' / revision).is_dir():
        raise RuntimeError(f'Install a pinned snapshot of {model} before using learned models.')
    return revision
