"""Cache the exact chunk-scoped head inputs used by playback, for training."""
import argparse
from pathlib import Path
import time

import numpy as np

from labeling import is_complete, project_dir, read_json, write_json
from provenance import digest, file_hash, pipeline_identity

LAYOUT = 'player-chunks-v1'


def chunk_inputs(feature_models, path, start, end, duration, kinds, temporal_offsets=()):
    from learned_runtime import GRID, STEP, feature_matrix
    times = np.arange(start + GRID / 2, end, GRID)
    centers = np.arange(np.floor(start / STEP) * STEP, np.ceil(end / STEP) * STEP + 1e-9, STEP)
    centers = centers[centers <= duration]
    features = {}
    for kind in kinds:
        if kind == 'clip':
            features[kind] = feature_models.compute(kind, path, max(0, start - 1.25), min(duration, end + 1.25))
        else:
            features[kind] = feature_models.compute(kind, path, start, end, centers, duration)
    return times, feature_matrix(features, times, kinds, temporal_offsets)


def cache_path(folder, window, kinds, offsets):
    profile = digest(dict(layout=LAYOUT, kinds=list(kinds), offsets=list(offsets)))[:16]
    return folder / 'player-inputs' / profile / f"{window['id']}.npz"


def contract(folder, window, kinds, offsets):
    from analysis_engine import ScanConfig
    root = folder.parent.parent
    # Bind exact decode/feature code, installed ONNX assets and preprocessing.
    pipeline = pipeline_identity(root, ScanConfig(nudity=False, speech=False, scenes=False), 1)
    runtime = root / 'learned-runtime'
    manifest = read_json(runtime/'manifest.json')
    assets = {m['file']:file_hash(runtime/m['file']) for m in manifest['models'].values()}
    assets.update({p.name:file_hash(p) for p in (runtime/'ast-extractor').glob('*.json')})
    assets['clip'] = file_hash(root/'clip/onnx/vision_model_quantized.onnx')
    return dict(layout=LAYOUT, kinds=list(kinds), offsets=list(offsets), duration=window['clip_duration'],
                source=file_hash(folder/'clips'/f"{window['id']}.mp4"), assets=assets,
                code=pipeline['code'], dependencies=pipeline['dependencies'],
                input_code=file_hash(Path(__file__)), tuning=pipeline['runtime_tuning'])


def load_matrix(folder, window, kinds, offsets=()):
    path = cache_path(folder, window, kinds, offsets)
    manifest = read_json(path.with_suffix('.json'), {})
    current=contract(folder,window,kinds,offsets)
    recorded={k:manifest.get(k) for k in current}
    if (not path.is_file() or manifest.get('identity')!=digest(recorded)
            or not compatible_reader(recorded,current)):
        raise ValueError(f'Missing or stale exact-player inputs for {window["id"]}; run player_inputs.py prepare')
    if file_hash(path) != manifest.get('data_sha256'):
        raise ValueError('Exact-player input cache was modified')
    with np.load(path) as data:
        return data['times'], data['matrix']


def compatible_reader(recorded,current):
    """Reading numeric arrays does not rerun the extraction environment.

    Keep its recorded dependencies bound to the data, while checking sources,
    code and assets against the current workspace. Preprocessing libraries must
    still match; the reader need not install the extractor's ONNX GPU package.
    """
    ignored={'dependencies'}
    if {k:v for k,v in recorded.items() if k not in ignored}!={k:v for k,v in current.items() if k not in ignored}:return False
    relevant=('numpy','av','opencv-python-headless','transformers','tokenizers')
    return all(recorded['dependencies'].get(k)==current['dependencies'].get(k) for k in relevant)


def prepare(folder, window, models, kinds, offsets=()):
    from analysis_engine import CHUNK
    path = cache_path(folder, window, kinds, offsets)
    expected = contract(folder, window, kinds, offsets)
    if path.exists():
        return load_matrix(folder, window, kinds, offsets)
    feature_models = models.learned_features()
    duration = window['clip_duration']
    pieces = [chunk_inputs(feature_models, folder/'clips'/f"{window['id']}.mp4", start,
                          min(start+CHUNK, duration), duration, kinds, offsets)
              for start in np.arange(0, duration, CHUNK)]
    times = np.concatenate([t for t,_ in pieces]); matrix = np.concatenate([x for _,x in pieces])
    if not np.isfinite(matrix).all() or len(times)!=len(matrix) or not np.all(np.diff(times)>0):
        raise ValueError('Invalid exact-player features')
    if digest(expected)!=digest(contract(folder, window, kinds, offsets)):
        raise ValueError('Source or processing changed during extraction')
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, times=times, matrix=matrix)
    write_json(path.with_suffix('.json'), dict(expected,identity=digest(expected),data_sha256=file_hash(path)))
    return times, matrix


def main():
    from analysis_engine import Models
    from model_setup import prepare_environment
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare']);p.add_argument('--projects',nargs='+',required=True)
    p.add_argument('--kinds',nargs='+',default=['clip','vjepa','ast'])
    p.add_argument('--temporal-offsets',nargs='*',type=float,default=[-.5,.5])
    args=p.parse_args();root=prepare_environment(); models=Models(evaluate_unpromoted=True)
    started=time.perf_counter()
    try:
        models.learned_features().warm_up(args.kinds)
        for name in args.projects:
            folder=project_dir(root,name)
            windows=[w for w in read_json(folder/'project.json')['windows']
                     if w['split']=='train' and is_complete(read_json(folder/'labels'/f"{w['id']}.json",{}))]
            for i,w in enumerate(windows,1):
                prepare(folder,w,models,args.kinds,args.temporal_offsets)
                print(f'{name}: {i}/{len(windows)} exact-player matrices ({time.perf_counter()-started:.1f}s)',flush=True)
    finally:
        models.close()


if __name__=='__main__':main()
