"""Export the learned feature models to ONNX for the player runtime, and check parity.

Runs in the training environment. The player then needs only ONNX Runtime, not
PyTorch. Each export is compared with PyTorch on real clips; a model whose
outputs differ beyond the tolerance is not written to the manifest.

    .train-venv\Scripts\python export_learned.py export
    .venv\Scripts\python export_learned.py check --project dev-check
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from features import AST, VJEPA, REVISIONS, AudioEvents, VJepa, load_features
from labeling import project_dir, read_json
from model_setup import prepare_environment


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def export_vjepa(root: Path, output: Path, half=False):
    """V-JEPA 2 with pooling. In half precision only the rotary angles stay in float32:
    the model builds them with a float16 Range that ONNX Runtime rejects."""
    import torch
    from transformers.models.vjepa2 import modeling_vjepa2 as modeling
    extractor = VJepa(root)
    dtype = torch.float16 if half else torch.float32

    class Pooled(torch.nn.Module):
        def __init__(self, model, mean, std):
            super().__init__()
            self.model, self.mean, self.std = model.to(dtype), mean, std

        def forward(self, frames):  # B, T, H, W, C pixel values carried as float16
            videos = (frames.float().permute(0, 1, 4, 2, 3) / 255 - self.mean) / self.std
            return self.model.get_vision_features(videos.to(dtype)).float().mean(1)

    original = modeling.rotate_queries_or_keys

    def rotate_in_float32(x, pos):
        return original(x.float(), pos.float()).to(x.dtype)

    module = Pooled(extractor.model, extractor.mean, extractor.std).eval()
    dummy = torch.zeros(1, extractor.frames_per_clip, 256, 256, 3, device='cuda', dtype=torch.float16)
    modeling.rotate_queries_or_keys = rotate_in_float32 if half else original
    try:
        torch.onnx.export(module, (dummy,), str(output), input_names=['frames'], output_names=['embeddings'],
                          dynamic_axes={'frames': {0: 'batch'}, 'embeddings': {0: 'batch'}}, opset_version=18, dynamo=False)
    finally:
        modeling.rotate_queries_or_keys = original
    return extractor


def export_ast(root: Path, output: Path):
    import torch
    extractor = AudioEvents(root)

    class Events(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, spectrogram):
            result = self.model(input_values=spectrogram, output_hidden_states=True)
            return torch.sigmoid(result.logits.float()), result.hidden_states[-1].float().mean(1)

    module = Events(extractor.model).eval()
    dummy = torch.zeros(1, 1024, 128, device='cuda', dtype=torch.float16)
    torch.onnx.export(module, (dummy,), str(output), input_names=['spectrogram'], output_names=['probabilities', 'embeddings'],
                      dynamic_axes={'spectrogram': {0: 'batch'}, 'probabilities': {0: 'batch'}, 'embeddings': {0: 'batch'}},
                      opset_version=18, dynamo=False)
    extractor.extractor.save_pretrained(output.parent / 'ast-extractor')
    return extractor


def cosine(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return float(np.min(np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)))


def check(root: Path, project: str, tolerance: float, probability_tolerance: float):
    """Compare ONNX features (player runtime, CUDA) with cached PyTorch features of real clips."""
    from analysis_engine import Models
    import learned_runtime
    folder = root / 'learned-runtime'
    reference = project_dir(root, project)
    windows = read_json(reference / 'project.json')['windows'][:3]
    runtime = learned_runtime.FeatureModels(Models(), root, require_manifest=False)
    manifest = dict(checked=time.strftime('%Y-%m-%dT%H:%M:%S'), reference=project, cosine_tolerance=tolerance,
                    probability_tolerance=probability_tolerance, models={})
    # Half precision first: the fastest file that matches PyTorch is installed.
    for kind, source, files in (('vjepa', VJEPA, ['vjepa-half.onnx', 'vjepa.onnx']), ('ast', AST, ['ast.onnx'])):
        for file in [f for f in files if (folder / f).is_file()]:
            runtime.files[kind] = file
            runtime.sessions.pop(kind, None)
            similarity, difference, started = [], [0.0], time.perf_counter()
            for window in windows:
                expected = load_features(reference, kind, window['id'], allow_legacy=True)
                computed = runtime.compute(kind, reference / 'clips' / f"{window['id']}.mp4", 0, window['clip_duration'],
                                           expected['times'], window['clip_duration'])
                similarity.append(cosine(expected['embeddings'], computed['embeddings']))
                if kind == 'ast':
                    difference.append(float(np.abs(expected['probabilities'].astype(np.float32) - computed['probabilities']).max()))
            print(f'{file}: embedding cosine >= {min(similarity):.5f}, event probability difference <= {max(difference):.4f} '
                  f'on {len(windows)} clips, {time.perf_counter() - started:.1f} s', flush=True)
            if min(similarity) >= tolerance and max(difference) <= probability_tolerance:
                path = folder / file
                manifest['models'][kind] = dict(source=source, revision=REVISIONS[source], file=file, sha256=sha256(path), bytes=path.stat().st_size,
                                                cosine=round(min(similarity), 6), probability_difference=round(max(difference), 5))
                break
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=1), encoding='utf-8')
    missing = {'vjepa', 'ast'} - set(manifest['models'])
    if missing:
        raise SystemExit(f'Not installed for the player (outside tolerance): {sorted(missing)}')
    print(f'Wrote {folder / "manifest.json"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('export', help='Write ONNX files (training environment, needs PyTorch)')
    sub.add_parser('preprocessor', help='Save the installed AST extractor locally without downloading or exporting weights')
    verify = sub.add_parser('check', help='Parity against cached features (player environment, CUDA)')
    verify.add_argument('--project', required=True, help='Labeling project with cached PyTorch features')
    verify.add_argument('--tolerance', type=float, default=.999, help='Minimum embedding cosine similarity to PyTorch')
    verify.add_argument('--probability-tolerance', type=float, default=.01, help='Largest audio event probability difference')
    args = parser.parse_args()
    root = prepare_environment()
    if args.command == 'preprocessor':
        from transformers import ASTFeatureExtractor
        extractor = ASTFeatureExtractor.from_pretrained(AST, revision=REVISIONS[AST], cache_dir=root / 'hf', local_files_only=True)
        extractor.save_pretrained(root / 'learned-runtime' / 'ast-extractor')
        print('Saved local AST preprocessor')
        return
    if args.command == 'check':
        return check(root, args.project, args.tolerance, args.probability_tolerance)
    folder = root / 'learned-runtime'
    folder.mkdir(exist_ok=True)
    for kind, export in (('vjepa', export_vjepa), ('vjepa-half', lambda r, o: export_vjepa(r, o, half=True)), ('ast', export_ast)):
        started = time.perf_counter()
        export(root, folder / f'{kind}.onnx')
        size = sum(f.stat().st_size for f in folder.glob(f'{kind}.onnx*'))
        print(f'{kind}: exported in {time.perf_counter() - started:.0f}s, {size / 1e6:.0f} MB', flush=True)


if __name__ == '__main__':
    main()
