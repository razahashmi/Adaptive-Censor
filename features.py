"""Cached learned features for labeling windows: CLIP frames, V-JEPA 2 motion and AST audio.

Runs in the training environment (.train-venv), not the player runtime. Features
cover each whole clip, context included, so heads can use surrounding time and
can be retrained without decoding video again.

    .train-venv\\Scripts\\python features.py --name test-1 --kind clip vjepa ast
"""
import argparse
from pathlib import Path
import time

import numpy as np

from analysis_engine import Models, audio_samples, native_frames
from labeling import project_dir, read_json
from labeling import write_json
from provenance import FEATURE_SCHEMA, cached_revision, digest, file_hash
from learned_runtime import CLIP_RATE, STEP, VJEPA_FRAMES, VJEPA_RATE, audio_segments, squash_frames, vjepa_clips
from model_setup import prepare_environment

VJEPA = 'facebook/vjepa2-vitl-fpc64-256'
AST = 'MIT/ast-finetuned-audioset-10-10-0.4593'
REVISIONS = {VJEPA: 'b3c1679b7c34d3255ef3547f27c7b226aefab26f', AST: 'f826b80d28226b62986cc218e5cec390b1096902'}
KINDS = ('clip', 'vjepa', 'ast')


def feature_path(folder: Path, kind: str, window_id: str) -> Path:
    return folder / 'features' / kind / f'{window_id}.npz'


def feature_contract(folder, kind, window_id):
    root = folder.parent.parent
    if kind == 'clip':
        models = [root / 'clip' / 'onnx' / 'vision_model_quantized.onnx']
    else:
        model = VJEPA if kind == 'vjepa' else AST
        snapshot = root / 'hf' / ('models--' + model.replace('/', '--')) / 'snapshots' / REVISIONS[model]
        models = list(snapshot.glob('*.safetensors')) + list(snapshot.glob('*.json'))
    return dict(schema=FEATURE_SCHEMA, kind=kind, source=file_hash(folder / 'clips' / f'{window_id}.mp4'),
                assets={p.name: file_hash(p) for p in models},
                preprocessing={p: file_hash(Path(__file__).parent / p) for p in ('features.py', 'learned_runtime.py')})


def load_features(folder: Path, kind: str, window_id: str, allow_legacy=False) -> dict | None:
    path = feature_path(folder, kind, window_id)
    if not path.is_file():
        return None
    manifest = read_json(path.with_suffix('.json'), {})
    if not manifest and not allow_legacy:
        raise ValueError(f'Unversioned features: {path}; regenerate with features.py --force or explicitly allow frozen legacy experiments')
    if manifest and manifest.get('identity') != digest(feature_contract(folder, kind, window_id)):
        raise ValueError(f'Stale feature cache: {path}; regenerate with features.py --force')
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def centers(duration: float) -> np.ndarray:
    return np.arange(1.0, max(1.0, duration - .5) + 1e-9, STEP)


class ClipFrames:
    """The installed CLIP ViT-B/32, whole picture resized, four frames per second."""
    def __init__(self, root):
        self.models = Models()

    def __call__(self, clip: Path, duration: float) -> dict:
        frames = native_frames(clip, 0, duration, CLIP_RATE)
        return dict(times=np.array([t for t, _ in frames], dtype=np.float32),
                    embeddings=self.models.clip_embeddings(frames, 'cpu', fit='squash').astype(np.float16))


class VJepa:
    """V-JEPA 2 ViT-L: 16 frames at ~8 fps (2 s) per center, mean-pooled tokens."""
    def __init__(self, root, batch=4):
        import torch
        from transformers import AutoModel
        self.torch, self.frames_per_clip, self.batch = torch, VJEPA_FRAMES, batch
        self.model = AutoModel.from_pretrained(VJEPA, revision=REVISIONS[VJEPA], cache_dir=root / 'hf', local_files_only=True, dtype=torch.float16).cuda().eval()
        # The published video preprocessor rescales to 0-1 and applies ImageNet
        # statistics; resizing is done here without its center crop.
        self.mean = torch.tensor([.485, .456, .406], device='cuda').view(1, 1, 3, 1, 1)
        self.std = torch.tensor([.229, .224, .225], device='cuda').view(1, 1, 3, 1, 1)

    def __call__(self, clip: Path, duration: float) -> dict:
        times, pixels = squash_frames(native_frames(clip, 0, duration, VJEPA_RATE))
        middle = centers(duration)
        clips = vjepa_clips(times, pixels, middle)
        embeddings = []
        for offset in range(0, len(clips), self.batch):
            videos = self.torch.from_numpy(np.stack(clips[offset:offset + self.batch])).cuda()
            videos = (videos.permute(0, 1, 4, 2, 3).float() / 255 - self.mean) / self.std  # B, T, C, H, W
            with self.torch.no_grad():
                hidden = self.model.get_vision_features(videos.half())
            embeddings.append(hidden.float().mean(1).cpu().numpy())
        return dict(times=middle.astype(np.float32), embeddings=np.concatenate(embeddings).astype(np.float16))


class AudioEvents:
    """AST AudioSet tagger on 2 s of audio per center: 527 event probabilities plus an embedding."""
    def __init__(self, root, span=2.0, batch=8):
        import torch
        from transformers import ASTFeatureExtractor, ASTForAudioClassification
        self.torch, self.span, self.batch = torch, span, batch
        self.extractor = ASTFeatureExtractor.from_pretrained(AST, revision=REVISIONS[AST], cache_dir=root / 'hf', local_files_only=True)
        self.model = ASTForAudioClassification.from_pretrained(AST, revision=REVISIONS[AST], cache_dir=root / 'hf', local_files_only=True, dtype=torch.float16).cuda().eval()
        self.labels = [self.model.config.id2label[i] for i in range(len(self.model.config.id2label))]

    def __call__(self, clip: Path, duration: float) -> dict:
        audio = audio_samples(clip, 0, duration)
        middle = centers(duration)
        segments = audio_segments(audio, 0, middle)
        probabilities, embeddings = [], []
        for offset in range(0, len(segments), self.batch):
            inputs = self.extractor(segments[offset:offset + self.batch], sampling_rate=16000, return_tensors='pt')
            with self.torch.no_grad():
                output = self.model(input_values=inputs['input_values'].cuda().half(), output_hidden_states=True)
            probabilities.append(self.torch.sigmoid(output.logits.float()).cpu().numpy())
            embeddings.append(output.hidden_states[-1].float().mean(1).cpu().numpy())
        return dict(times=middle.astype(np.float32), probabilities=np.concatenate(probabilities).astype(np.float16),
                    embeddings=np.concatenate(embeddings).astype(np.float16), labels=np.array(self.labels))


EXTRACTORS = dict(clip=ClipFrames, vjepa=VJepa, ast=AudioEvents)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--name', required=True, help='Labeling project')
    parser.add_argument('--kind', nargs='+', choices=KINDS, default=list(KINDS))
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    root = prepare_environment()
    folder = project_dir(root, args.name)
    project = read_json(folder / 'project.json')
    for kind in args.kind:
        extractor, started = None, time.perf_counter()
        for window in project['windows']:
            path = feature_path(folder, kind, window['id'])
            if path.is_file() and not args.force:
                load_features(folder, kind, window['id'])  # validate identity before reuse
                continue
            extractor = extractor or EXTRACTORS[kind](root)
            values = extractor(folder / 'clips' / f"{window['id']}.mp4", window['clip_duration'])
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **values)
            contract = feature_contract(folder, kind, window['id'])
            write_json(path.with_suffix('.json'), dict(contract, identity=digest(contract)))
        print(f'{kind}: {len(project["windows"])} windows ready in {time.perf_counter() - started:.1f}s', flush=True)


if __name__ == '__main__':
    main()
