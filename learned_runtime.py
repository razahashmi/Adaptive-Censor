"""Learned detectors without PyTorch: shared feature definitions, ONNX feature models and heads.

Training (features.py, learned.py) and the player (analysis_engine) both use these
functions, so heads see identical inputs in both places. Feature models run through
ONNX Runtime on CUDA; see export_learned.py.
"""
import json
import hashlib
import os
from pathlib import Path
import threading

import numpy as np
from frame_cache import BoundedCache
from provenance import cached_revision, file_hash

GRID = .25            # head decisions every 0.25 s
STEP = .5             # motion/audio feature centers every 0.5 s
CLIP_RATE = 4         # CLIP frames per second
VJEPA_FRAMES, VJEPA_RATE, VJEPA_SIZE = 16, 8, 256
AST_SPAN = 2.0
VERIFIABLE = ('sexual_activity', 'violence', 'gore', 'kissing')


# ---- feature definitions shared by training and the player -------------------
def vjepa_clips(times: np.ndarray, pixels: np.ndarray, centers) -> list:
    """Sixteen consecutive ~8 fps frames around each center, clamped at the media edges."""
    span = VJEPA_FRAMES / VJEPA_RATE
    clips = []
    for c in centers:
        first = int(np.argmin(np.abs(times - (c - span / 2))))
        first = max(0, min(first, len(pixels) - VJEPA_FRAMES))
        clips.append(pixels[[min(len(pixels) - 1, first + k) for k in range(VJEPA_FRAMES)]])
    return clips


def squash_frames(frames, size=VJEPA_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """Resize whole pictures, no center crop, so action at the edges stays visible."""
    import cv2
    return (np.array([t for t, _ in frames]),
            np.stack([cv2.resize(p, (size, size), interpolation=cv2.INTER_AREA) for _, p in frames]))


def audio_segments(audio: np.ndarray, offset: float, centers) -> list:
    """Two seconds of 16 kHz audio around each center; `audio` starts at `offset` seconds."""
    return [audio[max(0, int((c - AST_SPAN / 2 - offset) * 16000)):max(0, int((c + AST_SPAN / 2 - offset) * 16000))]
            for c in centers]


def ast_spectrogram(extractor, waveform: np.ndarray) -> np.ndarray:
    """The AST extractor's numpy path: log-mel bank, zero-padded to 1024 frames, then normalized."""
    from transformers.audio_utils import spectrogram
    fbank = spectrogram(np.squeeze(waveform).astype(np.float32), extractor.window, frame_length=400, hop_length=160,
                        fft_length=512, power=2.0, center=False, preemphasis=0.97, mel_filters=extractor.mel_filters,
                        log_mel='log', mel_floor=1.192092955078125e-07, remove_dc_offset=True).T
    padded = np.zeros((extractor.max_length, extractor.num_mel_bins), dtype=np.float32)
    padded[:min(len(fbank), extractor.max_length)] = fbank[:extractor.max_length]
    return (padded - extractor.mean) / (extractor.std * 2)


def nearest(times, values, at):
    return values[np.abs(np.asarray(times)[None, :] - at[:, None]).argmin(1)].astype(np.float32)


def feature_matrix(features: dict, times: np.ndarray, kinds, temporal_offsets=()) -> np.ndarray:
    """Head inputs at each time: CLIP now and averaged over ±1 s, then motion and audio embeddings."""
    columns = []
    for kind in kinds:
        data = features[kind]
        columns.append(nearest(data['times'], data['embeddings'], times))
        if kind == 'clip':
            values = data['embeddings'].astype(np.float32)
            weights = (np.abs(np.asarray(data['times'])[None, :] - times[:, None]) <= 1).astype(np.float32)
            columns.append((weights @ values) / np.maximum(weights.sum(1, keepdims=True), 1))
        if kind == 'vjepa':
            # Ordered context uses resident embeddings; it adds no video-model calls.
            for offset in temporal_offsets:
                columns.append(nearest(data['times'], data['embeddings'], times + offset))
    return np.concatenate(columns, 1)


def head_scores(head: dict, matrix: np.ndarray) -> np.ndarray:
    if head.get('model_type') == 'histogram-trees':
        from nonlinear_heads import tree_scores
        return tree_scores(head,matrix)
    z = (matrix - np.asarray(head['mean'])) / np.asarray(head['scale'])
    return z @ np.asarray(head['coef']) + head['intercept']


def head_outputs(heads: dict, times: np.ndarray, matrix: np.ndarray):
    """Trace candidates and automatic decisions from trained heads."""
    entries, events = [], []
    if heads.get('projection'):
        from nonlinear_heads import project_inputs
        matrix=project_inputs(heads['projection'],matrix)
    half = heads['grid'] / 2
    enabled = heads.get('enabled')  # chosen on validation; None means every trained head
    for category, head in heads['heads'].items():
        if ('coef' not in head and 'trees' not in head) or (enabled is not None and category not in enabled):
            continue
        for t, score in zip(times, head_scores(head, matrix)):
            if score >= head['candidate_threshold']:
                entries.append(dict(stage='classifier', category=category, start=float(t) - half, end=float(t) + half,
                                    sample_time=float(t), score=float(score)))
            if head['decision_threshold'] is not None and score >= head['decision_threshold']:
                events.append(dict(start=float(t) - half, end=float(t) + half, sample_time=float(t), category=category,
                                   confidence=1, severity=2, enabled=True, review_required=False, source='Learned head',
                                   score_type='decision', note=f'Learned {category} head score {score:.2f}'))
    return entries, events


def uncertain_spans(prediction: dict, stages=('classifier', 'audio'), gap=.3, longest=1.0):
    """Candidate spans not already decided, split into pieces of at most `longest` seconds.

    Confident decisions never reach the verifier, so it can only add evidence;
    it cannot veto them.
    """
    decided = {c: [(e['start'], e['end']) for e in prediction['events']
                   if e['category'] == c and not e.get('review_required')] for c in VERIFIABLE}
    spans = []
    for category in VERIFIABLE:
        times = sorted({round(e['sample_time'], 3) for e in prediction['trace']
                        if e['category'] == category and e['stage'] in stages
                        and not any(a <= e['sample_time'] < b for a, b in decided[category])})
        group = []
        for t in times + [None]:
            if group and (t is None or t - group[-1] > gap or t - group[0] >= longest):
                spans.append((category, group[0], group[-1]))
                group = []
            if t is not None:
                group.append(t)
    return spans


def verified_events(spans, check, low=float('-inf'), high=float('inf')):
    """One evidence check per span at its middle; absent evidence adds nothing."""
    entries, events = [], []
    for category, first, last in spans:
        middle = (first + last) / 2
        if not low <= middle < high:
            continue
        evidence = check(category, middle)
        entries.append(dict(stage='selective_evidence', category=category, start=first - GRID / 2, end=last + GRID / 2,
                            sample_time=middle, state=evidence['state'], detail=evidence['observation']))
        if evidence['state'] != 'absent':
            events.append(dict(start=first - GRID / 2, end=last + GRID / 2, sample_time=middle, category=category,
                               confidence=1, severity=2, enabled=True, review_required=evidence['state'] == 'uncertain',
                               source='Verified learned candidate', evidence=evidence['label'], score_type='decision',
                               note=evidence['observation']))
    return entries, events


# ---- heads available to the player ------------------------------------------
def available_heads(root: Path, promoted_only=True) -> list[str]:
    names = []
    for path in sorted((Path(root) / 'heads').glob('*.json')):
        heads = json.loads(path.read_text(encoding='utf-8'))
        if heads.get('promoted') or not promoted_only:
            names.append(path.stem)
    return names


def runtime_ready(root: Path) -> bool:
    manifest = Path(root) / 'learned-runtime' / 'manifest.json'
    return manifest.is_file() and {'vjepa', 'ast'} <= set(json.loads(manifest.read_text(encoding='utf-8'))['models'])


class FeatureModels:
    """CLIP, V-JEPA 2 and AST features for an absolute time range of any media file."""

    def __init__(self, models, root: Path, require_manifest=True):
        from transformers import ASTFeatureExtractor
        from features import AST, REVISIONS  # features.py imports PyTorch lazily
        self.models, self.root = models, Path(root)
        self.folder = self.root / 'learned-runtime'
        if require_manifest and not runtime_ready(self.root):
            raise RuntimeError('Learned feature models are not installed. Run export_learned.py.')
        local = self.folder / 'ast-extractor'
        self.extractor = (ASTFeatureExtractor.from_pretrained(local, local_files_only=True) if local.is_dir()
                          else ASTFeatureExtractor.from_pretrained(AST, revision=REVISIONS[AST],
                                                                   cache_dir=self.root / 'hf', local_files_only=True))
        self.sessions = {}
        self.session_lock = threading.RLock()
        self.inference_gate = threading.BoundedSemaphore(max(1, int(os.environ.get('ADAPTIVE_CENSOR_LEARNED_GPU_JOBS', '1'))))
        self.embedding_cache = BoundedCache(32 * 1024**2)
        self.vjepa_batch = max(1,min(2,int(os.environ.get('ADAPTIVE_CENSOR_VJEPA_BATCH','2'))))
        manifest = self.folder / 'manifest.json'
        installed = json.loads(manifest.read_text(encoding='utf-8'))['models'] if manifest.is_file() else {}
        self.files = {kind: installed.get(kind, {}).get('file', f'{kind}.onnx') for kind in ('vjepa', 'ast')}

    def session(self, kind):
        with self.session_lock:
            return self._session(kind)

    def _session(self, kind):
        if kind not in self.sessions:
            ort, options, providers = self.models.ort('cuda')
            if kind == 'vjepa':
                # Full-precision attention over 2048 tokens needs about 1 GB per clip and layer.
                providers = [(providers[0][0], dict(providers[0][1], gpu_mem_limit=6 * 1024**3)), *providers[1:]]
            self.sessions[kind] = ort.InferenceSession(str(self.folder / self.files[kind]), sess_options=options,
                                                       providers=providers)
        return self.sessions[kind]

    def warm_up(self, kinds):
        """Initialize each required session once before concurrent window work."""
        for kind in kinds:
            if kind in ('vjepa', 'ast'):
                self.session(kind)

    def motion_embeddings(self, clips):
        # Input hashes preserve edge clamping and exact sampled pixels.
        stamp=file_hash(self.folder/self.files['vjepa'])
        keys=[('vjepa',stamp,clip.shape,str(clip.dtype),hashlib.sha256(clip.tobytes()).hexdigest()) for clip in clips]
        def infer(indices):
            values=np.stack([clips[i] for i in indices]).astype(np.float16)
            with self.inference_gate:
                outputs=self.session('vjepa').run(None,{'frames':values})[0]
            return [row[None].copy() for row in outputs]
        return self.embedding_cache.get_many(keys,infer)

    def compute(self, kind, path, start, end, centers=None, duration=None):
        """Features whose evidence lies in [start, end); motion/audio centers default to the 0.5 s grid."""
        from analysis_engine import audio_samples, native_frames
        end = min(end, duration) if duration else end
        if centers is None:
            centers = np.arange(np.ceil(start / STEP) * STEP, end + 1e-9, STEP)
        centers = np.asarray(centers, dtype=np.float32)
        if kind == 'clip':
            frames = native_frames(path, max(0, start), end, CLIP_RATE)
            return dict(times=np.array([t for t, _ in frames], dtype=np.float32),
                        embeddings=self.models.clip_embeddings(frames, 'cpu', fit='squash'))
        if kind == 'vjepa':
            span = VJEPA_FRAMES / VJEPA_RATE
            frames = native_frames(path, max(0, float(centers.min()) - span / 2 - .2),
                                   float(centers.max()) + span / 2 + .2 if not duration else min(duration, float(centers.max()) + span / 2 + .2),
                                   VJEPA_RATE)
            times, pixels = squash_frames(frames)
            embeddings = []
            clips = vjepa_clips(times, pixels, centers)
            for offset in range(0,len(clips),self.vjepa_batch):
                embeddings += self.motion_embeddings(clips[offset:offset+self.vjepa_batch])
            return dict(times=centers, embeddings=np.concatenate(embeddings))
        if kind == 'ast':
            low = max(0.0, float(centers.min()) - AST_SPAN / 2)
            high = float(centers.max()) + AST_SPAN / 2
            if duration:
                high = min(duration, high)
            audio = audio_samples(path, low, high)
            spectra = np.stack([ast_spectrogram(self.extractor, segment) for segment in audio_segments(audio, low, centers)])
            session, probabilities, embeddings = self.session('ast'), [], []
            for offset in range(0, len(spectra), 8):
                with self.inference_gate:
                    p, e = session.run(None, {'spectrogram': spectra[offset:offset + 8].astype(np.float16)})
                probabilities.append(p); embeddings.append(e)
            return dict(times=centers, probabilities=np.concatenate(probabilities), embeddings=np.concatenate(embeddings))
        raise ValueError(f'Unknown feature kind {kind}')
