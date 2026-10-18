"""Fine-tune V-JEPA 2 on labeled training windows: an attentive probe, optionally the last blocks.

Use this when frozen-feature heads (learned.py) fall short, mainly for actions
such as violence and sexual activity. It trains only on train-split labels,
holds out whole titles for early stopping and threshold choice, and writes a
derived prediction run that evaluate_labels.py scores like any other.

    .train-venv\\Scripts\\python finetune.py train --projects train-1 --model ft-1 --unfreeze 2
    .train-venv\\Scripts\\python finetune.py apply --name test-1 --base rev9 --model ft-1 --run rev9-ft-1
"""
import argparse
import json
from pathlib import Path
import random
import time

import numpy as np

from labeling import LABEL_CATEGORIES, is_complete, project_dir, read_json
from learned import add_trace, derive_run, grid_labels, thresholds
from learned_runtime import GRID, STEP, VJEPA_RATE, squash_frames, vjepa_clips
from model_setup import prepare_environment


def window_clips(folder: Path, window: dict):
    """Center times and 16-frame clips covering the whole window clip, as uint8."""
    from analysis_engine import native_frames
    times, pixels = squash_frames(native_frames(folder / 'clips' / f"{window['id']}.mp4", 0, window['clip_duration'], VJEPA_RATE))
    centers = np.arange(1.0, max(1.0, window['clip_duration'] - .5) + 1e-9, STEP)
    return centers, np.stack(vjepa_clips(times, pixels, centers))


def build_model(root: Path, unfreeze: int, backbone=None):
    import torch
    from torch import nn

    class AttentiveProbe(nn.Module):
        """One learned query attends over all video tokens (V-JEPA's evaluation protocol)."""
        def __init__(self, dim, heads, outputs):
            super().__init__()
            self.query = nn.Parameter(torch.randn(1, 1, dim) * .02)
            self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
            self.norm = nn.LayerNorm(dim)
            self.output = nn.Linear(dim, outputs)

        def forward(self, tokens):
            pooled, _ = self.attention(self.query.expand(len(tokens), -1, -1), tokens, tokens, need_weights=False)
            return self.output(self.norm(pooled[:, 0]))

    class Classifier(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone
            dim = backbone.config.hidden_size
            self.probe = AttentiveProbe(dim, 16, len(LABEL_CATEGORIES))
            self.register_buffer('mean', torch.tensor([.485, .456, .406]).view(1, 1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([.229, .224, .225]).view(1, 1, 3, 1, 1))

        def forward(self, frames):  # B, T, H, W, C uint8
            videos = (frames.permute(0, 1, 4, 2, 3).float() / 255 - self.mean) / self.std
            return self.probe(self.backbone.get_vision_features(videos).float())

    if backbone is None:
        from transformers import AutoModel
        from features import VJEPA, REVISIONS
        backbone = AutoModel.from_pretrained(VJEPA, revision=REVISIONS[VJEPA], cache_dir=root / 'hf', local_files_only=True)
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    trainable = []
    if unfreeze:
        for block in [*backbone.encoder.layer[-unfreeze:], backbone.encoder.layernorm]:
            for parameter in block.parameters():
                parameter.requires_grad = True
                trainable.append(parameter)
    return Classifier(backbone), trainable


def collect(folders):
    """(window, labels, folder) for complete train-split labels."""
    rows = []
    for folder in folders:
        for window in read_json(folder / 'project.json')['windows']:
            labels = read_json(folder / 'labels' / f"{window['id']}.json", {})
            if window['split'] == 'train' and is_complete(labels):
                rows.append((folder, window, labels))
    return rows


def targets(window, labels, centers):
    """Per-center labels for every category: 1, 0 or -1 (unknown, masked out)."""
    return np.stack([grid_labels(window, labels, c, centers) for c in LABEL_CATEGORIES], 1)


def train(root, folders, unfreeze=0, epochs=8, batch=4, validation=.2, seed=7, device='cuda', backbone=None, log=print):
    import torch
    from sklearn.metrics import average_precision_score
    rows = collect(folders)
    if not rows:
        raise SystemExit('No complete train-split labels found.')
    titles = sorted({w['title'] for _, w, _ in rows})
    random.Random(seed).shuffle(titles)
    held = set(titles[:max(1, round(len(titles) * validation))])
    samples = {'fit': [], 'held': []}
    for folder, window, labels in rows:
        centers, clips = window_clips(folder, window)
        y = targets(window, labels, centers)
        keep = (y >= 0).any(1)
        for clip, label in zip(clips[keep], y[keep]):
            samples['held' if window['title'] in held else 'fit'].append((clip, label))
    fit_y = np.stack([y for _, y in samples['fit']])
    positives, negatives = (fit_y == 1).sum(0), (fit_y == 0).sum(0)
    model, trainable = build_model(root, unfreeze, backbone)
    model.to(device)
    weight = torch.tensor(np.clip(negatives / np.maximum(positives, 1), 1, 50), dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW([{'params': model.probe.parameters(), 'lr': 1e-3},
                                   {'params': trainable, 'lr': 1e-5}], weight_decay=.05)

    def run(split, learn):
        model.train(learn)
        order = list(range(len(samples[split])))
        if learn:
            random.Random(seed + epoch).shuffle(order)
        scores, labels = [], []
        for offset in range(0, len(order), batch):
            chosen = [samples[split][i] for i in order[offset:offset + batch]]
            frames = torch.from_numpy(np.stack([c for c, _ in chosen])).to(device)
            y = torch.from_numpy(np.stack([l for _, l in chosen])).to(device)
            with torch.set_grad_enabled(learn), torch.autocast(device_type='cuda' if device == 'cuda' else 'cpu',
                                                               dtype=torch.bfloat16, enabled=device == 'cuda'):
                logits = model(frames).float()
            mask = (y >= 0).float()
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(logits, y.clamp(min=0).float(), pos_weight=weight,
                                                                          reduction='none') * mask).sum() / mask.sum().clamp(min=1)
            if learn:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scores.append(logits.detach().cpu().numpy()); labels.append(y.cpu().numpy())
        return np.concatenate(scores), np.concatenate(labels)

    best = None
    for epoch in range(epochs):
        started = time.perf_counter()
        run('fit', True)
        scores, labels = run('held', False)
        ap = {c: float(average_precision_score(labels[:, k][labels[:, k] >= 0], scores[:, k][labels[:, k] >= 0]))
              for k, c in enumerate(LABEL_CATEGORIES) if (labels[:, k] == 1).any() and (labels[:, k] == 0).any()}
        mean_ap = float(np.mean(list(ap.values()))) if ap else 0.0
        log(f'epoch {epoch + 1}: held-out mean AP {mean_ap:.3f} {ap} ({time.perf_counter() - started:.0f}s)')
        if best is None or mean_ap > best[0]:
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                     if k.startswith('probe.') or any(k.startswith(f'backbone.encoder.layer.{i}.')
                                                      for i in range(len(model.backbone.encoder.layer) - unfreeze, len(model.backbone.encoder.layer)))
                     or (unfreeze and k.startswith('backbone.encoder.layernorm.'))}
            best = (mean_ap, epoch + 1, state, scores, labels, ap)
    mean_ap, epoch, state, scores, labels, ap = best
    operating = {}
    for k, category in enumerate(LABEL_CATEGORIES):
        known = labels[:, k] >= 0
        if (labels[known, k] == 1).sum() < 3:
            operating[category] = dict(skipped='fewer than 3 held-out positives')
            continue
        candidate, decision = thresholds(scores[known, k], labels[known, k])
        operating[category] = dict(candidate_threshold=candidate, decision_threshold=decision, held_out_ap=ap.get(category))
    return state, dict(unfreeze=unfreeze, epochs_run=epochs, best_epoch=epoch, held_out_titles=sorted(held),
                       train_titles=sorted(set(titles) - held), fit_samples=len(samples['fit']),
                       held_samples=len(samples['held']), positives=dict(zip(LABEL_CATEGORIES, positives.tolist())),
                       mean_held_out_ap=mean_ap, categories=operating)


def apply(root, folder, window, model, meta, device='cuda'):
    import torch
    centers, clips = window_clips(folder, window)
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=device == 'cuda'):
        logits = np.concatenate([model(torch.from_numpy(clips[i:i + 4]).to(device)).float().cpu().numpy()
                                 for i in range(0, len(clips), 4)])
    times = np.arange(GRID / 2, window['clip_duration'] - GRID / 2, GRID)
    nearest = np.abs(centers[None, :] - times[:, None]).argmin(1)
    entries, events = [], []
    for k, category in enumerate(LABEL_CATEGORIES):
        operating = meta['categories'][category]
        if 'candidate_threshold' not in operating:
            continue
        for t, score in zip(times, logits[nearest, k]):
            if score >= operating['candidate_threshold']:
                entries.append(dict(stage='classifier', category=category, start=float(t) - GRID / 2,
                                    end=float(t) + GRID / 2, sample_time=float(t), score=float(score), detail='fine-tuned'))
            if operating['decision_threshold'] is not None and score >= operating['decision_threshold']:
                events.append(dict(start=float(t) - GRID / 2, end=float(t) + GRID / 2, sample_time=float(t), category=category,
                                   confidence=1, severity=2, enabled=True, review_required=False, source='Fine-tuned V-JEPA 2',
                                   score_type='decision', note=f'Fine-tuned {category} score {score:.2f}'))
    return entries, events


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    fit = sub.add_parser('train')
    fit.add_argument('--projects', nargs='+', required=True); fit.add_argument('--model', required=True)
    fit.add_argument('--unfreeze', type=int, default=0, help='Also train the last N transformer blocks')
    fit.add_argument('--epochs', type=int, default=8)
    use = sub.add_parser('apply')
    use.add_argument('--name', required=True); use.add_argument('--base', required=True)
    use.add_argument('--model', required=True); use.add_argument('--run', required=True)
    args = parser.parse_args()
    root = prepare_environment()
    folder = root / 'finetuned'
    folder.mkdir(exist_ok=True)
    if args.command == 'train':
        state, meta = train(root, [project_dir(root, name) for name in args.projects], args.unfreeze, args.epochs)
        torch.save(state, folder / f'{args.model}.pt')
        meta.update(projects=args.projects, trained=time.strftime('%Y-%m-%dT%H:%M:%S'))
        (folder / f'{args.model}.json').write_text(json.dumps(meta, indent=1), encoding='utf-8')
        print(json.dumps({k: v for k, v in meta.items() if k != 'train_titles'}, indent=1))
        return
    meta = json.loads((folder / f'{args.model}.json').read_text(encoding='utf-8'))
    if args.name in meta['projects']:
        raise SystemExit('Apply a fine-tuned model to a project it was not trained on.')
    model, _ = build_model(root, meta['unfreeze'])
    model.load_state_dict(torch.load(folder / f'{args.model}.pt'), strict=False)
    model.cuda().eval()
    project = project_dir(root, args.name)
    run = derive_run(project, args.base, args.run, dict(finetuned=args.model))
    for window in read_json(project / 'project.json')['windows']:
        add_trace(run, window['id'], *apply(root, project, window, model, meta))
    print(f'Wrote {run}')


if __name__ == '__main__':
    main()
