"""Learned candidates and decisions layered on a prediction run: audio events and trained heads.

Heads train only on train-split labels; thresholds come from title-grouped
cross-validation, never from test windows. Each step writes a new prediction run
that evaluate_labels.py scores against the same labels as the base run.

    python learned.py audio --name test-1 --base rev8 --run rev8-audio
    python learned.py train --projects train-1 --heads heads-1
    python learned.py apply --name test-1 --base rev8-audio --heads heads-1 --run rev8-audio-heads-1
    python learned.py promote --heads heads-1 --name test-1 --base rev9 --candidate rev9-heads-1
"""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np

from features import load_features
from labeling import LABEL_CATEGORIES, is_complete, project_dir, read_json, write_json
from learned_runtime import GRID, feature_matrix, head_outputs, uncertain_spans, verified_events
from model_setup import prepare_environment
from provenance import FEATURE_SCHEMA, digest, file_hash, head_identity, qualification_identity

# AudioSet events that can point at violence. Sound is a candidate source only:
# films add gunfire to fireworks scenes and violence is often scored with music.
AUDIO_EVENTS = {
    'violence': ('Gunshot, gunfire', 'Machine gun', 'Fusillade', 'Artillery fire', 'Cap gun', 'Explosion',
                 'Slap, smack', 'Whack, thwack', 'Smash, crash', 'Screaming', 'Shatter', 'Thump, thud'),
}
CANDIDATE_RECALL = .97
DECISION_PRECISION = .95


# ---- shared ------------------------------------------------------------------
def derive_run(folder: Path, base: str, run: str, note: dict) -> Path:
    """Copy a prediction run so added candidates never alter the base run."""
    source, output = folder / 'predictions' / base, folder / 'predictions' / run
    if output.exists():
        raise FileExistsError(f'{output} already exists; choose a new run name')
    shutil.copytree(source, output)
    manifest = read_json(output / 'run.json')
    contract = dict(parent_identity=manifest.get('identity'), stage=note, derivation_code=file_hash(Path(__file__)))
    identity = digest(contract)
    write_json(output / 'run.json', {**manifest, **note, 'derived_from':base, 'pipeline':'cached-experiment',
                                    'identity':identity, 'derivation':contract, 'complete':False,
                                    'derived':time.strftime('%Y-%m-%dT%H:%M:%S')})
    for path in output.glob('*.json'):
        if path.name != 'run.json':
            prediction = read_json(path)
            write_json(path, dict(prediction, parent_run_identity=prediction.get('run_identity'), run_identity=identity))
    return output


def add_trace(run: Path, window_id: str, entries: list, events: list = ()):
    path = run / f'{window_id}.json'
    prediction = read_json(path)
    prediction['trace'] += entries
    prediction['events'] += list(events)
    write_json(path, prediction)


# ---- audio -------------------------------------------------------------------
def audio_candidates(features: dict, threshold: float) -> list:
    labels = [str(label) for label in features['labels']]
    entries = []
    for category, names in AUDIO_EVENTS.items():
        present = [name for name in names if name in labels]
        if not present:
            continue
        rows = features['probabilities'][:, [labels.index(name) for name in present]].astype(np.float32)
        for t, row in zip(features['times'], rows):
            if row.max() >= threshold:
                entries.append(dict(stage='audio', category=category, start=float(t) - 1, end=float(t) + 1,
                                    sample_time=float(t), score=float(row.max()), detail=present[int(row.argmax())]))
    return entries


# ---- heads -------------------------------------------------------------------
def grid_times(window: dict) -> np.ndarray:
    return np.arange(GRID / 2, window['clip_duration'] - GRID / 2, GRID)


def window_matrix(folder: Path, window: dict, kinds, allow_legacy=False, temporal_offsets=(), input_layout=None) -> tuple[np.ndarray, np.ndarray]:
    """Cached features at every grid time of a labeling window (see learned_runtime.feature_matrix)."""
    if input_layout=='player-chunks-v1':
        from player_inputs import load_matrix
        return load_matrix(folder,window,kinds,temporal_offsets)
    if input_layout not in (None,'legacy'):
        raise ValueError('Unknown learned input layout')
    times, features = grid_times(window), {}
    for kind in kinds:
        features[kind] = load_features(folder, kind, window['id'], allow_legacy=allow_legacy)
        if features[kind] is None:
            raise FileNotFoundError(f"Missing {kind} features for {window['id']}; run features.py first")
    return times, feature_matrix(features, times, kinds, temporal_offsets)


def grid_labels(window: dict, labels: dict, category: str, times: np.ndarray) -> np.ndarray:
    """1 positive, 0 reviewed clean, -1 unknown (outside target, ambiguous or unsure)."""
    y = np.full(len(times), -1)
    rule = labels['categories'][category]
    if rule['status'] != 'reviewed':
        return y
    low, high = window['target']
    inside = (times >= low) & (times < high)
    y[inside] = 0
    for a, b in rule['intervals']:
        y[inside & (times >= a) & (times < b)] = 1
    for a, b in rule['ambiguous']:
        y[(times >= a) & (times < b)] = -1
    return y


def thresholds(scores, y, minimum_hits=10):
    """Operating points from out-of-fold scores.

    Candidate: the highest score that still reaches the candidate recall target.
    Decision: the lowest score where precision of everything above it reaches the
    decision target with at least `minimum_hits` true positives; None otherwise,
    in which case the head only proposes and never decides.
    """
    scores, y = np.asarray(scores), np.asarray(y)
    if scores.ndim != 1 or scores.shape != y.shape or not len(y) or not np.isfinite(scores).all():
        raise ValueError('Thresholds need matching, nonempty finite scores and binary labels')
    if not np.isin(y, [0, 1]).all():
        raise ValueError('Threshold labels must be binary')
    order = np.argsort(-scores, kind='stable')
    # Evaluate only complete tied-score groups: deployment uses score >= threshold.
    ordered = scores[order]
    boundaries = np.r_[np.flatnonzero(ordered[:-1] != ordered[1:]), len(y)-1]
    hits = np.cumsum(y[order])[boundaries]
    ordered = ordered[boundaries]
    recall = hits / max(1, y.sum())
    precision = hits / (boundaries + 1)
    candidate = float(ordered[min(int(np.searchsorted(recall, CANDIDATE_RECALL)), len(ordered) - 1)])
    reliable = np.where((precision >= DECISION_PRECISION) & (hits >= minimum_hits))[0]
    return candidate, float(ordered[reliable[-1]]) if len(reliable) else None


def title_class_weights(y, groups):
    """Give each training title equal influence within each known class."""
    weights=np.ones(len(y),dtype=np.float64)
    for label in (0,1):
        selected=y==label
        titles=set(groups[selected])
        for title in titles:
            matching=selected & (groups==title)
            weights[matching]=selected.sum()/len(titles)/matching.sum()
    return weights


def train(roots, kinds, minimum_titles=3, allow_legacy=False, temporal_offsets=(), categories=LABEL_CATEGORIES,
          balance_titles=False, input_layout=None, include_titles=None):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from select_windows import normalize_title
    rows = []
    for folder in roots:
        project = read_json(folder / 'project.json')
        for window in project['windows']:
            labels = read_json(folder / 'labels' / f"{window['id']}.json", {})
            if window['split'] != 'train' or not is_complete(labels):
                continue
            if include_titles is not None and normalize_title(window['title']) not in include_titles:
                continue
            times, matrix = window_matrix(folder, window, kinds, allow_legacy, temporal_offsets, input_layout)
            rows.append((window, labels, times, matrix))
    if not rows:
        raise SystemExit('No complete train-split labels found.')
    heads = {}
    for category in categories:
        X, y, groups = [], [], []
        for window, labels, times, matrix in rows:
            target = grid_labels(window, labels, category, times)
            keep = target >= 0
            X.append(matrix[keep]); y.append(target[keep]); groups += [normalize_title(window['title'])] * int(keep.sum())
        X, y, groups = np.concatenate(X), np.concatenate(y), np.array(groups)
        titles = len(set(groups[y == 1]))
        if titles < minimum_titles:
            heads[category] = dict(skipped=f'positives from {titles} titles; need {minimum_titles}', positives=int(y.sum()))
            continue
        folds = GroupKFold(n_splits=min(5, len(set(groups))))
        best = None
        for C in (.003, .01, .03, .1):
            oof = np.zeros(len(y))
            for fit, held in folds.split(X, y, groups):
                if len(np.unique(y[fit])) < 2:
                    oof[held] = -100 if not y[fit].sum() else 100
                    continue
                weights=title_class_weights(y[fit],groups[fit]) if balance_titles else None
                model = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight='balanced', max_iter=3000)).fit(
                    X[fit], y[fit],logisticregression__sample_weight=weights)
                oof[held] = model.decision_function(X[held])
            ap = average_precision_score(y, oof)
            if best is None or ap > best[0]:
                best = (ap, C, oof)
        ap, C, oof = best
        candidate, decision = thresholds(oof, y)
        weights=title_class_weights(y,groups) if balance_titles else None
        fitted = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight='balanced', max_iter=3000)).fit(
            X, y,logisticregression__sample_weight=weights)
        scaler, model = fitted.steps[0][1], fitted.steps[1][1]
        mean, scale = scaler.mean_, scaler.scale_
        heads[category] = dict(C=C, cv_average_precision=round(float(ap), 4), positives=int(y.sum()), samples=int(len(y)),
                               positive_titles=titles, candidate_threshold=candidate, decision_threshold=decision,
                               mean=mean.tolist(), scale=scale.tolist(), coef=model.coef_[0].tolist(),
                               intercept=float(model.intercept_[0]))
    result = dict(kinds=list(kinds), grid=GRID, temporal_offsets=list(temporal_offsets), feature_schema=FEATURE_SCHEMA, windows=len(rows), titles=sorted({w['title'] for w, *_ in rows}),
                balance_titles=balance_titles,
                candidate_recall_target=CANDIDATE_RECALL, decision_precision_target=DECISION_PRECISION, heads=heads)
    if input_layout:result['input_layout']=input_layout
    return result


def apply(folder, window, heads, allow_legacy=False):
    if heads.get('attention_model'):
        raise ValueError('Attentive heads must be evaluated through predict_project.py, not legacy pooled features')
    times, matrix = window_matrix(folder, window, heads['kinds'], allow_legacy, heads.get('temporal_offsets', ()),heads.get('input_layout'))
    return head_outputs(heads, times, matrix)


def verify(run: Path, window: dict, clip: Path, verifier, burst=None):
    """One five-frame evidence check per uncertain span, at its middle."""
    from analysis_engine import sample_burst
    burst = burst or sample_burst

    def check(category, middle):
        frames, center = burst(clip, middle)
        return verifier.check_frame(frames, category, '', center_index=center)

    entries, events = verified_events(uncertain_spans(read_json(run / f"{window['id']}.json")), check, *window['target'])
    add_trace(run, window['id'], entries, events)
    return events


def promotion_check(base: dict, candidate: dict, tolerance=.05, minimum_precision=.95, minimum_recall=.95,
                    minimum_events=20, minimum_titles=5, maximum_false_alarm=.02, minimum_clean_windows=100):
    """Whether a candidate run may replace the base in the player, on the same labels.

    It must raise recall in at least one category and may not lower precision or
    recall by more than `tolerance` anywhere. Reports come from
    evaluate_labels.evaluate with identical labels and padding.
    """
    rows, improved, regressed = [], False, False
    if not base.get('benchmark_identity') or base.get('benchmark_identity') != candidate.get('benchmark_identity'):
        return False, [dict(category='benchmark', reason='Benchmark identities must match')]
    labelers = candidate.get('labelers', {})
    human = bool(labelers) and all(k.lower() == 'human' for k in labelers) and not candidate.get('drafts_shown', 0)
    for category in LABEL_CATEGORIES:
        b, c = base['categories'][category].get('all'), candidate['categories'][category].get('all')
        if not b or not c:
            continue
        reasons = []
        if (c['recall'] or 0) > (b['recall'] or 0) + 1e-9:
            improved = True
            bounds = candidate['categories'][category].get('title_bootstrap_95', {})
            if not human:
                reasons.append('independent human labels required')
            if (c['precision'] or 0) < minimum_precision or (c['recall'] or 0) < minimum_recall:
                reasons.append('precision and recall must each reach 95%')
            if c['counts'].get('events', 0) < minimum_events or candidate['categories'][category].get('positive_titles', 0) < minimum_titles:
                reasons.append('insufficient independent events/titles')
            if c['counts'].get('clean_windows', 0) < minimum_clean_windows or candidate['categories'][category].get('clean_titles', 0) < minimum_titles:
                reasons.append('insufficient clean windows/titles')
            if any(bounds.get(k, [0])[0] < .90 for k in ('precision', 'recall')):
                reasons.append('title confidence bounds below 90%')
        if (c.get('clean_window_false_alarm') or 0) > maximum_false_alarm:
            reasons.append('clean-window false alarms above budget')
        if not b['counts']['gold'] and c['counts'].get('fp', 0) > b['counts'].get('fp', 0) + 1e-9:
            reasons.append('added false positives in a category without gold positives')
        rows.append(dict(category=category, base_precision=b['precision'], precision=c['precision'],
                         base_recall=b['recall'], recall=c['recall'], reasons=reasons))
        if reasons:
            regressed = True
        if (b['recall'] or 0) - (c['recall'] or 0) > tolerance:
            regressed = True
        if (b['precision'] is not None and (c['precision'] or 0) < b['precision'] - tolerance) or (b['precision'] is None and c['precision'] is not None and c['precision'] < minimum_precision):
            regressed = True
    return improved and not regressed, rows


def category_choice(base: dict, candidate: dict, tolerance=.05):
    """Categories whose head improves on the base without losing more than `tolerance`.

    Decided on validation windows. A category qualifies when its recall rises
    and neither recall nor precision drops by more than `tolerance`; categories
    without labeled positives in the base report stay off.
    """
    enabled, rows = [], []
    for category in LABEL_CATEGORIES:
        b, c = base['categories'][category].get('all'), candidate['categories'][category].get('all')
        if not b or not c or not b['counts']['gold']:
            rows.append(dict(category=category, enabled=False, reason='no labeled positives'))
            continue
        gain = (c['recall'] or 0) - (b['recall'] or 0)
        precision_loss = (b['precision'] - (c['precision'] or 0)) if b['precision'] is not None else 0
        ok = gain > 1e-9 and precision_loss <= tolerance and (c['precision'] or 0) >= DECISION_PRECISION
        rows.append(dict(category=category, enabled=ok, base_precision=b['precision'], precision=c['precision'],
                         base_recall=b['recall'], recall=c['recall']))
        if ok:
            enabled.append(category)
    return enabled, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    audio = sub.add_parser('audio')
    audio.add_argument('--name', required=True); audio.add_argument('--base', required=True); audio.add_argument('--run', required=True)
    audio.add_argument('--threshold', type=float, default=.15)
    fit = sub.add_parser('train')
    fit.add_argument('--projects', nargs='+', required=True); fit.add_argument('--heads', required=True)
    fit.add_argument('--kinds', nargs='+', default=['clip', 'vjepa', 'ast'])
    fit.add_argument('--allow-legacy-features', action='store_true', help='Use frozen unversioned features for an experiment only')
    fit.add_argument('--temporal-offsets', nargs='*', type=float, default=[], help='Ordered motion context offsets, e.g. -.5 .5; no extra model calls')
    fit.add_argument('--categories', nargs='+', choices=LABEL_CATEGORIES, default=list(LABEL_CATEGORIES))
    fit.add_argument('--balance-titles',action='store_true',help='Equal training influence for each title within each class')
    fit.add_argument('--input-layout',choices=['legacy','player-chunks-v1'],default=None,
                     help='Train on provenance-backed exact-player matrices prepared with player_inputs.py')
    use = sub.add_parser('apply')
    use.add_argument('--name', required=True); use.add_argument('--base', required=True)
    use.add_argument('--heads', required=True); use.add_argument('--run', required=True)
    use.add_argument('--allow-legacy-features', action='store_true')
    check = sub.add_parser('verify', help='Scene-model evidence checks on uncertain learned/audio candidates')
    check.add_argument('--name', required=True); check.add_argument('--base', required=True); check.add_argument('--run', required=True)
    check.add_argument('--scene-model', default='4B', choices=['4B', '8B'])
    choose = sub.add_parser('select', help='Enable only the categories that improve on validation windows')
    choose.add_argument('--heads', required=True); choose.add_argument('--name', nargs='+', required=True)
    choose.add_argument('--base', required=True); choose.add_argument('--candidate', required=True)
    choose.add_argument('--split', default='val')
    promote = sub.add_parser('promote', help='Allow a head set in the player after a labeled comparison')
    promote.add_argument('--heads', required=True); promote.add_argument('--name', nargs='+', required=True)
    promote.add_argument('--base', required=True); promote.add_argument('--candidate', required=True)
    promote.add_argument('--split', default='test')
    promote.add_argument('--force', metavar='REASON', help='Make available for review despite failing qualification; never enables automatic decisions')
    args = parser.parse_args()
    root = prepare_environment()

    if args.command == 'audio':
        folder = project_dir(root, args.name)
        run = derive_run(folder, args.base, args.run, dict(audio_threshold=args.threshold, audio_events=AUDIO_EVENTS))
        for window in read_json(folder / 'project.json')['windows']:
            features = load_features(folder, 'ast', window['id'])
            if features is None:
                raise SystemExit(f"Missing audio features for {window['id']}; run features.py --kind ast")
            add_trace(run, window['id'], audio_candidates(features, args.threshold))
        print(f'Wrote {run}')

    elif args.command == 'train':
        if any(not np.isfinite(x) or not -1 <= x <= 1 for x in args.temporal_offsets):
            raise SystemExit('Temporal offsets must be finite and within ±1 second (the player context)')
        heads = train([project_dir(root, name) for name in args.projects], args.kinds, allow_legacy=args.allow_legacy_features,
                      temporal_offsets=sorted(set(args.temporal_offsets)-{0}), categories=args.categories,balance_titles=args.balance_titles,
                      input_layout=args.input_layout)
        heads['legacy_features'] = args.allow_legacy_features
        path = root / 'heads' / f'{args.heads}.json'
        if path.exists():
            raise FileExistsError(f'{path} exists; choose a new head name')
        write_json(path, dict(heads, trained=time.strftime('%Y-%m-%dT%H:%M:%S'), projects=args.projects))
        for category, head in heads['heads'].items():
            print(category, {k: v for k, v in head.items() if k not in ('mean', 'scale', 'coef')})
        print(f'Wrote {path}')
    elif args.command == 'select':
        from evaluate_labels import evaluate, load_projects
        project, labels, predictions = load_projects(root, args.name, [args.base, args.candidate])
        reports = {run: evaluate(project, labels, predictions[run], args.split, padding=True)
                   for run in (args.base, args.candidate)}
        enabled, rows = category_choice(reports[args.base], reports[args.candidate])
        fmt = lambda v: '—' if v is None else f'{v:.0%}'
        print(f"{reports[args.candidate]['windows']} labeled {args.split} windows")
        for row in rows:
            detail = row.get('reason') or (f"precision {fmt(row['base_precision'])} -> {fmt(row['precision'])}   "
                                          f"recall {fmt(row['base_recall'])} -> {fmt(row['recall'])}")
            print(f"{row['category']:16} {'ON ' if row['enabled'] else 'off'}  {detail}")
        path = root / 'heads' / f'{args.heads}.json'
        heads = read_json(path)
        if args.name and set(args.name) & set(heads.get('projects', [])):
            raise SystemExit('Choose categories on projects the heads were not trained on.')
        heads.update(enabled=enabled, selection=dict(date=time.strftime('%Y-%m-%dT%H:%M:%S'), projects=args.name,
                                                     split=args.split, base=args.base, candidate=args.candidate, rows=rows))
        heads.pop('promoted', None)  # a changed head set needs a fresh test comparison
        write_json(path, heads)
        print(f'Enabled {enabled or "no categories"} in {args.heads}')
    elif args.command == 'promote':
        from evaluate_labels import evaluate, load_projects
        project, labels, predictions = load_projects(root, args.name, [args.base, args.candidate])
        reports = {run: evaluate(project, labels, predictions[run], args.split, padding=True)
                   for run in (args.base, args.candidate)}
        passed, rows = promotion_check(reports[args.base], reports[args.candidate])
        fmt = lambda v: '—' if v is None else f'{v:.0%}'
        print(f"{reports[args.candidate]['windows']} labeled {args.split} windows")
        for row in rows:
            print(f"{row['category']:16} precision {fmt(row.get('base_precision'))} -> {fmt(row.get('precision'))}   "
                  f"recall {fmt(row.get('base_recall'))} -> {fmt(row.get('recall'))}  "
                  f"{row.get('reason') or '; '.join(row.get('reasons', []))}")
        passed = passed and args.split == 'test'
        if not reports[args.candidate]['windows'] or not (passed or args.force):
            raise SystemExit('Not qualified: complete independent human test labels, ≥95% precision/recall, event/title support, confidence bounds and clean false-alarm limits are required. --force permits review only.')
        path = root / 'heads' / f'{args.heads}.json'
        heads = read_json(path)
        identities, verification, qualified_assets = [], set(), set()
        for name in args.name:
            manifest = read_json(project_dir(root, name) / 'predictions' / args.candidate / 'run.json')
            identity = manifest.get('head_identity')
            if (identity != head_identity(heads) or manifest.get('pipeline') != 'player'
                    or not manifest.get('complete') or not manifest.get('identity') or manifest.get('benchmark_only')):
                raise SystemExit('Promotion requires a player-pipeline run of this exact head content; rerun predict_project.py.')
            identities.append(manifest.get('identity'))
            qualified_assets.add(qualification_identity(manifest['pipeline_identity']))
            verification.add(manifest.get('config', {}).get('learned_verification', 'off'))
        if len(verification) != 1 or len(qualified_assets) != 1:
            raise SystemExit('Candidate projects used different verification settings')
        from select_windows import normalize_title
        training_titles = {normalize_title(t) for t in heads.get('titles', [])}
        if any(normalize_title(w['title']) in training_titles for w in project['windows'] if w['split'] == args.split):
            raise SystemExit('Promotion titles overlap training titles')
        heads['promoted'] = dict(date=time.strftime('%Y-%m-%dT%H:%M:%S'), projects=args.name, split=args.split, base=args.base,
                                 candidate=args.candidate, comparison=rows, passed=passed, qualified=passed, forced=args.force,
                                 head_identity=head_identity(heads), verification=verification.pop(), run_identities=identities,
                                 qualification_identity=qualified_assets.pop(),
                                 benchmark_identity=reports[args.candidate]['benchmark_identity'])
        write_json(path, heads)
        print(f'{args.heads}: {"qualified for automatic filtering" if passed else "available for review only"}.')
    elif args.command == 'verify':
        from scene_verifier import SceneVerifier
        folder = project_dir(root, args.name)
        run = derive_run(folder, args.base, args.run, dict(selective_verification=args.scene_model))
        verifier = SceneVerifier(root, model_size=args.scene_model)
        try:
            checked = 0
            for window in read_json(folder / 'project.json')['windows']:
                checked += len(verify(run, window, folder / 'clips' / f"{window['id']}.mp4", verifier))
        finally:
            verifier.close()
        print(f'Wrote {run}: {checked} candidate spans confirmed or left for review')
    else:
        folder = project_dir(root, args.name)
        heads = read_json(root / 'heads' / f'{args.heads}.json')
        if args.name in heads['projects']:
            raise SystemExit('Apply heads to a project they were not trained on.')
        run = derive_run(folder, args.base, args.run, dict(heads=args.heads, head_identity=head_identity(heads), pipeline='cached-experiment'))
        for window in read_json(folder / 'project.json')['windows']:
            add_trace(run, window['id'], *apply(folder, window, heads, args.allow_legacy_features))
        print(f'Wrote {run}')

    if args.command in ('audio', 'apply', 'verify'):
        manifest=read_json(run/'run.json')
        write_json(run/'run.json',dict(manifest,complete=True))


if __name__ == '__main__':
    main()
