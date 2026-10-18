"""Score prediction runs against human labels: candidate recall, final precision/recall, events.

Only complete labels count. 'Unsure' categories and ambiguous spans are excluded.
Confidence intervals resample whole titles, because windows from one title are
correlated. Random windows estimate real-world rates; mined windows are enriched
with what the miners already find, so their recall is optimistic and reported apart.

    python evaluate_labels.py --name test-1 --run rev7-baseline --split test
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import sys

from core import compile_plan, default_policy
from evaluate import intersect, length, union
from labeling import LABEL_CATEGORIES, is_complete, project_dir, read_json, validate_labels
from model_setup import prepare_environment
from provenance import digest

# First-stage proposals. 'scene_evidence' is the second stage and is excluded.
CANDIDATE_STAGES = ('anatomy_raw', 'scene_sequence', 'clip', 'audio', 'classifier')
COUNTS = ('gold', 'tp', 'fp', 'fn', 'assisted_tp', 'assisted_fp', 'candidate_tp',
          'events', 'events_hit', 'events_covered', 'clean_windows', 'clean_windows_filtered', 'reviewed_seconds')


def difference(a, b):
    """Intervals of a not covered by b; both unions."""
    result = []
    for start, end in a:
        cursor = start
        for x, y in b:
            if y <= cursor or x >= end:
                continue
            if x > cursor:
                result.append([cursor, x])
            cursor = max(cursor, y)
        if cursor < end:
            result.append([cursor, end])
    return result


def isolate(category, padding):
    policy = default_policy()
    for name, rule in policy['categories'].items():
        rule.update(visual='hide' if name == category else 'allow', audio='allow')
    if not padding:
        policy.update(pre_padding=0, post_padding=0)
    return policy


def filtered(events, category, duration, padding, assisted=False):
    chosen = [dict(e, confirmed=True) if assisted else e for e in events if e['category'] == category]
    return union((p['start'], p['end']) for p in compile_plan(chosen, isolate(category, padding), duration))


def candidates(trace, category, stage=None):
    thresholds = {c: r['min_confidence'] for c, r in default_policy()['categories'].items()}
    keep = []
    for entry in trace:
        if entry['category'] != category or entry['stage'] not in CANDIDATE_STAGES:
            continue
        if stage and entry['stage'] != stage:
            continue
        # Similarity scores are continuous; only those above the operating threshold are proposals.
        if entry['stage'] == 'clip' and entry.get('score', 0) < thresholds[category]:
            continue
        keep.append((entry['start'], entry['end']))
    return union(keep)


def score_window(window, labels, prediction, category, padding):
    rule = labels['categories'][category]
    if rule['status'] != 'reviewed':
        return None
    scored = difference([list(window['target'])], union(map(tuple, rule['ambiguous'])))
    gold = intersect(union(map(tuple, rule['intervals'])), scored)
    duration = window['clip_duration']
    final = intersect(filtered(prediction['events'], category, duration, padding), scored)
    assisted = intersect(filtered(prediction['events'], category, duration, padding, True), scored)
    proposed = intersect(candidates(prediction['trace'], category), scored)
    counts = dict(reviewed_seconds=length(scored), gold=length(gold), tp=length(intersect(final, gold)), fp=length(difference(final, gold)),
                  fn=length(difference(gold, final)), assisted_tp=length(intersect(assisted, gold)),
                  assisted_fp=length(difference(assisted, gold)), candidate_tp=length(intersect(proposed, gold)),
                  events=len(gold), events_hit=sum(length(intersect(final, [g])) > .04 for g in gold),
                  events_covered=sum(length(intersect(final, [g])) >= .95 * (g[1] - g[0]) for g in gold),
                  clean_windows=int(not gold), clean_windows_filtered=int(not gold and length(final) > .04))
    for stage in CANDIDATE_STAGES:
        counts[f'candidate_tp_{stage}'] = length(intersect(intersect(candidates(prediction['trace'], category, stage), scored), gold))
    return counts


def rates(c):
    ratio = lambda a, b: a / b if b > 1e-9 else None
    result = dict(precision=ratio(c['tp'], c['tp'] + c['fp']), recall=ratio(c['tp'], c['gold']),
                  assisted_precision=ratio(c['assisted_tp'], c['assisted_tp'] + c['assisted_fp']),
                  assisted_recall=ratio(c['assisted_tp'], c['gold']), candidate_recall=ratio(c['candidate_tp'], c['gold']),
                  event_hit_recall=ratio(c['events_hit'], c['events']), event_covered_recall=ratio(c['events_covered'], c['events']),
                  clean_window_false_alarm=ratio(c['clean_windows_filtered'], c['clean_windows']))
    result['false_filter_seconds_per_hour'] = ratio(c['fp']*3600, c.get('reviewed_seconds',0))
    result.update({f'candidate_recall_{s}': ratio(c.get(f'candidate_tp_{s}', 0), c['gold']) for s in CANDIDATE_STAGES
                   if c.get(f'candidate_tp_{s}', 0) > 0})
    return result


def bootstrap(by_title, keys=('precision', 'recall', 'candidate_recall'), samples=2000, seed=7):
    titles = sorted(by_title)
    if len(titles) < 2:
        return {}
    rng, draws = random.Random(seed), defaultdict(list)
    for _ in range(samples):
        total = defaultdict(float)
        for title in rng.choices(titles, k=len(titles)):
            for k, v in by_title[title].items():
                total[k] += v
        for key, value in rates(total).items():
            if key in keys and value is not None:
                draws[key].append(value)
    return {k: [round(sorted(v)[int(.025 * len(v))], 3), round(sorted(v)[int(.975 * len(v)) - 1], 3)]
            for k, v in draws.items() if len(v) > samples / 2}


def evaluate(project, labels, predictions, split=None, padding=True):
    selected = [w for w in project['windows'] if not split or w['split'] == split]
    if len({w['id'] for w in project['windows']}) != len(project['windows']):
        raise ValueError('Duplicate window IDs in project')
    windows = [w for w in selected if is_complete(labels.get(w['id']))]
    missing = [w['id'] for w in windows if w['id'] not in predictions]
    if missing:
        raise ValueError(f'Missing predictions for {len(missing)} complete labeled windows: {missing[:8]}')
    for window in windows:
        validate_prediction(window, predictions[window['id']])
        validate_labels(window, labels[window['id']])
    labelers = [labels[w['id']].get('labeler', 'unknown') for w in windows]
    report = dict(windows=len(windows), split=split, drafts_shown=sum(labels[w['id']]['drafts_shown'] for w in windows),
                  coverage=dict(expected=len(windows), predicted=len(windows), incomplete_labels=len(selected)-len(windows)),
                  benchmark_identity=digest([(w, labels[w['id']]) for w in windows]),
                  labelers={name: labelers.count(name) for name in sorted(set(labelers))},
                  labeled_seconds=round(sum(w['target'][1] - w['target'][0] for w in windows), 1),
                  padding='balanced' if padding else 'none', categories={})
    for category in LABEL_CATEGORIES:
        strata, titles = defaultdict(lambda: defaultdict(float)), defaultdict(lambda: defaultdict(float))
        unknown, positive_titles, clean_titles = 0, set(), set()
        for window in windows:
            counts = score_window(window, labels[window['id']], predictions[window['id']], category, padding)
            if counts is None:
                unknown += 1
                continue
            if counts['gold'] > 0:
                positive_titles.add(window['title'])
            else:
                clean_titles.add(window['title'])
            stratum = 'random' if window['stratum'] == 'random' else 'mined' if window['stratum'].startswith('mined') else window['stratum']
            for key, value in counts.items():
                for bucket in (strata['all'], strata[stratum], titles[window['title']]):
                    bucket[key] += value
        report['categories'][category] = dict(
            positive_titles=len(positive_titles),
            clean_titles=len(clean_titles),
            unsure_windows=unknown,
            **{name: dict(counts={k: round(v, 3) for k, v in c.items() if k in COUNTS}, **rates(c))
               for name, c in strata.items()},
            title_bootstrap_95=bootstrap(titles))
    return report


def validate_prediction(window, prediction):
    """Reject corrupt, mismatched and manually assisted model benchmark inputs."""
    from core import CATEGORIES, validate_events
    if not isinstance(prediction, dict) or not isinstance(prediction.get('events'), list) or not isinstance(prediction.get('trace'), list):
        raise ValueError(f"Invalid prediction schema for {window['id']}")
    if prediction.get('window', window['id']).split('/')[-1] != window['id'].split('/')[-1]:
        raise ValueError(f"Prediction belongs to another window: {window['id']}")
    validate_events(prediction['events'], window['clip_duration'])
    for e in prediction['events']:
        if e.get('confirmed') or e.get('source') == 'manual':
            raise ValueError('Manual confirmations must not enter model-only metrics')
    for entry in prediction['trace']:
        if entry.get('category') not in CATEGORIES or not isinstance(entry.get('stage'), str):
            raise ValueError('Invalid trace category or stage')
        a, b = entry.get('start'), entry.get('end')
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (a, b)) or a >= b:
            raise ValueError('Trace intervals must be finite and ordered')
        # Candidate context may extend slightly beyond the clip; scoring clips it.
        if b <= 0 or a >= window['clip_duration']:
            raise ValueError('Trace lies entirely outside its clip')
        if 'score' in entry and not math.isfinite(entry['score']):
            raise ValueError('Trace score must be finite')


def table(report):
    fmt = lambda v: '—' if v is None else f'{v:.0%}'
    lines = [f"{report['windows']} windows, {report['labeled_seconds']}s, padding {report['padding']}, "
             f"drafts seen on {report['drafts_shown']}, labels by {report['labelers']}",
             '| Category | Gold s | Candidate recall | Precision | Recall | Assisted recall | Events hit | Clean windows falsely filtered |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for category, result in report['categories'].items():
        a = result.get('all')
        if not a:
            lines.append(f'| {category} | — | | | | | | |')
            continue
        ci = result['title_bootstrap_95'].get('recall')
        lines.append(f"| {category} | {a['counts']['gold']:.1f} | {fmt(a['candidate_recall'])} | {fmt(a['precision'])} | "
                     f"{fmt(a['recall'])}{f' ({ci[0]:.0%}–{ci[1]:.0%})' if ci else ''} | {fmt(a['assisted_recall'])} | "
                     f"{a['counts']['events_hit']:.0f}/{a['counts']['events']:.0f} | "
                     f"{a['counts']['clean_windows_filtered']:.0f}/{a['counts']['clean_windows']:.0f} |")
    return '\n'.join(lines)


def load_projects(root, names, runs):
    """Windows, labels and predictions from several projects, merged under name-prefixed ids."""
    windows, labels, predictions = [], {}, {run: {} for run in runs}
    if len(set(names)) != len(names):
        raise ValueError('Duplicate project names')
    for name in names:
        folder = project_dir(root, name)
        for window in read_json(folder / 'project.json')['windows']:
            key = f"{name}/{window['id']}"
            windows.append(dict(window, id=key))
            labels[key] = read_json(folder / 'labels' / f"{window['id']}.json", {})
            for run in runs:
                path = folder / 'predictions' / run / f"{window['id']}.json"
                if path.is_file():
                    prediction = read_json(path)
                    manifest = read_json(path.parent/'run.json', {})
                    if manifest.get('complete') is False:
                        raise ValueError(f'Incomplete prediction run: {path.parent}')
                    if manifest.get('identity') and prediction.get('run_identity') != manifest['identity']:
                        raise ValueError(f'Prediction/run identity mismatch: {path}')
                    predictions[run][key] = prediction
    return dict(windows=windows), labels, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--name', nargs='+', required=True, help='One or more labeling projects, pooled')
    parser.add_argument('--run', nargs='+', required=True)
    parser.add_argument('--split', default='test')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding='utf-8')
    root = prepare_environment()
    project, labels, predictions = load_projects(root, args.name, args.run)
    folder = project_dir(root, args.name[0])
    results = {}
    for run in args.run:
        missing = [n for n in args.name if not (project_dir(root, n) / 'predictions' / run).is_dir()]
        if missing:
            raise SystemExit(f'Run {run} is missing in {missing}')
        results[run] = dict(projects=args.name,
                            **{mode: evaluate(project, labels, predictions[run], args.split, padding)
                               for mode, padding in (('balanced', True), ('no_padding', False))})
        print(f'\n## {run}\n' + table(results[run]['balanced']))
    output = args.output or folder / 'reports' / f"{'+'.join(args.name)}-{'+'.join(args.run)}-{args.split}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=1), encoding='utf-8')
    print(f'\nFull report: {output}')


if __name__ == '__main__':
    main()
