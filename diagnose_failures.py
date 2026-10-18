"""Locate missed seconds in frozen prediction traces, without rerunning inference.

These are temporal diagnostics. A proposal at the right time can still point at
the wrong object; this report does not prove that a later check made a mistake.
"""
import argparse
from collections import defaultdict
import math
from pathlib import Path

from evaluate import intersect, length, union
from evaluate_labels import candidates, difference, filtered, validate_prediction
from labeling import LABEL_CATEGORIES, is_complete, project_dir, read_json, validate_labels, write_json
from model_setup import prepare_environment
from provenance import digest, file_hash


def diagnose_window(window, labels, prediction, category, padding=True):
    validate_prediction(window, prediction)
    validate_labels(window, labels)
    rule = labels['categories'][category]
    if rule['status'] != 'reviewed':
        return None
    scored = difference([list(window['target'])], union(rule['ambiguous']))
    gold = intersect(union(rule['intervals']), scored)
    final = intersect(filtered(prediction['events'], category, window['clip_duration'], padding), scored)
    assisted = filtered(prediction['events'], category, window['clip_duration'], padding, assisted=True)
    missed = difference(gold, final)
    reviewable = intersect(missed, assisted)
    proposed = intersect(missed, candidates(prediction['trace'], category))
    buckets = dict(reviewable=reviewable,
                   proposed_without_review=difference(proposed, reviewable),
                   no_recorded_support=difference(missed, union(proposed+reviewable)))
    if not math.isclose(sum(length(v) for v in buckets.values()), length(missed), abs_tol=1e-8):
        raise RuntimeError('Missed-second partition is not conserved')
    result = dict(scored_seconds=length(scored), gold_seconds=length(gold),
                  caught_seconds=length(intersect(gold, final)), missed_seconds=length(missed),
                  false_seconds=length(difference(final, gold)),
                  missed=missed, false=difference(final, gold),
                  miss_buckets={k: dict(seconds=length(v), intervals=v) for k,v in buckets.items()})
    if category == 'nudity':
        raw = candidates(prediction['trace'], category, 'anatomy_raw')
        strong = union((t['start'], t['end']) for t in prediction['trace']
                       if t['category']==category and t['stage']=='anatomy_raw' and t.get('score',0)>=.55)
        strong = intersect(missed, strong)
        weak = difference(intersect(missed, raw), strong)
        result['raw_proposal_overlap'] = {
            k: dict(seconds=length(v), intervals=v) for k,v in
            dict(score_at_least_055=strong, lower_score=weak,
                 no_raw_proposal=difference(missed, raw)).items()}
    return result


def load_frozen(root, names, prediction_dir=None, run=None):
    if len(set(names)) != len(names):
        raise ValueError('Duplicate project names')
    shared = read_json(prediction_dir/'run.json') if prediction_dir is not None else None
    if shared is not None:
        if not shared.get('complete') or not shared.get('identity'):
            raise ValueError('Prediction run is incomplete or lacks provenance')
        if shared.get('identity') != digest(shared.get('project_contracts')):
            raise ValueError('Invalid paired-run identity')
    records = []
    manifests = {}
    for name in names:
        folder = project_dir(root, name)
        project = read_json(folder/'project.json')
        directory = prediction_dir/name if shared is not None else folder/'predictions'/run
        manifest = shared if shared is not None else read_json(directory/'run.json')
        if not manifest.get('complete') or not manifest.get('identity'):
            raise ValueError(f'Incomplete or unversioned run for {name}')
        contract = manifest['project_contracts'].get(name,{}) if shared is not None else manifest
        if contract.get('project_identity') != digest(project):
            raise ValueError(f'Project changed since prediction: {name}')
        manifests[name] = dict(identity=manifest['identity'], config=contract.get('config'),
                               pipeline=contract.get('pipeline'), benchmark_only=manifest.get('benchmark_only',False))
        if len({w['id'] for w in project['windows']}) != len(project['windows']):
            raise ValueError('Duplicate window IDs')
        for window in project['windows']:
            labels = read_json(folder/'labels'/f"{window['id']}.json", {})
            if not is_complete(labels):
                raise ValueError(f"Incomplete labels: {name}/{window['id']}")
            if window['id'] not in contract.get('expected_windows',[]):
                raise ValueError(f"Window absent from run contract: {name}/{window['id']}")
            source = folder/'clips'/f"{window['id']}.mp4"
            if not source.is_file() or contract.get('sources',{}).get(window['id']) != file_hash(source):
                raise ValueError(f"Source changed since prediction: {name}/{window['id']}")
            prediction = read_json(directory/f"{window['id']}.json")
            if prediction.get('run_identity') != manifest['identity']:
                raise ValueError(f"Prediction/run identity mismatch: {name}/{window['id']}")
            records.append((name, window, labels, prediction))
    return records, manifests


def report(records, manifests, categories, padding=True):
    if len(set(categories)) != len(categories):
        raise ValueError('Duplicate categories')
    identities = [(name,w['id']) for name,w,_,_ in records]
    if len(set(identities)) != len(identities):
        raise ValueError('Duplicate diagnostic windows')
    totals = {c: defaultdict(float) for c in categories}
    unknown = defaultdict(int)
    rows = []
    labelers = defaultdict(int)
    for name, window, labels, prediction in records:
        labelers[labels.get('labeler','unknown')] += 1
        row = dict(project=name, window=window['id'], title=window['title'], split=window['split'], categories={})
        for category in categories:
            result = diagnose_window(window, labels, prediction, category, padding)
            if result is None:
                unknown[category] += 1
                continue
            row['categories'][category] = result
            for key in ('scored_seconds','gold_seconds','caught_seconds','missed_seconds','false_seconds'):
                totals[category][key] += result[key]
            for key, bucket in result['miss_buckets'].items():
                totals[category]['miss_'+key] += bucket['seconds']
            for key, bucket in result.get('raw_proposal_overlap',{}).items():
                totals[category]['raw_'+key] += bucket['seconds']
        rows.append(row)
    return dict(scope='Frozen development failure diagnosis; not a new accuracy comparison or qualification',
                interpretation='Miss buckets partition actual missed intervals: reviewable first, then recorded '
                'first-stage proposals, then no recorded support. Assisted coverage uses player padding and '
                'may overstate useful review. Proposal overlap is temporal only, not proof of a correct object '
                'or evidence that rejection was wrong. Raw score >=0.55 is not temporal support or probability. '
                'Unknown spans are excluded. AI annotations remain separate from human qualification.',
                windows=len(records), labelers=dict(labelers), padding='balanced' if padding else 'none',
                benchmark_identity=digest([(name,w,l,p) for name,w,l,p in records]), runs=manifests,
                categories={c:dict(totals[c],unsure_windows=unknown[c]) for c in categories}, windows_detail=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', nargs='+', required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--prediction-dir', type=Path, help='One branch of benchmark_roi output, containing run.json and project subfolders')
    source.add_argument('--run', help='Complete provenance-backed run in each labeling project')
    parser.add_argument('--category', nargs='+', choices=LABEL_CATEGORIES, default=list(LABEL_CATEGORIES))
    parser.add_argument('--no-padding', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records, manifests = load_frozen(prepare_environment(), args.name, args.prediction_dir, args.run)
    result = report(records, manifests, args.category, not args.no_padding)
    write_json(args.output, result)
    print(f"{result['windows']} windows, labels {result['labelers']}; temporal development diagnosis")
    for category, counts in result['categories'].items():
        print(category, {k:round(v,3) for k,v in counts.items()})


if __name__ == '__main__':
    main()
