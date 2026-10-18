"""Prepare independent, draft-free human review and expanded validation clips.

Existing test failures become development audits, never a fresh final test.
The validation expansion respects the frozen title split and excludes used times.
"""
import argparse
from collections import defaultdict
from pathlib import Path
import shutil
import time

from evaluate_labels import score_window
from labeling import LABEL_CATEGORIES, add_windows, is_complete, project_dir, read_json, write_json
from model_setup import prepare_environment
from select_windows import normalize_title, occupied, target


def audit(root, names, run, output, limit=24):
    destination = project_dir(root, output)
    if (destination/'project.json').exists():
        raise FileExistsError(f'{output} already exists; choose another project name')
    ranked = defaultdict(list)
    for name in names:
        folder = project_dir(root, name)
        for window in read_json(folder/'project.json')['windows']:
            labels = read_json(folder/'labels'/f"{window['id']}.json", {})
            prediction = read_json(folder/'predictions'/run/f"{window['id']}.json", {})
            if not is_complete(labels) or not prediction:
                continue
            for category in LABEL_CATEGORIES:
                counts = score_window(window, labels, prediction, category, True)
                if counts:
                    for kind in ('fp', 'fn', 'gold'):
                        if counts[kind] > .04:
                            ranked[(category, kind)].append((counts[kind], name, window))
    for bucket in ranked.values():
        bucket.sort(key=lambda r:-r[0])
    chosen, seen = [], set()
    while len(chosen) < limit and any(ranked.values()):
        for key, bucket in sorted(ranked.items()):
            while bucket:
                amount, name, window = bucket.pop(0)
                identity = name, window['id']
                if identity not in seen:
                    seen.add(identity)
                    chosen.append((name, window, key))
                    break
            if len(chosen) >= limit:
                break
    windows = []
    for name, window, (category, kind) in chosen:
        new_id = f"{name}-{window['id']}"
        source = project_dir(root, name)/'clips'/f"{window['id']}.mp4"
        clip = destination/'clips'/f'{new_id}.mp4'
        clip.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, clip)
        windows.append(dict(window, id=new_id, split='dev', stratum=f'audit:{category}:{kind}',
                            origin=dict(project=name, window=window['id'], run=run)))
    write_json(destination/'project.json', dict(version=1, name=output, context=2, windows=windows,
               created=time.strftime('%Y-%m-%dT%H:%M:%S'), purpose='Independent human development audit; labels and model drafts withheld'))
    return windows


def expand_validation(root, output, positives=3):
    folder = root/'labeling'
    library, mined = read_json(folder/'library.json'), read_json(folder/'mined-val.json')
    # Offline/disconnected sources do not leave half-created projects.
    library['files'] = [f for f in library['files'] if Path(f['source']).is_file()]
    available = {f['source'] for f in library['files']}
    mined['episodes'] = [e for e in mined['episodes'] if e['source'] in available]
    taken = occupied(list(folder.glob('*/project.json')))
    windows = target(library, mined, lambda _:None, {}, 'val', positives, 0, 0, 12, 17, taken,
                     positive_order=('sexual_activity', 'nudity', 'gore'))
    write_json(folder/f'windows-{output}.json', dict(windows=windows, purpose='Unlabeled enriched validation; proposals are not ground truth'))
    return add_windows(root, output, windows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command', required=True)
    a=sub.add_parser('audit'); a.add_argument('--projects', nargs='+', required=True)
    a.add_argument('--run', required=True); a.add_argument('--name', required=True); a.add_argument('--limit', type=int, default=24)
    v=sub.add_parser('validation'); v.add_argument('--name', required=True); v.add_argument('--positives-per-title', type=int, default=3)
    args=p.parse_args(); root=prepare_environment()
    if args.command == 'audit':
        windows=audit(root,args.projects,args.run,args.name,args.limit)
    else:
        windows=expand_validation(root,args.name,args.positives_per_title)['windows']
    print(f'{args.name}: {len(windows)} unlabeled windows / {len({normalize_title(w["title"]) for w in windows})} titles. Open /label.html for independent review.')


if __name__ == '__main__':
    main()
