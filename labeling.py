"""Human labeling projects: window clips, continuous interval labels and their local API.

Labels are the ground truth for accuracy claims. A window only counts once every
category has an explicit status, and all times are clip seconds: predictions run
on the same extracted clips, so labels and detections share one timeline.

    python labeling.py create --name test-1 --windows windows-test-1.json
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import re
import time

import av
from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import FileResponse

from analysis_engine import media_info
from core import run_ffmpeg

LABEL_CATEGORIES = ('nudity', 'sexual_activity', 'violence', 'gore', 'kissing')
DEFINITIONS_REVISION = 1
DEFINITIONS = {
    'nudity': 'Visibly exposed female nipples, genitals, anus or bare buttocks, even briefly or partly. '
              'Male chests, cleavage, underwear, sheer clothing without visible anatomy, shoulders and poses are not nudity.',
    'sexual_activity': 'A depicted sexual act, including clearly simulated intercourse movement without visible genitals. '
                       'Kissing, undressing, lying in bed, nudity alone and aftermath are not sufficient.',
    'violence': 'An active physical attack: striking, kicking, stabbing, shooting (a visible discharge counts even if the target is offscreen), '
                'or forcefully grabbing, dragging, choking or restraining a person. Combat sports count. Aiming or holding a weapon, '
                'tackles in non-combat sport, distress and aftermath alone are not violence.',
    'gore': 'Graphic open wounds, mutilation, exposed organs, dismemberment or severe bloody injury. '
            'Small cuts, light blood, red lighting and bandaged or treated wounds without visible injury are not gore.',
    'kissing': 'Visible mouth-to-mouth kissing. Cheek kisses and faces merely close together are not kissing.',
}
STATUSES = ('reviewed', 'unsure')
LABELERS = ('human', 'claude')  # who decided; reports keep AI labels apart from human ones
CONTEXT = 2.0
NAME = re.compile(r'^[a-z0-9][a-z0-9-]{0,63}$')
WINDOW_ID = re.compile(r'^[a-z0-9][a-z0-9-]{0,119}$')


def slug(text: str) -> str:
    return re.sub(r'[^a-z0-9]+', '-', text.casefold()).strip('-') or 'media'


def project_dir(root: Path, name: str) -> Path:
    if not NAME.match(name or ''):
        raise ValueError('Project names use lowercase letters, digits and hyphens')
    return Path(root) / 'labeling' / name


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        if default is not None:
            return default
        raise


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=1), encoding='utf-8')
    temporary.replace(path)


def clean_intervals(values, low: float, high: float, limit: int = 500) -> list[list[float]]:
    """Finite, merged, half-open intervals clipped to [low, high)."""
    if not isinstance(values, list) or len(values) > limit:
        raise ValueError(f'Intervals must be a list of at most {limit} pairs')
    clipped = []
    for pair in values:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError('Each interval needs a start and end')
        a, b = map(float, pair)
        if not math.isfinite(a) or not math.isfinite(b) or a >= b:
            raise ValueError('Intervals need finite starts before their ends')
        a, b = max(a, low), min(b, high)
        if b - a > 1e-6:
            clipped.append([round(a, 3), round(b, 3)])
    merged = []
    for a, b in sorted(clipped):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged


def validate_labels(window: dict, data: dict) -> dict:
    """Partial saves are allowed; a window is complete when every category has a status."""
    if not isinstance(data, dict) or not isinstance(data.get('categories', {}), dict):
        raise ValueError('Labels need a categories object')
    low, high = window['target']
    categories = {}
    for category, value in data.get('categories', {}).items():
        if category not in LABEL_CATEGORIES:
            raise ValueError(f'Unknown label category {category!r}')
        if not isinstance(value, dict) or value.get('status') not in (*STATUSES, None):
            raise ValueError('Category status must be reviewed, unsure or empty')
        intervals = clean_intervals(value.get('intervals', []), low, high)
        ambiguous = clean_intervals(value.get('ambiguous', []), low, high)
        if value.get('status') == 'unsure' and intervals:
            raise ValueError('An unsure category cannot also contain definite intervals; mark those spans ambiguous')
        categories[category] = dict(status=value.get('status'), intervals=intervals, ambiguous=ambiguous)
    labeler = data.get('labeler', 'human')
    if labeler not in LABELERS:
        raise ValueError(f'Labeler must be one of {LABELERS}')
    spent = data.get('seconds_spent', 0)
    if not isinstance(spent, (int, float)) or not math.isfinite(spent) or not 0 <= spent < 86400:
        raise ValueError('Labeling time must be a finite number of seconds')
    return dict(window=window['id'], definitions_revision=DEFINITIONS_REVISION, categories=categories,
                drafts_shown=bool(data.get('drafts_shown') or window.get('model_exposed')), labeler=labeler, seconds_spent=round(float(spent), 1),
                note=str(data.get('note', ''))[:500], updated=time.strftime('%Y-%m-%dT%H:%M:%S'))


def is_complete(labels: dict | None) -> bool:
    return bool(labels) and all((labels['categories'].get(c) or {}).get('status') in STATUSES
                                for c in LABEL_CATEGORIES)


def extract_clip(source: Path, start: float, length: float, destination: Path) -> dict:
    """Frame-accurate re-encode of [start - context, start + length + context)."""
    offset = max(0.0, start - CONTEXT)
    temporary = destination.with_name(destination.stem + '.part.mp4')
    run_ffmpeg(['-y', '-ss', f'{offset:.3f}', '-i', str(source), '-t', f'{start - offset + length + CONTEXT:.3f}',
                '-map', '0:v:0', '-map', '0:a:0?', '-vf', 'scale=w=min(1920\\,iw):h=-2,setsar=1',
                '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18', '-pix_fmt', 'yuv420p',
                '-c:a', 'aac', '-ac', '2', '-b:a', '160k', '-sn', '-dn', '-map_metadata', '-1',
                '-avoid_negative_ts', 'make_zero', '-movflags', '+faststart', str(temporary)], timeout=900)
    info = media_info(temporary)
    temporary.replace(destination)
    with av.open(str(destination)) as container:
        rate = container.streams.video[0].average_rate
    target_start = start - offset
    target_end = min(info['duration'], target_start + length)
    if target_end - target_start < 1:
        raise ValueError(f'{source.name} is too short at {start:.1f}s')
    return dict(clip_offset=round(offset, 3), clip_duration=info['duration'], has_audio=info['has_audio'],
                fps=float(rate) if rate else 24.0, target=[round(target_start, 3), round(target_end, 3)])


def add_windows(root: Path, name: str, windows: list[dict], workers: int = 3, log=print) -> dict:
    """Create or extend a project. Existing window ids are kept unchanged."""
    folder = project_dir(root, name)
    project = read_json(folder / 'project.json', dict(version=1, name=name, created=time.strftime('%Y-%m-%dT%H:%M:%S'),
                                                      context=CONTEXT, windows=[]))
    known = {w['id'] for w in project['windows']}
    pending = []
    for window in windows:
        if not WINDOW_ID.match(window.get('id', '')):
            raise ValueError(f"Invalid window id {window.get('id')!r}")
        if window['id'] in known:
            continue
        if window.get('split') not in ('train', 'val', 'test', 'dev'):
            raise ValueError(f"Window {window['id']} needs a train, val, test or dev split")
        known.add(window['id'])
        pending.append(window)
    (folder / 'clips').mkdir(parents=True, exist_ok=True)

    def extract(window):
        details = extract_clip(Path(window['source']), float(window['start']), float(window['length']),
                               folder / 'clips' / f"{window['id']}.mp4")
        log(f"clip {window['id']}")
        return dict(window, **details)

    with ThreadPoolExecutor(max(1, workers)) as pool:
        created = list(pool.map(extract, pending))
    project['windows'] += created
    project['definitions_revision'] = DEFINITIONS_REVISION
    write_json(folder / 'project.json', project)
    return project


def latest_predictions(folder: Path, run: str | None = None) -> Path | None:
    runs = sorted((p for p in (folder / 'predictions').glob('*') if p.is_dir()
                   and read_json(p/'run.json',{}).get('complete') is not False), key=lambda p: p.stat().st_mtime)
    if run:
        runs = [p for p in runs if p.name == run]
    return runs[-1] if runs else None


def router(root: Path) -> APIRouter:
    api = APIRouter(prefix='/api/labeling')

    def load(name):
        try:
            folder = project_dir(root, name)
            return folder, read_json(folder / 'project.json')
        except (ValueError, FileNotFoundError):
            raise HTTPException(404, 'Labeling project not found')

    def window_of(project, window_id):
        window = next((w for w in project['windows'] if w['id'] == window_id), None)
        if window is None:
            raise HTTPException(404, 'Window not found')
        return window

    def checked_clip(folder, window):
        path = folder / 'clips' / f"{window['id']}.mp4"
        if window.get('clip_sha256'):
            from provenance import file_hash
            if file_hash(path) != window['clip_sha256']:
                raise HTTPException(409, 'The labeling clip changed or is missing. Restore its original content before labeling.')
        return path

    @api.get('')
    def projects():
        result = []
        for path in sorted((Path(root) / 'labeling').glob('*/project.json')):
            project = read_json(path)
            done = sum(is_complete(read_json(path.parent / 'labels' / f"{w['id']}.json", {})) for w in project['windows'])
            result.append(dict(name=project['name'], windows=len(project['windows']), complete=done))
        return result

    @api.get('/{name}')
    def project(name: str):
        folder, data = load(name)
        drafts = latest_predictions(folder)
        windows = []
        for window in data['windows']:
            labels = read_json(folder / 'labels' / f"{window['id']}.json", {})
            windows.append(dict({k: window[k] for k in ('id', 'title', 'episode', 'split', 'stratum', 'target',
                                                           'clip_duration', 'clip_offset', 'fps')},
                                complete=is_complete(labels), started=bool(labels),
                                has_drafts=bool(drafts and (drafts / f"{window['id']}.json").is_file())))
        return dict(name=data['name'], windows=windows, categories=LABEL_CATEGORIES,
                    definitions=DEFINITIONS, definitions_revision=DEFINITIONS_REVISION,
                    draft_run=drafts.name if drafts else None)

    @api.get('/{name}/clip/{window_id}')
    def clip(name: str, window_id: str):
        folder, data = load(name)
        window = window_of(data, window_id)
        return FileResponse(checked_clip(folder, window), media_type='video/mp4')

    @api.get('/{name}/labels/{window_id}')
    def get_labels(name: str, window_id: str):
        folder, data = load(name)
        window_of(data, window_id)
        return read_json(folder / 'labels' / f'{window_id}.json', {})

    @api.put('/{name}/labels/{window_id}')
    def put_labels(name: str, window_id: str, labels: dict = Body(...)):
        folder, data = load(name)
        window = window_of(data, window_id)
        checked_clip(folder, window)
        try:
            clean = validate_labels(window, labels)
        except (ValueError, TypeError) as error:
            raise HTTPException(422, str(error)) from error
        write_json(folder / 'labels' / f'{window_id}.json', clean)
        return dict(clean, complete=is_complete(clean))

    @api.get('/{name}/drafts/{window_id}')
    def drafts(name: str, window_id: str):
        folder, data = load(name)
        window_of(data, window_id)
        run = latest_predictions(folder)
        path = run / f'{window_id}.json' if run else None
        if not path or not path.is_file():
            raise HTTPException(404, 'No model drafts for this window')
        prediction = read_json(path)
        return dict(run=run.name, events=[e for e in prediction['events'] if e['category'] in LABEL_CATEGORIES],
                    candidates=[c for c in prediction.get('trace', []) if c['category'] in LABEL_CATEGORIES])

    return api


if __name__ == '__main__':
    from model_setup import prepare_environment
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['create'])
    parser.add_argument('--name', required=True)
    parser.add_argument('--windows', type=Path, required=True, help='JSON from select_windows.py choose')
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    windows = json.loads(args.windows.read_text(encoding='utf-8'))['windows']
    project = add_windows(prepare_environment(), args.name, windows, args.workers)
    print(f"{project['name']}: {len(project['windows'])} windows")
