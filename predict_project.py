"""Run the detection pipeline over labeling windows; store decisions and candidate traces.

Each run records the code commit, configuration and revisions so comparisons are
reproducible. Predictions also appear as drafts on the labeling page.

    python predict_project.py --name test-1 --run rev7-baseline
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
from pathlib import Path
import subprocess
import time

from analysis_engine import ANALYSIS_REVISION, CHUNK, Models, ScanConfig
from labeling import project_dir, read_json, write_json
from model_setup import prepare_environment
from scene_verifier import VERIFIER_REVISION
from provenance import digest, file_hash, head_identity, pipeline_identity


def code_revision() -> dict:
    here = Path(__file__).parent
    def git(*args):
        return subprocess.run(['git', '-c', f'safe.directory={here.as_posix()}', *args], cwd=here, capture_output=True, text=True).stdout.strip()
    return dict(commit=git('rev-parse', 'HEAD'), dirty=bool(git('status', '--porcelain', '--untracked-files=no')))


def predict_window(models, clip: Path, window: dict, config: ScanConfig) -> dict:
    low, high = window['target']
    events, trace = [], []
    started = time.perf_counter()
    # Match the player's global six-second chunk grid, including its context.
    start = math.floor(low / CHUNK) * CHUNK
    while start < high - 1e-6:
        end = min(start + CHUNK, window['clip_duration'])
        events += models.analyze(clip, start, end, window['clip_duration'], config, window['has_audio'], trace=trace)
        start = end
    return dict(window=window['id'], events=events, trace=trace, seconds=round(time.perf_counter() - started, 2))


def run_contract(root, folder, project, windows, config, workers, evaluate_unpromoted):
    pipeline = pipeline_identity(root, config, workers)
    heads = read_json(root / 'heads' / f'{config.learned_heads}.json') if config.learned_heads else None
    settings = dict(config=config.model_dump(), analysis_revision=ANALYSIS_REVISION, verifier_revision=VERIFIER_REVISION,
                    workers=workers, evaluate_unpromoted=evaluate_unpromoted, pipeline='player', pipeline_identity=pipeline,
                    prediction_code=file_hash(Path(__file__)), project_identity=digest(project),
                    expected_windows=[w['id'] for w in windows],
                    sources={w['id']: file_hash(folder / 'clips' / f"{w['id']}.mp4") for w in windows},
                    head_certificate=file_hash(root/'heads'/f'{config.learned_heads}.json') if config.learned_heads and not evaluate_unpromoted else None,
                    head_identity=head_identity(heads) if heads else None)
    return dict(settings, identity=digest(settings))


def ensure_run(manifest, settings):
    if manifest.is_file():
        previous = read_json(manifest)
        if previous.get('identity') != settings['identity']:
            raise ValueError('Run identity changed (code, assets, workers, source or selection); choose another run name.')
        return previous
    write_json(manifest, dict(settings, code=code_revision(), started=time.strftime('%Y-%m-%dT%H:%M:%S')))
    return read_json(manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--name', required=True, help='Labeling project')
    parser.add_argument('--run', required=True, help='Name for this prediction run')
    parser.add_argument('--split', help='Only windows in this split')
    parser.add_argument('--config', type=json.loads, default={},
                        help='ScanConfig overrides as JSON, e.g. \'{"anatomy_model": "320n"}\'')
    parser.add_argument('--force', action='store_true', help='Recompute windows that already have predictions')
    parser.add_argument('--workers', type=int, default=1, help='Windows analyzed concurrently (scene model slots)')
    parser.add_argument('--evaluate-unpromoted', action='store_true', help='Allow a learned head set that is not promoted yet')
    parser.add_argument('--limit', type=int, help='Only the first N windows (speed comparisons)')
    args = parser.parse_args()

    root = prepare_environment()
    folder = project_dir(root, args.name)
    project = read_json(folder / 'project.json')
    # Measure every visual detector automatically; speech is evaluated separately.
    config = ScanConfig(**{'speech': False, 'anatomy_model': '640m', 'scene_filtering': 'automatic', **args.config})
    output = folder / 'predictions' / args.run
    manifest = output / 'run.json'
    if args.workers < 1 or args.workers > 8 or (args.limit is not None and args.limit < 1):
        raise SystemExit('Workers must be 1–8 and limit must be positive')
    windows = [w for w in project['windows'] if not args.split or w['split'] == args.split][:args.limit]
    settings = run_contract(root, folder, project, windows, config, args.workers, args.evaluate_unpromoted)
    try:
        ensure_run(manifest, settings)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    models = Models(workers=args.workers, evaluate_unpromoted=args.evaluate_unpromoted)
    try:
        if config.learned_heads:
            models.learned_features().warm_up(models.heads(config.learned_heads)['kinds'])
        for window in windows:
            path = output / f"{window['id']}.json"
            if path.is_file() and read_json(path).get('run_identity') != settings['identity']:
                raise RuntimeError(f'{path.name} has stale or missing provenance; choose a new run')
        pending = [w for w in windows if args.force or not (output / f"{w['id']}.json").is_file()]
        started = time.perf_counter()
        with ThreadPoolExecutor(max(1, args.workers)) as pool:
            futures = {pool.submit(predict_window, models, folder / 'clips' / f"{w['id']}.mp4", w, config): w for w in pending}
            for number, future in enumerate(as_completed(futures), 1):
                window, result = futures[future], future.result()
                if run_contract(root, folder, project, windows, config, args.workers, args.evaluate_unpromoted)['identity'] != settings['identity']:
                    raise RuntimeError('Models or code changed during prediction; results were not committed')
                write_json(output / f"{window['id']}.json", dict(result, run_identity=settings['identity']))
                print(f"[{number}/{len(pending)}] {window['id']}: {len(result['events'])} events, "
                      f"{len(result['trace'])} candidates, {result['seconds']}s", flush=True)
        media = sum(w['target'][1] - w['target'][0] for w in pending)
        if media:
            print(f'{media:.0f} media s in {time.perf_counter() - started:.0f} s wall '
                  f'({(time.perf_counter() - started) / media:.2f} s per media s, {args.workers} workers)')
        completed = sum((output / f"{w['id']}.json").is_file() for w in windows)
        write_json(manifest, dict(read_json(manifest), completed=completed, complete=completed == len(windows),
                                 last_wall_seconds=round(time.perf_counter()-started, 3)))
    finally:
        models.close()


if __name__ == '__main__':
    main()
