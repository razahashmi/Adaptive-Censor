"""Where analysis time goes: per-stage seconds per media second on real labeling clips.

Each stage is timed by wrapping the function that does its work; nested stages
are reported inside their parent, so the total equals measured wall time.

    python profile_speed.py --name test-1 --windows 12 --config '{"speech": false}'
"""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import json
import time

import analysis_engine
from analysis_engine import CHUNK, Models, ScanConfig
from labeling import project_dir, read_json
from model_setup import prepare_environment

totals, counts, stack = defaultdict(float), defaultdict(int), []


@contextmanager
def stage(name):
    stack.append(name)
    started = time.perf_counter()
    try:
        yield
    finally:
        totals['/'.join(stack)] += time.perf_counter() - started
        counts['/'.join(stack)] += 1
        stack.pop()


def timed(owner, attribute, name):
    original = getattr(owner, attribute)

    def wrapper(*args, **kwargs):
        with stage(name):
            return original(*args, **kwargs)
    setattr(owner, attribute, wrapper)


def instrument(models):
    for attribute, name in [('sample_frames', 'decode 2 fps (scenes)'), ('native_frames', 'decode native frames'),
                            ('sample_burst', 'decode burst'), ('audio_samples', 'decode audio'),
                            ('shot_ids', 'cut detection'), ('track_anatomy', 'region tracking'),
                            ('refine_anatomy', 'crop rechecks')]:
        timed(analysis_engine, attribute, name)
    for attribute, name in [('_speech_events', 'speech'), ('_verified_scene_events', 'scene model'),
                            ('_anatomy_events', 'anatomy'), ('_learned_events', 'learned detectors')]:
        timed(models, attribute, name)
    original_verifier = models.verifier

    def verifier(*args, **kwargs):
        model = original_verifier(*args, **kwargs)
        if not getattr(model, '_profiled', False):
            timed(model, 'verify', 'sequence pass')
            timed(model, 'check_frame', 'evidence checks')
            model._profiled = True
        return model
    models.verifier = verifier
    import learned_runtime
    original_compute = learned_runtime.FeatureModels.compute

    def compute(self, kind, *args, **kwargs):
        with stage(f'features {kind}'):
            return original_compute(self, kind, *args, **kwargs)
    learned_runtime.FeatureModels.compute = compute
    original_nude = models.nude

    def nude(*args, **kwargs):
        model = original_nude(*args, **kwargs)
        if not getattr(model, '_profiled', False):
            timed(model, 'detect', 'anatomy inference')
            model._profiled = True
        return model
    models.nude = nude


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--name', default='test-1')
    parser.add_argument('--windows', type=int, default=12)
    parser.add_argument('--config', type=json.loads, default={})
    parser.add_argument('--output', help='Optional JSON report path')
    args = parser.parse_args()
    folder = project_dir(prepare_environment(), args.name)
    windows = read_json(folder / 'project.json')['windows'][:args.windows]
    config = ScanConfig(**{'speech': False, 'anatomy_model': '640m', 'scene_filtering': 'automatic', **args.config})
    models = Models(evaluate_unpromoted=True)
    instrument(models)
    # Warm every model once so startup is reported separately from steady state.
    first = windows[0]
    with stage('warm-up'):
        models.analyze(folder / 'clips' / f"{first['id']}.mp4", first['target'][0], first['target'][0] + CHUNK,
                       first['clip_duration'], config, first['has_audio'])
    warm = dict(totals)
    totals.clear(); counts.clear()
    media = 0.0
    started = time.perf_counter()
    for window in windows:
        low, high = window['target']
        start = low
        while start < high - 1e-6:
            end = min(start + CHUNK, high)
            with stage('chunk'):
                models.analyze(folder / 'clips' / f"{window['id']}.mp4", start, end, window['clip_duration'],
                               config, window['has_audio'])
            media += end - start
            start = end
    wall = time.perf_counter() - started
    models.close()
    print(f"{len(windows)} windows, {media:.0f} media s, {wall:.1f} s wall -> {wall / media:.2f} s per media s "
          f"(warm-up {warm.get('warm-up', 0):.1f} s excluded)")
    rows = sorted(totals.items(), key=lambda item: item[0])
    for path, seconds in rows:
        depth = path.count('/')
        print(f"{'  ' * depth}{path.split('/')[-1]:<28} {seconds / media:6.3f} s/s  {seconds:7.1f} s  x{counts[path]}")
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as handle:
            json.dump(dict(config=config.model_dump(), media_seconds=media, wall_seconds=wall,
                           stages={k: v / media for k, v in totals.items()}), handle, indent=1)


if __name__ == '__main__':
    main()
