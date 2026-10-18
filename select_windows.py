"""Choose labeling windows from a local library with fixed, title-separated splits.

Splits are assigned once per title and stored; they never change afterwards, so a
test title is never used for tuning. Titles examined during earlier development
are forced into 'dev'. Windows mix random positions (unbiased, mostly clean) with
positions mined by fast detectors (enriched positives); evaluation reports each
stratum separately because mined windows favor what the miners already find.

    python select_windows.py splits --roots sample-media/shows sample-media/movies
    python select_windows.py mine --split test --episodes-per-title 3 --output mined-test.json
    python select_windows.py choose --split test --mined mined-test.json --output windows-test-1.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import time

VIDEO = {'.mkv', '.mp4', '.avi', '.m4v', '.mov', '.wmv', '.ts'}
# Titles whose clips informed revisions 1-7. They may train models, never test them.
DEVELOPMENT_TITLES = {'american gods', 'hannibal', 'brooklyn nine nine', 'ballers', 'daredevil',
                      'true detective', 'westworld', 'the proposal', 'hit man'}
SPLIT_SALT = 'adaptive-censor-splits-v1'
MINED_CATEGORIES = ('nudity', 'sexual_activity', 'violence', 'gore', 'kissing')
# Look-alikes of each category: labeled windows of these teach detectors what is NOT a positive.
CONFUSERS = {
    'kissing': ['two people hugging each other', 'a couple embracing closely', 'two faces close together talking'],
    'violence': ['people dancing energetically', 'a man aiming a gun', 'people playing a contact sport',
                 'people arguing angrily face to face', 'a crowd at a protest', 'a person hitting a punching bag',
                 'people exercising in a gym', 'people playfully wrestling', 'a martial arts class'],
    'nudity': ['a shirtless man', 'a woman in a bikini', 'a woman in lingerie', 'a person wrapped in a towel'],
    'sexual_activity': ['a couple lying in bed talking', 'people dancing closely in a club', 'a person getting a massage'],
    'gore': ['a surgeon in an operating room', 'raw meat being cut in a kitchen', 'a person with a bandage',
             'a room lit with red light'],
}
NEUTRAL = ['an ordinary peaceful scene', 'people having a conversation', 'a landscape or a room',
           'a person doing an everyday activity']
# Weakest categories first, so scarce positives get windows before common ones.
POSITIVE_ORDER = ('sexual_activity', 'nudity', 'gore', 'violence', 'kissing')


# Folder spellings that name an already-split title.
TITLE_ALIASES = {'the handmaids tale': 'the handmaid s tale'}
EPISODE_PATTERNS = (
    re.compile(r'[Ss](\d{1,2})[ ._-]?[Ee](\d{1,3})'),     # S01E02, s1 e2
    re.compile(r'\b(\d{1,2})x(\d{2})\b'),                  # 1x02
    re.compile(r'[._ ](\d)(\d{2})[._ ]'),                    # castle.2009.801.hdtv
    re.compile(r'[._ ]([0-2]\d)(\d{2})[._ -]'),              # one.tree.hill.0504-yestv
)


def normalize_title(text: str) -> str:
    text = re.sub(r'^www\.\S+\s*-\s*', '', text, flags=re.I)
    text = re.sub(r'[\[(].*$', '', text)
    text = re.sub(r'[ ._-]S\d{1,2}([ ._-]|E\d|$).*$', '', text, flags=re.I)  # The.Bear.S02.COMPLETE...
    text = re.sub(r'\b(19|20)\d\d\b.*$', '', text)
    title = re.sub(r'[^a-z0-9]+', ' ', text.casefold()).strip()
    return TITLE_ALIASES.get(title, title)


def episode_of(name: str, folder: str = '') -> str:
    stem = re.sub(r"\b(19|20)\d\d\b", "", Path(name).stem)  # years are not episode numbers
    for pattern in EPISODE_PATTERNS:
        match = pattern.search(stem)
        if match:
            return f's{int(match[1]):02d}e{int(match[2]):02d}'
    season = re.search(r'\bSeason[ ._-]?(\d{1,2})\b', folder, flags=re.I)
    number = re.fullmatch(r'(\d{1,3})( \(\d+\))?', stem.strip())
    if season and number:  # 'New Girl/Season 7/10.mp4'
        return f's{int(season[1]):02d}e{int(number[1]):02d}'
    part = re.search(r'\bPart[ ._-]?(\d{1,2})\b', stem, flags=re.I)
    return f's01e{int(part[1]):02d}' if part else 'film'


def inventory(roots: list[Path], show_dirs: list[Path] = ()) -> list[dict]:
    """One entry per distinct video. A root holds one folder per title; a show dir is one title.

    The same episode copied to several drives is listed once (first root wins);
    films are matched by title and file size.
    """
    files, seen = [], set()

    def add(path, folder):
        if path.suffix.lower() not in VIDEO or 'sample' in path.stem.casefold():
            return
        title, episode = normalize_title(folder), episode_of(path.name, path.parent.name)
        key = (title, episode) if episode != 'film' else (title, path.stat().st_size)
        if key in seen:
            return
        seen.add(key)
        files.append(dict(title=title, display=folder.strip(), episode=episode, source=str(path)))

    for root in roots:
        for path in sorted(root.rglob('*')):
            relative = path.relative_to(root)
            add(path, relative.parts[0] if len(relative.parts) > 1 else path.stem)
    for folder in show_dirs:
        for path in sorted(folder.rglob('*')):
            add(path, folder.name)
    return files


def assign_splits(titles: list[str], fractions=(('test', .2), ('val', .2))) -> dict[str, str]:
    ordered = sorted((t for t in set(titles) if t not in DEVELOPMENT_TITLES),
                     key=lambda t: hashlib.sha256((SPLIT_SALT + t).encode()).hexdigest())
    splits, index = {}, 0
    for name, fraction in fractions:
        count = round(len(ordered) * fraction)
        splits.update({t: name for t in ordered[index:index + count]})
        index += count
    splits.update({t: 'train' for t in ordered[index:]})
    splits.update({t: 'dev' for t in set(titles) if t in DEVELOPMENT_TITLES})
    return splits


def load_splits(root: Path) -> dict:
    path = root / 'labeling' / 'library.json'
    if not path.is_file():
        raise SystemExit('Run: python select_windows.py splits --roots <library folders>')
    return json.loads(path.read_text(encoding='utf-8'))


def duration(path: str) -> float | None:
    import av
    try:
        with av.open(path) as container:
            return float(container.duration / av.time_base) if container.duration else None
    except (av.error.FFmpegError, OSError):
        return None


def keyframes(path: str, width: int = 640):
    """Decode keyframes only. Times are relative to the container start, matching ffmpeg -ss."""
    import av
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.codec_context.skip_frame = 'NONKEY'
        stream.thread_type = 'AUTO'
        origin = (container.start_time or 0) / av.time_base
        for frame in container.decode(stream):
            if frame.time is None:
                continue
            height = max(2, int(frame.height * min(width, frame.width) / frame.width) // 2 * 2)
            yield float(frame.time) - origin, frame.reformat(width=min(width, frame.width), height=height,
                                                             format='rgb24').to_ndarray()


def mine_episode(models, path: str, device: str, save: Path | None = None) -> list[dict]:
    """Fast hints only: anatomy scores and CLIP similarities on keyframes.

    With `save`, keyframe CLIP embeddings are stored so windows can later be
    ranked by any text prompt, including look-alikes of each category.
    """
    import numpy as np
    from detectors import NUDITY_CLASSES
    nude = models.nude(device, '640m')
    rows, batch, stored = [], [], []

    def flush():
        embeddings = models.clip_embeddings(batch, device)
        stored.append(embeddings.astype(np.float16))
        for (t, pixels), clip_scores in zip(batch, models.scene_scores(batch, device, '', embeddings)):
            hits = [h for h in nude.detect(pixels[:, :, ::-1].copy()) if h['class'] in NUDITY_CLASSES]
            rows.append(dict(time=round(t, 3), nudity=max((h['score'] for h in hits), default=0.0),
                             **{k: round(v, 4) for k, v in clip_scores.items()}))
        batch.clear()

    for item in keyframes(path):
        batch.append(item)
        if len(batch) == 32:
            flush()
    if batch:
        flush()
    if save is not None and stored:
        save.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save, times=np.array([r['time'] for r in rows], dtype=np.float32),
                            embeddings=np.concatenate(stored))
    return rows


def embedding_path(root: Path, source: str) -> Path:
    return root / 'labeling' / 'mined-embeddings' / (hashlib.sha1(source.encode()).hexdigest()[:20] + '.npz')


def episodes_for(library: dict, split: str, per_title: int, seed: int) -> list[dict]:
    rng = random.Random(f'{seed}-{split}')
    chosen = []
    for title in sorted(t for t, s in library['splits'].items() if s == split):
        files = [f for f in library['files'] if f['title'] == title]
        chosen += rng.sample(files, min(per_title, len(files)))
    return chosen


def choose(library: dict, mined: dict | None, split: str, random_per_title: int, mined_per_title: int,
           length: float, seed: int) -> list[dict]:
    """Random windows per title plus top mined windows, spread across categories, never overlapping."""
    rng = random.Random(f'{seed}-{split}-choose')
    windows, taken = [], {}

    def free(source, start):
        return all(abs(start - other) >= length + 4 for other in taken.get(source, []))

    def add(file, start, stratum, score=None):
        start = round(max(0.0, start), 1)
        if not free(file['source'], start):
            return False
        taken.setdefault(file['source'], []).append(start)
        key = f"{file['title']}-{file['episode']}-{int(start)}"
        windows.append(dict(id=re.sub(r'[^a-z0-9]+', '-', key)[:120].strip('-'), title=file['display'],
                            episode=file['episode'], source=file['source'], split=split, start=start,
                            length=length, stratum=stratum, **({'score': score} if score is not None else {})))
        return True

    titles = sorted(t for t, s in library['splits'].items() if s == split)
    for title in titles:
        files = [f for f in library['files'] if f['title'] == title]
        if mined:
            scans = [m for m in mined['episodes'] if m['title'] == title and m['rows']]
            # Round-robin over categories so one dominant miner cannot fill the quota.
            ranked = {c: sorted(((row[c], scan, row['time']) for scan in scans for row in scan['rows'] if c in row),
                                key=lambda item: -item[0]) for c in MINED_CATEGORIES}
            added, order = 0, list(MINED_CATEGORIES)
            rng.shuffle(order)
            while added < mined_per_title and any(ranked.values()):
                for category in order:
                    while ranked[category]:
                        score, scan, t = ranked[category].pop(0)
                        file = next(f for f in files if f['source'] == scan['source'])
                        if add(file, t - length / 2, f'mined:{category}', round(score, 4)):
                            added += 1
                            break
                    if added >= mined_per_title:
                        break
        added, attempts = 0, 0
        while added < random_per_title and attempts < 50:
            attempts += 1
            file = rng.choice(files)
            total = duration(file['source'])
            if total and total >= 120 and add(file, rng.uniform(30, total - 60 - length), 'random'):
                added += 1
    return windows


def used_sources(paths) -> set:
    """Sources already covered by mined files (episodes) or labeling projects (windows)."""
    sources = set()
    for path in paths:
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        sources |= {e['source'] for e in data.get('episodes', [])} | {w['source'] for w in data.get('windows', [])}
    return sources


def occupied(paths) -> dict:
    taken = {}
    for path in paths:
        for w in json.loads(Path(path).read_text(encoding='utf-8')).get('windows', []):
            taken.setdefault(w['source'], []).append(float(w['start']))
    return taken


def confuser_scores(bank, data):
    """Per keyframe and category: best look-alike prompt similarity minus best neutral similarity."""
    import numpy as np
    pictures = data['embeddings'].astype(np.float32)
    return {category: (pictures @ prompts.T).max(1) - (pictures @ neutral.T).max(1)
            for category, (prompts, neutral) in bank.items()}


def target(library, mined, embeddings, bank, split, positives, confusers, randoms, length, seed, taken,
           positive_order=POSITIVE_ORDER, confuser_order=None):
    """Per title: likely positives (weakest categories first), likely look-alikes, random windows."""
    rng = random.Random(f'{seed}-{split}-target')
    windows = []
    taken = {k: list(v) for k, v in taken.items()}

    def add(file, start, stratum, score=None):
        start = round(max(0.0, start), 1)
        if any(abs(start - other) < length + 4 for other in taken.get(file['source'], [])):
            return False
        taken.setdefault(file['source'], []).append(start)
        key = f"{file['title']}-{file['episode']}-{int(start)}"
        windows.append(dict(id=re.sub(r'[^a-z0-9]+', '-', key)[:120].strip('-'), title=file['display'],
                            episode=file['episode'], source=file['source'], split=split, start=start, length=length,
                            stratum=stratum, **({'score': round(float(score), 4)} if score is not None else {})))
        return True

    titles = sorted(t for t, s in library['splits'].items() if s == split)
    for number, title in enumerate(titles):
        files = {f['source']: f for f in library['files'] if f['title'] == title}
        candidates = {}  # stratum -> [(score, source, time)]
        for scan in (m for m in mined['episodes'] if m['title'] == title and m['rows']):
            for row in scan['rows']:
                for category in MINED_CATEGORIES:
                    candidates.setdefault(f'mined:{category}', []).append((row[category], scan['source'], row['time']))
            data = embeddings(scan['source'])
            if data is not None:
                for category, margins in confuser_scores(bank, data).items():
                    candidates.setdefault(f'confuser:{category}', []).extend(
                        (float(m), scan['source'], float(t)) for t, m in zip(data['times'], margins))
        shift = number % len(positive_order)
        order = tuple(positive_order[shift:]) + tuple(positive_order[:shift])
        confusing = tuple(confuser_order) if confuser_order else order[::-1]
        if confuser_order:
            confusing = confusing[number % len(confusing):] + confusing[:number % len(confusing)]
        for quota, prefix, categories in ((positives, 'mined', order), (confusers, 'confuser', confusing)):
            added = 0
            for category in categories * max(1, quota):
                if added >= quota:
                    break
                for score, source, t in sorted(candidates.get(f'{prefix}:{category}', []), key=lambda c: -c[0])[:300]:
                    if source in files and add(files[source], t - length / 2, f'{prefix}:{category}', score):
                        added += 1
                        break
        added, attempts = 0, 0
        while added < randoms and attempts < 50 and files:
            attempts += 1
            file = rng.choice(list(files.values()))
            total = duration(file['source'])
            if total and total >= 120 and add(file, rng.uniform(30, total - 60 - length), 'random'):
                added += 1
    return windows


def main():
    from model_setup import prepare_environment
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    splits = sub.add_parser('splits', help='Inventory the library and fix title splits once')
    splits.add_argument('--roots', type=Path, nargs='+', required=True, help='Folders holding one subfolder per title')
    splits.add_argument('--show-dirs', type=Path, nargs='*', default=[], help='Folders that are each a single title')
    mine = sub.add_parser('mine', help='Scan sampled episodes with fast detectors (GPU)')
    mine.add_argument('--split', required=True)
    mine.add_argument('--episodes-per-title', type=int, default=3)
    mine.add_argument('--device', default='cuda')
    mine.add_argument('--seed', type=int, default=1)
    mine.add_argument('--limit', type=int, help='Stop after this many new episodes; rerun to resume')
    mine.add_argument('--output', type=Path, required=True)
    mine.add_argument('--skip', type=Path, nargs='*', default=[], help='Mined files or projects whose sources are already used')
    pick = sub.add_parser('choose', help='Write labeling windows')
    pick.add_argument('--split', required=True)
    pick.add_argument('--mined', type=Path)
    pick.add_argument('--random-per-title', type=int, default=2)
    pick.add_argument('--mined-per-title', type=int, default=3)
    pick.add_argument('--length', type=float, default=12)
    pick.add_argument('--seed', type=int, default=1)
    pick.add_argument('--output', type=Path, required=True)
    aim = sub.add_parser('target', help='Windows for likely positives, look-alikes and random positions')
    aim.add_argument('--split', required=True)
    aim.add_argument('--mined', type=Path, required=True)
    aim.add_argument('--positives', type=int, default=2)
    aim.add_argument('--confusers', type=int, default=1)
    aim.add_argument('--random', type=int, default=0)
    aim.add_argument('--positive-categories', nargs='+', choices=MINED_CATEGORIES, default=list(POSITIVE_ORDER))
    aim.add_argument('--confuser-categories', nargs='+', choices=MINED_CATEGORIES)
    aim.add_argument('--length', type=float, default=12)
    aim.add_argument('--seed', type=int, default=2)
    aim.add_argument('--avoid', type=Path, nargs='*', default=[], help='Window files or projects that must not overlap')
    aim.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = prepare_environment()

    if args.command == 'splits':
        path = root / 'labeling' / 'library.json'
        files = inventory(args.roots, args.show_dirs)
        previous = json.loads(path.read_text(encoding='utf-8'))['splits'] if path.is_file() else {}
        fresh = assign_splits([f['title'] for f in files])
        # Existing assignments are permanent; only new titles receive a split.
        library = dict(roots=[str(r) for r in args.roots], show_dirs=[str(d) for d in args.show_dirs], salt=SPLIT_SALT, updated=time.strftime('%Y-%m-%d'),
                       splits={**fresh, **previous}, files=files)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(library, indent=1), encoding='utf-8')
        counts = {s: sum(v == s for v in library['splits'].values()) for s in ('train', 'val', 'test', 'dev')}
        print(f"{len(files)} files, {len(library['splits'])} titles: {counts}")
        for split in ('test', 'val'):
            print(split + ':', ', '.join(sorted(t for t, s in library['splits'].items() if s == split)))
        return

    library = load_splits(root)
    if args.command == 'mine':
        from analysis_engine import Models
        models = Models()
        output = json.loads(args.output.read_text(encoding='utf-8')) if args.output.is_file() else dict(episodes=[])
        done = {e['source'] for e in output['episodes']} | used_sources(args.skip)
        # Files on a disconnected drive stay pending for a later run instead of failing.
        pool = dict(library, files=[f for f in library['files'] if f['source'] not in done and Path(f['source']).exists()])
        # The quota counts episodes this output already holds, so resumed runs move on to other titles.
        have = {}
        for episode in output['episodes']:
            have[episode['title']] = have.get(episode['title'], 0) + 1
        pending = []
        for file in episodes_for(pool, args.split, args.episodes_per_title, args.seed):
            if have.get(file['title'], 0) < args.episodes_per_title:
                have[file['title']] = have.get(file['title'], 0) + 1
                pending.append(file)
        for file in pending[:args.limit]:
            started = time.perf_counter()
            try:
                rows = mine_episode(models, file['source'], args.device, embedding_path(root, file['source']))
            except Exception as error:  # A damaged file should not stop a long scan.
                rows, file = [], dict(file, error=f'{type(error).__name__}: {error}'[:300])
            output['episodes'].append(dict(file, rows=rows, seconds=round(time.perf_counter() - started, 1)))
            args.output.write_text(json.dumps(output), encoding='utf-8')
            print(f"{file['display']} {file['episode']}: {len(rows)} keyframes in {output['episodes'][-1]['seconds']}s", flush=True)
        return

    if args.command == 'target':
        import numpy as np
        from analysis_engine import Models
        models = Models()
        neutral = models.text_embeddings(NEUTRAL)
        bank = {c: (models.text_embeddings(prompts), neutral) for c, prompts in CONFUSERS.items()}

        def embeddings(source):
            path = embedding_path(root, source)
            if not path.is_file():
                return None
            with np.load(path) as data:
                return dict(times=data['times'], embeddings=data['embeddings'])

        mined = json.loads(args.mined.read_text(encoding='utf-8'))
        windows = target(library, mined, embeddings, bank, args.split, args.positives, args.confusers, args.random,
                         args.length, args.seed, occupied(args.avoid), args.positive_categories, args.confuser_categories)
        args.output.write_text(json.dumps(dict(split=args.split, seed=args.seed, windows=windows), indent=1), encoding='utf-8')
        strata = {}
        for w in windows:
            strata[w['stratum']] = strata.get(w['stratum'], 0) + 1
        print(f'{len(windows)} windows: {strata}')
        return

    mined = json.loads(args.mined.read_text(encoding='utf-8')) if args.mined else None
    windows = choose(library, mined, args.split, args.random_per_title, args.mined_per_title, args.length, args.seed)
    args.output.write_text(json.dumps(dict(split=args.split, seed=args.seed, windows=windows), indent=1), encoding='utf-8')
    strata = {}
    for w in windows:
        strata[w['stratum']] = strata.get(w['stratum'], 0) + 1
    print(f'{len(windows)} windows: {strata}')


if __name__ == '__main__':
    main()
