"""Mine fresh drive episodes without changing permanent splits or old caches."""
import argparse
from pathlib import Path
import random
import time
from labeling import read_json, write_json, NAME
from model_setup import prepare_environment
from provenance import digest, file_hash
from select_windows import embedding_path, mine_episode, normalize_title

QUOTA = [
    ('oz', 'F:'), ('vikings', 'F:'), ('the young pope', 'F:'), ('house of lies', 'F:'),
    ('euphoria', 'H:'), ('mr inbetween', 'H:'),
    ('your friends and neighbors', 'H:'), ('presumed innocent', 'H:'),
    ('shameless', 'F:'), ('fargo', 'F:'),
    ('the night manager', 'F:'), ('sons of anarchy', 'F:'),
]


def collect(root, name, quota=QUOTA, seed=1012):
    from analysis_engine import Models, media_info
    if not NAME.fullmatch(name):
        raise ValueError('Invalid collection name')
    folder = root / 'data-collections' / name
    library_path = root / 'labeling/library.json'
    library = read_json(library_path)
    output = folder / 'mined.json'
    previous = read_json(output, {})
    if previous:
        contract = previous['contract']
        if contract['library_sha256'] != file_hash(library_path):
            raise ValueError('Library changed; use a new collection name')
        if contract['code_sha256'] != file_hash(Path(__file__)):
            raise ValueError('Collector changed; use a new collection name')
    else:
        used = {e['source'] for p in (root / 'labeling').glob('mined*.json')
                for e in read_json(p).get('episodes', [])}
        used |= {w['source'] for p in (root / 'labeling').glob('*/project.json')
                 for w in read_json(p)['windows']}
        used |= {e['source'] for p in (root / 'data-collections').glob('*/mined.json')
                 for e in read_json(p).get('episodes', [])}
        rng = random.Random(seed)
        sources = []
        for title, drive in quota:
            choices = sorted((f for f in library['files']
                              if normalize_title(f['title']) == title
                              and Path(f['source']).drive == drive
                              and f['episode'] != 'film' and f['source'] not in used
                              and not embedding_path(root, f['source']).exists()
                              and Path(f['source']).is_file()), key=lambda f: f['source'])
            if not choices:
                raise ValueError('No fresh available episode for ' + title)
            row = rng.choice(choices)
            stat = Path(row['source']).stat()
            sources.append(dict(row, split=library['splits'][title], size=stat.st_size,
                                mtime_ns=stat.st_mtime_ns,
                                duration=media_info(Path(row['source']))['duration']))
        contract = dict(sources=sources, seed=seed, library_sha256=file_hash(library_path),
                        code_sha256=file_hash(Path(__file__)),
                        miner_sha256=file_hash(Path(__file__).with_name('select_windows.py')),
                        clip_sha256=file_hash(root / 'clip/onnx/vision_model_quantized.onnx'),
                        anatomy_sha256=file_hash(root / 'scene-verifier/640m.onnx'),
                        scope='Unlabelled collection hints, not inference or accuracy evidence')
        previous = dict(contract=contract, identity=digest(contract), episodes=[], complete=False)
        write_json(output, previous)
    for row in contract['sources']:
        stat = Path(row['source']).stat()
        if (stat.st_size, stat.st_mtime_ns) != (row['size'], row['mtime_ns']):
            raise ValueError('Source episode changed')
        if library['splits'][row['title']] != row['split']:
            raise ValueError('Permanent title split changed')
    if previous['complete']:
        print('Existing frozen drive collection retained', flush=True)
        return previous
    models = Models(workers=1)
    try:
        models.warm_up(('640m',))
        known = {e['source'] for e in previous['episodes']}
        for index, row in enumerate(contract['sources'], 1):
            if row['source'] in known:
                continue
            started = time.perf_counter()
            print(f"Mining {index}/{len(contract['sources'])}: {row['display']} {row['episode']} ({Path(row['source']).drive}, {row['split']})", flush=True)
            cache = folder / 'embeddings' / embedding_path(root, row['source']).name
            rows = mine_episode(models, row['source'], 'cuda', cache)
            if not rows:
                raise ValueError('No keyframes decoded')
            previous['episodes'].append(dict(row, rows=rows, cache_sha256=file_hash(cache),
                                             seconds=round(time.perf_counter() - started, 3)))
            write_json(output, previous)
            print(f'Completed {index}: {len(rows)} real keyframes', flush=True)
        for row in contract['sources']:
            stat = Path(row['source']).stat()
            if (stat.st_size, stat.st_mtime_ns) != (row['size'], row['mtime_ns']):
                raise ValueError('Source changed while mining')
        if file_hash(library_path) != contract['library_sha256']:
            raise ValueError('Library changed while mining')
        previous['complete'] = True
        write_json(output, previous)
        return previous
    finally:
        models.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    collect(prepare_environment(), args.name)
