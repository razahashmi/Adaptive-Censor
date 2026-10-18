"""Label validation, completeness and the labeling API; no media or models."""
import json
from pathlib import Path
import tempfile
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from labeling import LABEL_CATEGORIES, clean_intervals, is_complete, router, validate_labels, write_json
from select_windows import assign_splits, normalize_title, DEVELOPMENT_TITLES

WINDOW = dict(id='show-s01e01-100', title='Show', episode='s01e01', split='test', stratum='random',
              target=[2.0, 14.0], clip_duration=16.0, clip_offset=98.0, fps=24.0, has_audio=True)


def reviewed(**intervals):
    return {'categories': {c: dict(status='reviewed', intervals=intervals.get(c, [])) for c in LABEL_CATEGORIES}}


class LabelTests(unittest.TestCase):
    def test_intervals_are_clipped_merged_and_validated(self):
        self.assertEqual(clean_intervals([[1, 3], [2.5, 4], [13, 20], [15, 16]], 2, 14), [[2, 4], [13, 14]])
        for bad in [[[3, 2]], [[1, float('nan')]], [[1]], 'x']:
            with self.assertRaises(ValueError):
                clean_intervals(bad, 0, 10)

    def test_partial_labels_are_incomplete_and_unsure_cannot_hold_definite_spans(self):
        partial = validate_labels(WINDOW, {'categories': {'nudity': {'status': 'reviewed', 'intervals': [[3, 4]]}}})
        self.assertFalse(is_complete(partial))
        self.assertTrue(is_complete(validate_labels(WINDOW, reviewed(violence=[[5, 6]]))))
        with self.assertRaises(ValueError):
            validate_labels(WINDOW, {'categories': {'gore': {'status': 'unsure', 'intervals': [[3, 4]]}}})
        with self.assertRaises(ValueError):
            validate_labels(WINDOW, {'categories': {'language': {'status': 'reviewed'}}})

    def test_api_saves_labels_and_serves_drafts_from_latest_run(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_json(root / 'labeling' / 'p1' / 'project.json', dict(name='p1', windows=[WINDOW]))
            app = FastAPI(); app.include_router(router(root)); client = TestClient(app)
            self.assertEqual(client.get('/api/labeling').json(), [dict(name='p1', windows=1, complete=0)])
            self.assertEqual(client.get('/api/labeling/p1/labels/show-s01e01-100').json(), {})
            self.assertEqual(client.get('/api/labeling/p1/drafts/show-s01e01-100').status_code, 404)
            saved = client.put('/api/labeling/p1/labels/show-s01e01-100', json=reviewed(kissing=[[1, 3]]))
            self.assertTrue(saved.json()['complete'])
            self.assertEqual(saved.json()['categories']['kissing']['intervals'], [[2.0, 3.0]])
            self.assertEqual(client.put('/api/labeling/p1/labels/show-s01e01-100', json={'categories': {'x': {}}}).status_code, 422)
            self.assertEqual(client.get('/api/labeling/p1/labels/other').status_code, 404)
            self.assertEqual(client.get('/api/labeling/..%2Fp1').status_code, 404)
            write_json(root / 'labeling' / 'p1' / 'predictions' / 'run-a' / 'show-s01e01-100.json',
                       dict(events=[dict(category='nudity', start=3, end=4), dict(category='language', start=1, end=2)],
                            trace=[dict(stage='clip', category='violence', start=5, end=6)]))
            drafts = client.get('/api/labeling/p1/drafts/show-s01e01-100').json()
            self.assertEqual([e['category'] for e in drafts['events']], ['nudity'])
            self.assertEqual(drafts['run'], 'run-a')
            project = client.get('/api/labeling/p1').json()
            self.assertTrue(project['windows'][0]['complete'] and project['windows'][0]['has_drafts'])


class SplitTests(unittest.TestCase):
    def test_development_titles_never_enter_test_or_validation(self):
        titles = [f'show {i}' for i in range(40)] + sorted(DEVELOPMENT_TITLES)
        splits = assign_splits(titles)
        self.assertTrue(all(splits[t] == 'dev' for t in DEVELOPMENT_TITLES))
        self.assertEqual(sum(s == 'test' for s in splits.values()), 8)
        self.assertEqual(splits, assign_splits(list(reversed(titles))))

    def test_title_normalization_groups_releases(self):
        self.assertEqual(normalize_title('Hit Man (2023) [1080p] [WEBRip]'), 'hit man')
        self.assertEqual(normalize_title('www.UIndex.org    -    Obsession (2025) 1080p'), 'obsession')
        self.assertEqual(normalize_title('Brooklyn Nine Nine'), 'brooklyn nine nine')
        self.assertEqual(normalize_title('The.Bear.S02.COMPLETE.1080p.HULU.WEB.h264-EDITH[TGx]'), 'the bear')
        self.assertEqual(normalize_title('the bear'), 'the bear')
        self.assertEqual(normalize_title('Presumed.Innocent.S01.COMPLETE.1080p.ATVP.WEB-DL'), 'presumed innocent')
        self.assertEqual(normalize_title('The Handmaids Tale'), normalize_title("The Handmaid's Tale"))
        self.assertEqual(normalize_title("You're The Worst"), normalize_title('You’re the Worst'))
        self.assertEqual(normalize_title('Mr Inbetween'), 'mr inbetween')

    def test_episode_names_in_several_release_styles(self):
        from select_windows import episode_of
        for name, episode in [('Greys Anatomy S07E01.avi', 's07e01'), ('90210 - S01E01E02.avi', 's01e01'),
                              ('castle.2009.801.hdtv-lol.mp4', 's08e01'), ('shameless.us.605.hdtv-lol[ettv].mp4', 's06e05'),
                              ('Big.Little.Lies.Part.3.HDTV.x264-SVA[PRiME].mkv', 's01e03'), ('Show 2x07 title.mkv', 's02e07'), ('one.tree.hill.0504-yestv.avi', 's05e04'),
                              ('Mr. Inbetween (2018) - S03E01 - Coulda Shoulda (1080p AMZN WEB-DL x265 Ghost).mkv', 's03e01'),
                              ('Hit.Man.2023.1080p.WEBRip.x264.AAC5.1-[YTS.MX].mp4', 'film')]:
            self.assertEqual(episode_of(name), episode, name)

    def test_inventory_lists_a_copied_episode_once(self):
        from select_windows import inventory
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for drive in ('f', 'h'):
                (root / drive / 'Castle').mkdir(parents=True)
                (root / drive / 'Castle' / f'Castle S01E01 {drive}.mkv').write_bytes(b'x')
            (root / 'h' / 'Castle' / 'castle.2009.801.hdtv.mp4').write_bytes(b'x')
            (root / 'The.Bear.S02.COMPLETE').mkdir()
            (root / 'The.Bear.S02.COMPLETE' / 'The.Bear.S02E01.mkv').write_bytes(b'x')
            files = inventory([root / 'f', root / 'h'], [root / 'The.Bear.S02.COMPLETE'])
            self.assertEqual(sorted((f['title'], f['episode']) for f in files),
                             [('castle', 's01e01'), ('castle', 's08e01'), ('the bear', 's02e01')])
            self.assertTrue(next(f for f in files if f['episode'] == 's01e01')['source'].endswith('f.mkv'))


if __name__ == '__main__':
    unittest.main()
