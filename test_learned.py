"""Learned-head plumbing on synthetic features: labels, thresholds, title-held-out training."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from labeling import LABEL_CATEGORIES, write_json
import learned
import learned_runtime
from learned import grid_labels, grid_times, thresholds, audio_candidates


def reviewed(**intervals):
    return {'categories': {c: dict(status='reviewed', intervals=intervals.get(c, []), ambiguous=[]) for c in LABEL_CATEGORIES},
            'drafts_shown': False}


class HeadTests(unittest.TestCase):
    def test_grid_labels_cover_only_the_reviewed_target(self):
        window = dict(target=[2.0, 14.0], clip_duration=16.0)
        times = grid_times(window)
        labels = reviewed(violence=[[4, 5]])
        labels['categories']['violence']['ambiguous'] = [[8, 9]]
        y = grid_labels(window, labels, 'violence', times)
        self.assertTrue((y[times < 2] == -1).all() and (y[times >= 14] == -1).all())
        self.assertEqual(int((y == 1).sum()), 4)
        self.assertTrue((y[(times >= 8) & (times < 9)] == -1).all())
        labels['categories']['gore']['status'] = 'unsure'
        self.assertTrue((grid_labels(window, labels, 'gore', times) == -1).all())

    def test_thresholds_reach_targets_or_refuse_to_decide(self):
        scores = np.linspace(1, 0, 100)
        y = np.r_[np.ones(40), np.zeros(20), np.ones(2), np.zeros(38)].astype(int)
        candidate, decision = thresholds(scores, y)
        self.assertGreaterEqual(y[scores >= candidate].sum() / y.sum(), learned.CANDIDATE_RECALL)
        self.assertGreaterEqual(y[scores >= decision].mean(), learned.DECISION_PRECISION)
        self.assertIsNone(thresholds(scores, (np.arange(100) % 2 == 0).astype(int))[1])

    def test_audio_events_become_violence_candidates(self):
        labels = np.array(['Speech', 'Gunshot, gunfire', 'Music'])
        features = dict(times=np.array([1., 1.5]), labels=labels,
                        probabilities=np.array([[.9, .05, .1], [.2, .6, .3]], dtype=np.float16))
        entries = audio_candidates(features, .15)
        self.assertEqual([(e['category'], e['sample_time'], e['detail']) for e in entries], [('violence', 1.5, 'Gunshot, gunfire')])

    def test_selective_verification_only_checks_undecided_spans(self):
        from unittest.mock import Mock
        trace = [dict(stage='classifier', category='violence', sample_time=t, start=t - .125, end=t + .125, score=1)
                 for t in (3.125, 3.375, 3.625, 3.875, 4.125, 4.375, 6.125, 9.125)]
        trace.append(dict(stage='scene_sequence', category='gore', sample_time=5, start=4.75, end=5.25))
        decided = [dict(category='violence', start=9, end=9.5, review_required=False)]
        spans = learned_runtime.uncertain_spans(dict(trace=trace, events=decided))
        self.assertEqual(spans, [('violence', 3.125, 3.875), ('violence', 4.125, 4.375), ('violence', 6.125, 6.125)])
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            write_json(run / 'w.json', dict(window='w', trace=trace, events=decided))
            verifier = Mock()
            verifier.check_frame.side_effect = [dict(state='present', label='person_striking_person', observation='Punch lands'),
                                                dict(state='absent', label='none', observation='No contact'),
                                                dict(state='uncertain', label='unclear', observation='Obscured')]
            burst = Mock(return_value=([(0, None)] * 5, 2))
            events = learned.verify(run, dict(id='w', target=[2.0, 14.0]), Path('clip.mp4'), verifier, burst)
            self.assertEqual([(e['start'], e['review_required']) for e in events], [(3.0, False), (6.0, True)])
            self.assertEqual(burst.call_args_list[0].args[1], 3.5)

    def test_promotion_needs_a_gain_without_large_losses(self):
        def report(**values):
            return {'benchmark_identity':'same', 'labelers':{'human':100}, 'categories': {c: {'positive_titles':10, 'clean_titles':10,
                    'title_bootstrap_95':{'precision':[.95,1], 'recall':[.95,1]},
                    'all': dict(counts=dict(gold=10, events=30, fp=0, clean_windows=100), clean_window_false_alarm=0,
                    precision=values.get(c, (.96, .5))[0], recall=values.get(c, (.96, .5))[1])} for c in LABEL_CATEGORIES}}
        base = report()
        self.assertTrue(learned.promotion_check(base, report(violence=(.97, .96)))[0])
        self.assertFalse(learned.promotion_check(base, report(violence=(.78, .7)))[0])
        self.assertFalse(learned.promotion_check(base, report())[0])
        self.assertFalse(learned.promotion_check(base, report(violence=(.7, .9)))[0])
        self.assertFalse(learned.promotion_check(base, report(violence=(.8, .9), gore=(.8, .4)))[0])

    def test_category_choice_enables_only_categories_that_improve(self):
        def report(**values):
            return {'categories': {c: {'all': dict(counts=dict(gold=10 if c != 'gore' else 0), precision=values.get(c, (.8, .5))[0],
                                                   recall=values.get(c, (.8, .5))[1])} for c in LABEL_CATEGORIES}}
        enabled, rows = learned.category_choice(report(), report(violence=(.6, .9), kissing=(.97, .7), gore=(.9, .9)))
        self.assertEqual(enabled, ['kissing'])
        self.assertEqual({r['category']: r['enabled'] for r in rows}['gore'], False)

    def test_disabled_categories_produce_no_candidates_or_decisions(self):
        head = dict(mean=[0], scale=[1], coef=[1], intercept=0, candidate_threshold=-10, decision_threshold=-10)
        heads = dict(grid=.25, heads=dict(violence=head, kissing=head), enabled=['kissing'])
        entries, events = learned_runtime.head_outputs(heads, np.array([1.0, 1.25]), np.zeros((2, 1)))
        self.assertEqual({e['category'] for e in entries + events}, {'kissing'})
        entries, _ = learned_runtime.head_outputs(dict(heads, enabled=None), np.array([1.0]), np.zeros((1, 1)))
        self.assertEqual({e['category'] for e in entries}, {'violence', 'kissing'})

    def _engine_with_heads(self, folder, promoted=True):
        from unittest.mock import patch
        from analysis_engine import Models
        head = lambda decision: dict(mean=[0, 0], scale=[1, 1], coef=[10, 0], intercept=-5,
                                     candidate_threshold=0, decision_threshold=decision)
        write_json(folder / 'heads' / 'h1.json', dict(kinds=['vjepa'], grid=.25, promoted=promoted,
                                                     heads=dict(violence=head(4), gore=head(None))))
        with patch('analysis_engine.prepare_environment', return_value=folder):
            return Models(evaluate_unpromoted=promoted)

    def test_player_pass_decides_follows_review_setting_and_verifies_only_candidates(self):
        from unittest.mock import Mock, patch
        from analysis_engine import ScanConfig

        class Features:
            def __init__(self, models, root):
                pass

            def compute(self, kind, path, start, end, centers=None, duration=None):
                signal = ((centers >= 2) & (centers < 3)).astype(np.float32)
                return dict(times=centers, embeddings=np.stack([signal, np.zeros_like(signal)], 1))

        with tempfile.TemporaryDirectory() as directory:
            models = self._engine_with_heads(Path(directory))
            verifier = Mock()
            verifier.verify.return_value = dict(tags=[])
            verifier.check_frame.return_value = dict(state='present', label='open_wound', observation='Open wound visible')
            frames = [(float(t), np.zeros((12, 12, 3), dtype=np.uint8)) for t in np.arange(0, 8, .5)]
            with patch('learned_runtime.FeatureModels', Features), patch('analysis_engine.sample_frames', return_value=frames), \
                    patch('analysis_engine.sample_burst', return_value=([(0, None)] * 5, 2)), patch.object(models, 'verifier', return_value=verifier):
                automatic = models.analyze(Path('clip.mp4'), 0, 6, 10, ScanConfig(speech=False, nudity=False, scenes=False,
                                           learned_heads='h1', scene_filtering='automatic'), False)
                reviewed = models.analyze(Path('clip.mp4'), 0, 6, 10, ScanConfig(speech=False, nudity=False, scenes=False,
                                          learned_heads='h1'), False)
                verified = models.analyze(Path('clip.mp4'), 0, 6, 10, ScanConfig(speech=False, nudity=False, learned_heads='h1',
                                          scene_filtering='automatic', learned_verification='selective'), False)
            decided = [e for e in automatic if e['source'] == 'Learned head']
            self.assertTrue(decided and all(e['category'] == 'violence' and 1.75 <= e['sample_time'] < 3.25 for e in decided))
            self.assertTrue(all(not e['review_required'] for e in decided))
            self.assertTrue(all(e['review_required'] for e in reviewed))
            gore = [e for e in verified if e['source'] == 'Verified learned candidate']
            self.assertTrue(gore and all(e['category'] == 'gore' for e in gore))
            self.assertTrue(all(call.args[1] == 'gore' for call in verifier.check_frame.call_args_list))
            verifier.verify.reset_mock(); verifier.check_frame.reset_mock()
            with patch('learned_runtime.FeatureModels', Features), \
                    patch('analysis_engine.sample_frames', side_effect=AssertionError('learned mode decodes no 2 fps frames')), \
                    patch('analysis_engine.sample_burst', return_value=([(0, None)] * 5, 2)), patch.object(models, 'verifier', return_value=verifier):
                learned = models.analyze(Path('clip.mp4'), 0, 6, 10, ScanConfig(speech=False, nudity=False, scene_mode='learned',
                                         learned_heads='h1', scene_filtering='automatic', learned_verification='selective'), False)
                with self.assertRaisesRegex(RuntimeError, 'needs a promoted'):
                    models.analyze(Path('clip.mp4'), 0, 6, 10, ScanConfig(speech=False, nudity=False, scene_mode='learned'), False)
            verifier.verify.assert_not_called()
            self.assertTrue(verifier.check_frame.called)
            self.assertEqual({e['source'] for e in learned}, {'Learned head', 'Verified learned candidate'})

    def test_player_refuses_heads_that_were_not_promoted(self):
        from analysis_engine import ScanConfig
        with tempfile.TemporaryDirectory() as directory:
            models = self._engine_with_heads(Path(directory), promoted=False)
            with self.assertRaisesRegex(RuntimeError, 'not been promoted'):
                models.analyze(Path('clip.mp4'), 0, 6, 10, ScanConfig(speech=False, nudity=False, scenes=False, learned_heads='h1'), False)

    @unittest.skipUnless(importlib.util.find_spec('sklearn'), 'training needs the .train-venv environment')
    def test_training_holds_out_titles_and_finds_a_separable_signal(self):
        rng = np.random.default_rng(3)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / 'labeling' / 'train-x'
            windows = []
            for i in range(12):
                window = dict(id=f'w{i}', title=f'title {i % 6}', split='train', target=[2.0, 14.0], clip_duration=16.0)
                windows.append(window)
                labels = reviewed(violence=[[4, 7]] if i % 2 else [])
                write_json(folder / 'labels' / f'w{i}.json', labels)
                times = grid_times(window)
                signal = ((times >= 4) & (times < 7) & bool(i % 2)).astype(np.float32)
                embeddings = rng.normal(size=(len(times), 8)).astype(np.float32)
                embeddings[:, 0] += 4 * signal
                (folder / 'features' / 'vjepa').mkdir(parents=True, exist_ok=True)
                np.savez(folder / 'features' / 'vjepa' / f'w{i}.npz', times=times, embeddings=embeddings)
            write_json(folder / 'project.json', dict(name='train-x', windows=windows))
            result = learned.train([folder], ['vjepa'], allow_legacy=True)
            head = result['heads']['violence']
            self.assertGreater(head['cv_average_precision'], .9)
            self.assertIn('skipped', result['heads']['gore'])
            entries, events = learned.apply(folder, windows[1], result, allow_legacy=True)
            self.assertTrue(entries and all(e['stage'] == 'classifier' for e in entries))
            self.assertTrue(all(4 - .2 < e['sample_time'] < 7 + .2 for e in events))


if __name__ == '__main__':
    unittest.main()
