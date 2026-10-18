"""Label-based scoring: exact second counts, exclusions, candidate recall and strata."""
import unittest

from evaluate_labels import bootstrap, difference, evaluate, score_window
from labeling import LABEL_CATEGORIES

WINDOW = dict(id='a', title='Show A', split='test', stratum='random', target=[2.0, 14.0], clip_duration=16.0)


def event(category, start, end, **extra):
    return dict(dict(category=category, start=start, end=end, confidence=1, severity=2, enabled=True,
                     source='Qwen verified frame', review_required=False), **extra)


def labels(**categories):
    result = {c: dict(status='reviewed', intervals=[], ambiguous=[]) for c in LABEL_CATEGORIES}
    for category, value in categories.items():
        result[category] = dict(dict(status='reviewed', intervals=[], ambiguous=[]), **value)
    return dict(categories=result, drafts_shown=False)


class ScoreTests(unittest.TestCase):
    def test_difference(self):
        self.assertEqual(difference([[0, 10]], [[2, 3], [5, 6]]), [[0, 2], [3, 5], [6, 10]])
        self.assertEqual(difference([[0, 1]], [[0, 1]]), [])

    def test_seconds_exclusions_and_assisted_review(self):
        gold = labels(violence=dict(intervals=[[4, 8]], ambiguous=[[10, 11]]))
        prediction = dict(events=[event('violence', 6, 9), event('violence', 10, 11), event('violence', 12, 13, review_required=True)],
                          trace=[dict(stage='scene_sequence', category='violence', start=3.5, end=5)])
        counts = score_window(WINDOW, gold, prediction, 'violence', padding=False)
        self.assertAlmostEqual(counts['gold'], 4)
        self.assertAlmostEqual(counts['tp'], 2)
        self.assertAlmostEqual(counts['fp'], 1)          # 8-9; the ambiguous 10-11 is excluded
        self.assertAlmostEqual(counts['fn'], 2)
        self.assertAlmostEqual(counts['assisted_fp'], 2)  # review item 12-13 counts only when assisted
        self.assertAlmostEqual(counts['candidate_tp'], 1)  # 4-5: the proposal, even though it was never accepted
        self.assertEqual((counts['events'], counts['events_hit'], counts['events_covered']), (1, 1, 0))

    def test_candidates_exclude_second_stage_and_weak_similarity(self):
        gold = labels(gore=dict(intervals=[[2, 14]]))
        trace = [dict(stage='scene_evidence', category='gore', start=2, end=4, state='absent'),
                 dict(stage='clip', category='gore', start=4, end=6, score=.5),
                 dict(stage='clip', category='gore', start=6, end=7, score=.9)]
        counts = score_window(WINDOW, gold, dict(events=[], trace=trace), 'gore', padding=False)
        self.assertAlmostEqual(counts['candidate_tp'], 1)
        self.assertAlmostEqual(counts['candidate_tp_clip'], 1)

    def test_unsure_is_unknown_and_clean_false_alarms_are_counted(self):
        unsure = labels(kissing=dict(status='unsure'))
        self.assertIsNone(score_window(WINDOW, unsure, dict(events=[], trace=[]), 'kissing', True))
        counts = score_window(WINDOW, labels(), dict(events=[event('nudity', 5, 5.5, confidence=.9, source='NudeNet')], trace=[]), 'nudity', True)
        self.assertEqual((counts['clean_windows'], counts['clean_windows_filtered']), (1, 1))
        self.assertAlmostEqual(counts['fp'], 1.15)  # .3 before and .35 after with balanced padding

    def test_project_report_separates_strata_and_requires_complete_labels(self):
        windows = [WINDOW, dict(WINDOW, id='b', title='Show B', stratum='mined:violence'), dict(WINDOW, id='c', title='Show C')]
        project = dict(windows=windows)
        all_labels = {'a': labels(violence=dict(intervals=[[4, 8]])), 'b': labels(violence=dict(intervals=[[2, 4]])),
                      'c': dict(categories={'violence': dict(status='reviewed', intervals=[], ambiguous=[])}, drafts_shown=False)}
        predictions = {w: dict(events=[event('violence', 4, 8)], trace=[]) for w in 'abc'}
        report = evaluate(project, all_labels, predictions, 'test', padding=False)
        self.assertEqual(report['windows'], 2)
        violence = report['categories']['violence']
        self.assertAlmostEqual(violence['all']['recall'], 4 / 6)
        self.assertAlmostEqual(violence['random']['recall'], 1)
        self.assertAlmostEqual(violence['mined']['recall'], 0)

    def test_bootstrap_resamples_titles(self):
        titles = {'a': dict(gold=4, tp=4, fp=0, candidate_tp=4), 'b': dict(gold=4, tp=0, fp=0, candidate_tp=4)}
        interval = bootstrap(titles)
        self.assertEqual(interval['candidate_recall'], [1, 1])
        self.assertLess(interval['recall'][0], interval['recall'][1])


if __name__ == '__main__':
    unittest.main()
