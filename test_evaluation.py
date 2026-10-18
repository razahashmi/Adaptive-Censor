import unittest
from core import default_policy
from evaluate import evaluate


class EvaluationTests(unittest.TestCase):
    def test_metrics_count_overlap_once_and_preserve_missed_intervals(self):
        policy=default_policy();policy.update(pre_padding=0,post_padding=0)
        predicted=[{'start':1,'end':3,'category':'language'},{'start':2,'end':4,'category':'language'}]
        gold=[{'start':0,'end':2,'category':'language'},{'start':5,'end':6,'category':'language'}]
        result=evaluate(predicted,gold,10,policy)['by_category']['language']
        self.assertEqual(result['missed_seconds'],2)
        self.assertEqual(result['unnecessary_filter_seconds'],2)
        self.assertAlmostEqual(result['time_recall'],1/3)
        self.assertEqual(result['events_at_least_95_percent_covered'],0)
        self.assertAlmostEqual(result['time_precision'],1/3)
        self.assertAlmostEqual(result['time_f1'],1/3)

    def test_unreviewed_time_is_not_counted_as_a_false_alarm(self):
        policy=default_policy();policy.update(pre_padding=0,post_padding=0)
        predicted=[{'start':1,'end':3,'category':'violence'}, {'start':8,'end':9,'category':'violence'}]
        gold=[{'start':1,'end':2,'category':'violence'}]
        report=evaluate(predicted,gold,10,policy,{'violence':[[0,4]]})['by_category']
        self.assertEqual(report['violence']['unnecessary_filter_seconds'],1)
        self.assertEqual(report['violence']['unreviewed_seconds'],6)
        self.assertEqual(report['violence']['time_precision'],.5)
        self.assertEqual(report['violence']['time_recall'],1)
        self.assertIsNone(report['nudity']['time_recall'])
        self.assertEqual(report['nudity']['reviewed_seconds'],0)

    def test_partial_events_and_invalid_reviewed_windows(self):
        policy=default_policy();policy.update(pre_padding=0,post_padding=0)
        gold=[{'start':1,'end':3,'category':'language'}]
        report=evaluate(gold,gold,10,policy,{'language':[[2,4]]})['by_category']['language']
        self.assertEqual(report['gold_events'],0)
        self.assertEqual(report['partially_reviewed_gold_events'],1)
        self.assertEqual(report['events_at_least_95_percent_covered'],0)
        self.assertEqual(report['time_recall'],1)
        for reviewed in [{'language':[[1,float('nan')]]},{'language':[[2,11]]},{'wrong':[]}]:
            with self.assertRaises(ValueError):
                evaluate([],[],10,policy,reviewed)
