import copy
import unittest
from evaluate_samples import evaluate_samples, rates


class SampleEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.cases=[dict(id='a',title='one',start=0,end=6,times=[0,.5,1,1.5])]
        self.gold=dict(definitions={'violence':'active attack'},
                       cases={'a':{'violence':dict(positive=[1],unknown=[3])}})
        self.pred=dict(results=[dict(id='a',events=[dict(start=.5,end=1,category='violence',confidence=1)])])

    def score(self):return evaluate_samples(self.cases,self.gold,self.pred)

    def test_half_open_boundaries_unknowns_and_padding_are_separate(self):
        result=self.score()['modes']
        raw=result['without_padding']['by_category']['violence']
        self.assertEqual([raw[k] for k in ('tp','fp','fn','tn','unknown')],[1,0,0,2,1])
        self.assertEqual(result['with_padding']['by_category']['violence']['fp'],1)

    def test_many_clean_samples_do_not_hide_zero_recall(self):
        result=rates(dict(tp=0,fp=0,fn=1,tn=99,unknown=0))
        self.assertEqual(result['overall_accuracy'],.99)
        self.assertEqual(result['recall'],0)
        self.assertEqual(result['sample_target_status'],'below_target')
        self.assertEqual(rates(dict(tp=0,fp=0,fn=0,tn=100,unknown=0))['sample_target_status'],'insufficient_evidence')

    def test_review_candidate_is_a_miss_on_a_known_positive(self):
        self.pred['results'][0]['events'][0]['review_required']=True
        self.assertEqual(self.score()['modes']['without_padding']['by_category']['violence']['fn'],1)

    def test_partial_duplicate_and_unreviewed_predictions_are_rejected(self):
        for rows in [[],self.pred['results']*2,[dict(id='unreviewed',events=[])]]:
            with self.subTest(rows=rows),self.assertRaises(ValueError):
                evaluate_samples(self.cases,self.gold,dict(results=rows))

    def test_invalid_or_overlapping_labels_do_not_silently_score(self):
        for change in [dict(positive=[1],unknown=[1]),dict(positive=[9],unknown=[]),
                       dict(positive=[True],unknown=[])]:
            gold=copy.deepcopy(self.gold);gold['cases']['a']['violence']=change
            with self.assertRaises(ValueError):evaluate_samples(self.cases,gold,self.pred)
        self.cases[0]['times']=[.5,.5]
        with self.assertRaises(ValueError):self.score()

    def test_manual_confirmations_are_not_model_accuracy(self):
        self.pred['results'][0]['events'][0]['confirmed']=True
        with self.assertRaises(ValueError):self.score()

    def test_title_breakdown_is_not_lost_in_aggregate(self):
        self.cases.append(dict(id='b',title='two',start=0,end=6,times=[.5]))
        self.gold['cases']['b']={'violence':dict(positive=[0],unknown=[])}
        self.pred['results'].append(dict(id='b',events=[]))
        result=self.score()['modes']['without_padding']
        self.assertEqual(result['by_category']['violence']['recall'],.5)
        self.assertEqual(result['by_title']['one']['violence']['recall'],1)
        self.assertEqual(result['by_title']['two']['violence']['recall'],0)


if __name__=='__main__':unittest.main()
