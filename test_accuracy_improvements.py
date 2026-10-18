"""Exact-player input parity, confirmation vetoes and independent calibration."""
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock,patch
import numpy as np

from labeling import write_json


class AccuracyImprovementTests(unittest.TestCase):
    def test_collection_recovers_real_frames_when_implied_frame_grid_is_empty(self):
        from active_training import screening_frames
        frames=[(float(t),np.array([t])) for t in np.arange(.021,12,.0417)]
        with patch('analysis_engine.native_frames',side_effect=RuntimeError('No sampled frames')),patch(
                'analysis_engine.decoded_range',return_value=(23.976,frames)):
            result=screening_frames(Path('fixture'),0,12)
        self.assertEqual(len(result),12)
        self.assertEqual(len(set(t for t,_ in result)),12)
        self.assertTrue(all(any(p is frame[1] for frame in frames) for _,p in result))
        with patch('analysis_engine.native_frames',side_effect=RuntimeError('No frames')),patch(
                'analysis_engine.decoded_range',return_value=(24,[])):
            with self.assertRaises(RuntimeError):screening_frames(Path('fixture'),0,12)

    def test_array_reader_preserves_extraction_contract_across_environments(self):
        from copy import deepcopy
        from player_inputs import compatible_reader
        recorded=dict(source='bound-source',code={'decoder':'bound-code'},assets={'encoder':'bound-model'},
                      dependencies={'numpy':'2.0','av':'14','onnxruntime':'1.30','onnxruntime-gpu':'1.23'})
        current=deepcopy(recorded)
        current['dependencies'].update(onnxruntime='1.23',**{'onnxruntime-gpu':None})
        self.assertTrue(compatible_reader(recorded,current))
        current['dependencies']['numpy']='2.1'
        self.assertFalse(compatible_reader(recorded,current))
        current=deepcopy(recorded);current['assets']['encoder']='different-model'
        self.assertFalse(compatible_reader(recorded,current))

    def test_cached_training_scores_equal_playback_across_chunk_boundaries(self):
        from analysis_engine import Models,ScanConfig
        from player_inputs import prepare,load_matrix
        from learned_runtime import head_scores
        class Features:
            def compute(self,kind,path,start,end,centers=None,duration=None):
                centers=np.asarray(centers)
                return dict(times=centers,embeddings=centers[:,None])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'labeling/p';(folder/'clips').mkdir(parents=True)
            (folder/'clips/w.mp4').write_bytes(b'fixture')
            write_json(root/'learned-runtime/manifest.json',dict(models={}))
            window=dict(id='w',clip_duration=13.)
            features=Features();models=Mock(root=root);models.learned_features.return_value=features
            times,matrix=prepare(folder,window,models,['vjepa'],[-.5,.5])
            _,cached=load_matrix(folder,window,['vjepa'],[-.5,.5])
            self.assertTrue(np.array_equal(cached,matrix))
            # At each boundary the owned offset clamps, matching the player.
            before=np.where(times==5.875)[0][0];after=np.where(times==6.125)[0][0]
            self.assertEqual(matrix[before,2],6.);self.assertEqual(matrix[after,1],6.)
            head=dict(mean=[0]*3,scale=[1]*3,coef=[1,10,100],intercept=0,
                      candidate_threshold=-1,decision_threshold=None)
            heads=dict(kinds=['vjepa'],grid=.25,temporal_offsets=[-.5,.5],heads={'violence':head})
            engine=Models.__new__(Models);engine.root=root;engine.backends={};engine.lock=threading.RLock()
            engine.sessions={};engine.evaluate_unpromoted=True
            engine.heads=lambda name:heads;engine.learned_features=lambda:features
            trace=[];config=ScanConfig(scenes=False,speech=False,nudity=False,learned_heads='fixture')
            for start in (0,6,12):engine._learned_events(folder/'clips/w.mp4',start,min(start+6,13),13,config,trace.append)
            self.assertTrue(np.array_equal([t['sample_time'] for t in trace],times))
            np.testing.assert_array_equal([t['score'] for t in trace],head_scores(head,cached))
            with np.load(folder/'player-inputs'/next((folder/'player-inputs').iterdir()).name/'w.npz') as data:
                altered=data['matrix'].copy()
            altered[0,0]+=1
            path=next((folder/'player-inputs').rglob('*.npz'));np.savez(path,times=times,matrix=altered)
            with self.assertRaisesRegex(ValueError,'modified'):load_matrix(folder,window,['vjepa'],[-.5,.5])

    def test_confirmation_rejects_confident_decision_and_retains_uncertain_for_review(self):
        from learned_confirmation import confirm_outputs
        entries=[dict(category='violence',stage='classifier',sample_time=1.125,start=1,end=1.25,score=20)]
        events=[dict(category='violence',sample_time=1.125,start=1,end=1.25,review_required=False)]
        evidence=lambda state:Mock(return_value=dict(state=state,label='none',observation='fixture'))
        trace,result=confirm_outputs(entries,events,evidence('absent'),0,6)
        self.assertFalse(result);self.assertEqual(trace[0]['state'],'absent')
        _,result=confirm_outputs(entries,events,evidence('uncertain'),0,6)
        self.assertTrue(result[0]['review_required'])
        _,result=confirm_outputs(entries,events,evidence('present'),0,6)
        self.assertFalse(result[0]['review_required'])

    def test_confirmation_budget_never_leaves_unchecked_decisions_automatic(self):
        from learned_confirmation import confirm_outputs
        entries=[dict(category='violence',stage='classifier',sample_time=t,start=t-.125,end=t+.125,score=t)
                 for t in np.arange(.125,6,.25)]
        events=[dict(e,review_required=False) for e in entries]
        check=Mock(return_value=dict(state='absent',label='none',observation='No attack'))
        _,result=confirm_outputs(entries,events,check,0,6,maximum_checks=2)
        self.assertEqual(check.call_count,2);self.assertTrue(result)
        self.assertTrue(all(e['review_required'] for e in result))

    def test_calibration_does_not_count_many_frames_as_independent_events(self):
        from human_calibration import operating_points,partitions
        scores=np.r_[np.ones(40),np.zeros(100)];y=np.r_[np.ones(40),np.zeros(100)]
        groups=np.array(['one']*40+['negative']*100);events=np.array(['one-event']*40+['']*100)
        result=operating_points(scores,y,groups,events)
        self.assertIsNone(result['decision_threshold'])
        self.assertEqual(result['calibration_positive_events'],1)
        groups[:40]=np.tile(['aaa','bbb','ccc','ddd'],10)
        events[:40]=np.tile(['one-event','two-event','six-event','ten-event','yes-event'],8)
        self.assertIsNotNone(operating_points(scores,y,groups,events)['decision_threshold'])
        rows=[(None,dict(title=f'Title {i}'),None) for i in range(40)]
        split=partitions(rows)
        self.assertFalse(split['fit']&split['tune'] or split['fit']&split['calibration'] or split['tune']&split['calibration'])
        self.assertEqual(len(set.union(*split.values())),40)

    def test_balanced_calibration_reserves_supported_positive_titles_deterministically(self):
        from copy import deepcopy
        from human_calibration import balanced_partitions
        from labeling import LABEL_CATEGORIES
        rows=[]
        for i in range(20):
            window=dict(id=f'w{i}',title=f'Title {i}',clip_duration=6.,target=[0.,6.])
            label=dict(categories={c:dict(status='reviewed',ambiguous=[],
                intervals=[[1.,2.],[3.,4.]] if i<10 else []) for c in LABEL_CATEGORIES})
            rows.append((None,window,label))
        before=deepcopy(rows)
        split=balanced_partitions(rows,trials=10000)
        self.assertEqual(rows,before)
        self.assertEqual(split,balanced_partitions(list(reversed(rows)),trials=10000))
        self.assertFalse(split['fit']&split['calibration'])
        self.assertEqual(len(split['fit']|split['calibration']),20)
        positive={f'title {i}' for i in range(10)}
        for name in ('fit','calibration'):
            self.assertGreaterEqual(len(split[name]&positive),3)

    def test_balanced_calibration_refuses_impossible_supported_title_constraints(self):
        from human_calibration import balanced_partitions
        from labeling import LABEL_CATEGORIES
        rows=[]
        # Four reserved titles cannot contain three titles from each of two
        # disjoint positive sets. Refuse instead of quietly dropping support.
        for i in range(20):
            window=dict(id=f'w{i}',title=f'Title {i}',clip_duration=6.,target=[0.,6.])
            categories={c:dict(status='reviewed',ambiguous=[],intervals=[]) for c in LABEL_CATEGORIES}
            for c,inside in [('nudity',i<8),('violence',8<=i<16)]:
                if inside:categories[c]['intervals']=[[1.,2.],[3.,4.]]
            rows.append((None,window,dict(categories=categories)))
        with self.assertRaisesRegex(ValueError,'No supported whole-title'):
            balanced_partitions(rows,trials=10000)


if __name__=='__main__':unittest.main()
