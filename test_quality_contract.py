"""Regression coverage for benchmark validity, concurrency and saved model identity."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np
from analysis_engine import Engine, MediaJob, ScanConfig
from core import compile_plan, default_policy
from evaluate_labels import evaluate
from frame_cache import BoundedCache
from learned import promotion_check, thresholds
from learned_runtime import FeatureModels
from learned_runtime import feature_matrix
from labeling import LABEL_CATEGORIES, write_json
from predict_project import ensure_run, predict_window
from provenance import digest, head_identity, qualified


def fixture():
    windows = [dict(id=str(i), title=f'Title {i}', split='test', target=[0, 4], stratum='random', clip_duration=4)
               for i in range(2)]
    labels = {w['id']: dict(drafts_shown=False, labeler='human', categories={c:dict(status='reviewed', intervals=[], ambiguous=[])
                                                                          for c in LABEL_CATEGORIES}) for w in windows}
    predictions = {w['id']: dict(window=w['id'], events=[], trace=[]) for w in windows}
    return dict(windows=windows), labels, predictions


class QualityTests(unittest.TestCase):
    def test_title_weights_keep_class_totals_and_equalize_title_influence(self):
        from learned import title_class_weights
        y=np.array([1,1,1,0,0,0,0,0])
        groups=np.array(['a','a','b','a','a','b','b','b'])
        weights=title_class_weights(y,groups)
        for label in (0,1):
            self.assertAlmostEqual(weights[y==label].sum(),float((y==label).sum()))
            self.assertAlmostEqual(weights[(y==label)&(groups=='a')].sum(),weights[(y==label)&(groups=='b')].sum())

    def test_changed_projection_invalidates_head_qualification(self):
        heads=dict(kinds=['clip'],grid=.25,heads={},enabled=[],projection={'components':[[1,0]]})
        heads['promoted']=dict(qualified=True,verification='off',head_identity=head_identity(heads))
        self.assertTrue(qualified(heads))
        heads['projection']['components'][0][0]=2
        self.assertFalse(qualified(heads))

    def test_saved_default_config_keeps_its_fingerprint(self):
        config=ScanConfig()
        saved=json.loads(json.dumps(config.model_dump()))
        restored=ScanConfig(**saved)
        self.assertEqual(digest(config.model_dump()),digest(restored.model_dump()))

    def test_derived_run_is_incomplete_until_stage_finishes_and_never_overwrites(self):
        from learned import derive_run
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            write_json(folder/'predictions/base/run.json',dict(pipeline='player',identity='base'))
            write_json(folder/'predictions/base/a.json',dict(window='a',events=[],trace=[],run_identity='base'))
            output=derive_run(folder,'base','candidate',dict(pipeline='cached-experiment',heads='h'))
            manifest=json.loads((output/'run.json').read_text())
            self.assertFalse(manifest['complete'])
            self.assertEqual(manifest['pipeline'],'cached-experiment')
            self.assertEqual(json.loads((output/'a.json').read_text())['run_identity'],manifest['identity'])
            with self.assertRaises(FileExistsError):derive_run(folder,'base','candidate',{})

    def test_feature_cache_requires_source_and_preprocessing_identity(self):
        from features import feature_path,feature_contract,load_features
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)/'labeling'/'p'
            path=feature_path(folder,'clip','w');path.parent.mkdir(parents=True)
            np.savez(path,times=np.array([0.]),embeddings=np.ones((1,2)))
            with self.assertRaisesRegex(ValueError,'Unversioned'):load_features(folder,'clip','w')
            self.assertIsNotNone(load_features(folder,'clip','w',allow_legacy=True))
            contract=feature_contract(folder,'clip','w')
            write_json(path.with_suffix('.json'),dict(contract,identity=digest(contract)))
            self.assertIsNotNone(load_features(folder,'clip','w'))
            (folder/'clips').mkdir();(folder/'clips/w.mp4').write_bytes(b'new source')
            with self.assertRaisesRegex(ValueError,'Stale'):load_features(folder,'clip','w')

    def test_inflight_results_are_rejected_when_head_changes(self):
        started,release=threading.Event(),threading.Event()
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            write_json(root/'heads/h.json',dict(kinds=[],grid=.25,heads={},enabled=[]))
            class Fake:
                def __init__(self,**kwargs):self.root=root;self.backends={};self.warnings=[]
                def analyze(self,*args):started.set();release.wait(3);return []
            with patch('analysis_engine.Models',Fake):engine=Engine(workers=1)
            try:
                job=MediaJob('x','x',root/'missing.mp4',6,False,active=True,config=ScanConfig(speech=False,nudity=False,scenes=False,learned_heads='h'))
                with engine.condition:engine.jobs[job.id]=job;engine.condition.notify_all()
                self.assertTrue(started.wait(2))
                with engine.condition:
                    job.active=False
                    write_json(root/'heads/h.json',dict(kinds=[],grid=.25,heads={},enabled=['violence']))
                release.set()
                deadline=time.monotonic()+2
                while job.running and time.monotonic()<deadline:time.sleep(.01)
                self.assertEqual(job.chunks,{})
                self.assertEqual(job.generation,1)
            finally:release.set();engine.close()

    def test_unqualified_heads_stay_in_review_and_default_verification_is_off(self):
        from test_learned import HeadTests
        with tempfile.TemporaryDirectory() as temporary:
            models=HeadTests()._engine_with_heads(Path(temporary))
            models.evaluate_unpromoted=False
            features=Mock();features.compute.return_value=dict(times=np.array([0.,6.]),embeddings=np.array([[1.,0.],[1.,0.]],dtype=np.float32))
            verifier=Mock()
            with patch.object(models,'learned_features',return_value=features), patch.object(models,'verifier',return_value=verifier):
                events=models.analyze(Path('unused'),0,6,6,ScanConfig(speech=False,nudity=False,scene_mode='learned',
                                      learned_heads='h1',scene_filtering='automatic'),False)
            self.assertTrue(events)
            self.assertTrue(all(e['review_required'] for e in events))
            verifier.check_frame.assert_not_called()

    def test_merged_scene_events_do_not_cross_shot_bounds(self):
        from analysis_engine import merge_scene_events
        events=[dict(start=0,end=1,category='violence',source='Qwen verified frame',review_required=False,time_bounds=[0,1]),
                dict(start=1,end=2,category='violence',source='Qwen verified frame',review_required=False,time_bounds=[1,2])]
        self.assertEqual(len(merge_scene_events(events)),2)

    def test_pixel_metrics_separate_coverage_and_clean_disturbance(self):
        from evaluate_masks import counts, rates
        c=counts([[1,1],[0,0]],[[1,0],[1,0]])
        self.assertEqual(c['tp'],1); self.assertEqual(c['fp'],1); self.assertEqual(c['fn'],1)
        self.assertEqual(rates(c)['coverage'],.5)
        self.assertEqual(rates(c)['clean_pixel_disturbance'],.5)
        self.assertEqual(rates(c)['iou'],1/3)

    def test_batch_cache_keeps_duplicate_and_overlapping_inputs_in_order(self):
        cache=BoundedCache(1024)
        calls=[]
        def load(indices):
            calls.append(indices)
            return [np.array([i]) for i in indices]
        values=cache.get_many(['a','b','a'],load)
        self.assertEqual(calls,[[0,1]])
        self.assertEqual([int(v[0]) for v in values],[0,1,0])
        values=cache.get_many(['b','c'],load)
        self.assertEqual(calls[-1],[1])
        self.assertEqual([int(v[0]) for v in values],[1,1])

    def test_temporal_motion_columns_keep_order(self):
        data={'vjepa':dict(times=np.array([0.,1.,2.]),embeddings=np.array([[1],[2],[3]],dtype=np.float32))}
        matrix=feature_matrix(data,np.array([1.]),['vjepa'],[-1,1])
        np.testing.assert_array_equal(matrix,[[2,1,3]])

    def test_roi_gate_can_only_demote_supported_regions(self):
        from roi_gate import AnatomyGate
        gate=AnatomyGate.__new__(AnatomyGate)
        gate.models=Mock();gate.models.clip_embeddings.return_value=np.array([[0.],[2.]])
        gate.embedding_cache=BoundedCache(1024)
        gate.head=dict(mean=[0],scale=[1],coef=[1],intercept=0,decision_threshold=1)
        frames=[(0,np.zeros((20,20,3),dtype=np.uint8)),(1,np.full((20,20,3),255,dtype=np.uint8))]
        hit=dict(box=[2,2,8,8],score=.9,**{'class':'FEMALE_BREAST_EXPOSED'})
        candidates=[(0,0,hit,[1,2,3],.9,0,'raw'),(1,0,hit,[1,2,3],.9,0,'raw'),(0,0,hit,[],.3,0,'raw')]
        result=gate.refine(frames,candidates)
        self.assertEqual([len(r[3]) for r in result],[0,3,0])
        self.assertEqual(len(result),len(candidates))
        self.assertEqual(gate.refine(frames,candidates),result)
        self.assertEqual(gate.models.clip_embeddings.call_count,1)

    def test_region_gate_leaves_untrained_anatomy_classes_alone(self):
        from roi_gate import AnatomyGate
        gate=AnatomyGate.__new__(AnatomyGate);gate.models=Mock()
        gate.head=dict(supported_classes=['FEMALE_BREAST_EXPOSED'])
        candidate=(0,0,dict(box=[2,2,8,8],score=.9,**{'class':'BUTTOCKS_EXPOSED'}),[1,2,3],.9,0,'raw')
        self.assertEqual(gate.refine([(0,np.zeros((20,20,3),dtype=np.uint8))],[candidate]),[candidate])
        gate.models.clip_embeddings.assert_not_called()

    def test_body_context_feature_order_and_local_geometry_match_training(self):
        from roi_gate import AnatomyGate,crop,body_crop
        gate=AnatomyGate.__new__(AnatomyGate);gate.models=Mock();gate.embedding_cache=BoundedCache(1024)
        gate.head=dict(mean=[0,0],scale=[1,1],coef=[-1,1],intercept=0,decision_threshold=0,feature_views=['local','body'])
        pixels=np.zeros((100,200,3),dtype=np.uint8);pixels[40:60,100:120]=255
        gate.models.clip_embeddings.return_value=np.array([[2.],[1.]])
        candidate=(0,0,dict(box=[100,40,20,20],score=.9,**{'class':'FEMALE_BREAST_EXPOSED'}),[1,2,3],.9,0,'raw')
        result=gate.refine([(0,pixels)],[candidate])
        self.assertEqual(len(result[0][3]),0)
        pictures=gate.models.clip_embeddings.call_args.args[0]
        np.testing.assert_array_equal(pictures[0][1],crop(pixels,[.5,.4,.1,.2]))
        np.testing.assert_array_equal(pictures[1][1],body_crop(pixels,[.5,.4,.1,.2]))
        self.assertEqual(pictures[1][1].shape,(100,100,3))

    def test_promotion_rejects_new_low_precision_detector_and_zero_gold_false_alarms(self):
        from test_learned import LABEL_CATEGORIES
        def report():
            return dict(benchmark_identity='x',labelers={'human':200},categories={c:dict(positive_titles=10,clean_titles=10,
                 title_bootstrap_95={'precision':[.95,1],'recall':[.95,1]},all=dict(precision=.99,recall=.5,
                 counts=dict(gold=10,events=30,fp=0,clean_windows=100),clean_window_false_alarm=0)) for c in LABEL_CATEGORIES})
        base,candidate=report(),report()
        base['categories']['sexual_activity']['all']['precision']=None
        candidate['categories']['sexual_activity']['all'].update(precision=.042,recall=.97)
        self.assertFalse(promotion_check(base,candidate)[0])
        base,candidate=report(),report()
        candidate['categories']['violence']['all']['recall']=.97
        base['categories']['gore']['all']['counts']['gold']=0
        candidate['categories']['gore']['all']['counts'].update(gold=0,fp=24)
        self.assertFalse(promotion_check(base,candidate)[0])

    def test_ties_never_claim_prefix_precision(self):
        scores, y = np.ones(100), np.r_[np.ones(10), np.zeros(90)]
        self.assertIsNone(thresholds(scores, y)[1])
        self.assertEqual(thresholds(scores, y)[0], 1)

    def test_missing_labeled_window_is_an_error(self):
        project, labels, predictions = fixture()
        predictions.pop('1')
        with self.assertRaisesRegex(ValueError, 'Missing predictions'):
            evaluate(project, labels, predictions)

    def test_duplicate_windows_and_manual_benchmark_events_are_rejected(self):
        project, labels, predictions = fixture()
        project['windows'].append(project['windows'][0])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            evaluate(project, labels, predictions)
        project['windows'].pop()
        predictions['0']['events'] = [dict(start=1, end=2, category='violence', confirmed=True)]
        with self.assertRaisesRegex(ValueError, 'Manual'):
            evaluate(project, labels, predictions)

    def test_padding_stops_at_cut(self):
        event = dict(start=.7, end=1, category='nudity', confidence=.9, time_bounds=[0, 1], region=[.1,.1,.2,.2])
        plan = compile_plan([event], default_policy(), 3)
        self.assertEqual(plan[-1]['end'], 1)
        self.assertAlmostEqual(plan[0]['start'], .4)

    def test_single_model_load_with_two_first_callers(self):
        feature = FeatureModels.__new__(FeatureModels)
        feature.sessions, feature.session_lock = {}, threading.RLock()
        feature.folder, feature.files = Path('.'), {'ast':'ast.onnx'}
        def load(*args, **kwargs):
            time.sleep(.03)
            return object()
        ort = Mock(); ort.InferenceSession.side_effect = load
        feature.models = Mock(); feature.models.ort.return_value = (ort, None, ['CPUExecutionProvider'])
        with ThreadPoolExecutor(2) as pool:
            sessions = list(pool.map(lambda _:feature.session('ast'), range(2)))
        self.assertIs(sessions[0], sessions[1])
        self.assertEqual(ort.InferenceSession.call_count, 1)

    def test_cuda_anatomy_calls_do_not_overlap(self):
        from analysis_engine import Models
        with tempfile.TemporaryDirectory() as temporary:
            models=Models.__new__(Models)
            models.root=Path(temporary)
            models.lock,models.sessions,models.backends=threading.RLock(),{},{}
            path=models.root/'scene-verifier/640m.onnx'
            path.parent.mkdir();path.write_bytes(b'fixture')
            session=Mock()
            session.get_providers.return_value=['CUDAExecutionProvider']
            session.get_inputs.return_value=[Mock(name='input')]
            ort=Mock();ort.InferenceSession.return_value=session
            active,peak=0,0
            counter_lock=threading.Lock()
            def infer(model,image,include_class_scores=False):
                nonlocal active,peak
                self.assertFalse(include_class_scores)
                with counter_lock:active+=1;peak=max(peak,active)
                time.sleep(.02)
                with counter_lock:active-=1
                return image
            with patch.object(models,'ort',return_value=(ort,None,['CUDAExecutionProvider'])),patch('anatomy_runtime.detect',side_effect=infer):
                detector=models.nude('cuda','640m')
                with ThreadPoolExecutor(2) as pool:
                    self.assertEqual(list(pool.map(detector.detect,range(8))),list(range(8)))
            self.assertEqual(peak,1)
            self.assertEqual(ort.InferenceSession.call_count,1)

    def test_benchmark_checks_regions_and_allows_only_diagnostic_score_jitter(self):
        from copy import deepcopy
        from bench_runtime import concurrency_parity
        windows=[dict(id='w',clip_duration=3)]
        first=[dict(events=[dict(start=1,end=2,category='nudity',confidence=.9,region=[.1,.1,.2,.2])],
                    trace=[dict(score=.8,category='nudity',time=1)])]
        second=deepcopy(first);second[0]['trace'][0]['score']+=1e-9
        self.assertTrue(concurrency_parity(windows,first,second)['passed'])
        second[0]['events'][0]['region'][0]+=.001
        result=concurrency_parity(windows,first,second)
        self.assertFalse(result['passed'])
        self.assertFalse(result['windows'][0]['plan_identical'])

    def test_cache_coalesces_concurrent_loads_and_evicts(self):
        cache, calls = BoundedCache(64), []
        def load():
            calls.append(1); time.sleep(.03); return np.zeros(8)
        with ThreadPoolExecutor(2) as pool:
            list(pool.map(lambda _:cache.get('a', load), range(2)))
        self.assertEqual(len(calls), 1)
        cache.get('b', lambda:np.zeros(8))
        self.assertLessEqual(cache.bytes, 64)
        self.assertNotIn('a', cache.values)

    def test_head_certification_is_bound_to_weights(self):
        heads = dict(kinds=['clip'], grid=.25, enabled=['violence'], heads={'violence':{'coef':[1]}})
        heads['promoted'] = dict(qualified=True, verification='off', head_identity=head_identity(heads))
        self.assertTrue(qualified(heads))
        heads['heads']['violence']['coef'] = [2]
        self.assertFalse(qualified(heads))
        self.assertFalse(qualified(dict(heads, promoted=True)))

    def test_run_resume_rejects_changed_identity_without_rewriting(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'run.json'
            ensure_run(path, dict(identity='a'))
            content = path.read_bytes()
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                ensure_run(path, dict(identity='b'))
            self.assertEqual(content, path.read_bytes())

    def test_predictions_use_player_chunk_grid(self):
        models = Mock(); models.analyze.return_value=[]
        predict_window(models, Path('unused'), dict(id='x', target=[2,14], clip_duration=16, has_audio=False), ScanConfig())
        self.assertEqual([(c.args[1], c.args[2]) for c in models.analyze.call_args_list], [(0,6),(6,12),(12,16)])

    def test_changed_head_invalidates_saved_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root/'heads'/'h.json', dict(kinds=[], grid=.25, heads={}, enabled=[]))
            class Fake:
                def __init__(self, **kwargs): self.root=root; self.backends={}; self.warnings=[]
            with patch('analysis_engine.Models', Fake):
                engine = Engine(workers=1)
            try:
                job = MediaJob('x','x',root/'absent.mp4',12,False, config=ScanConfig(speech=False,nudity=False,scenes=False,learned_heads='h'))
                engine.refresh_identity(job); job.chunks={0:[]}
                write_json(root/'heads'/'h.json', dict(kinds=[], grid=.25, heads={}, enabled=['violence']))
                engine.refresh_identity(job)
                self.assertEqual(job.chunks,{})
                self.assertEqual(job.generation,1)
            finally: engine.close()

    def test_wall_speed_differs_from_worker_service_speed(self):
        job=MediaJob('x','x',Path('unused'),12,False)
        job.chunks={0:[],1:[]}; job.timings=[6,6]; job.wall_seconds=6
        models=Mock(); models.backends={}; models.warnings=[]
        result=job.snapshot(models)
        self.assertEqual(result['speed'],2)
        self.assertEqual(result['service_speed'],1)


if __name__ == '__main__':
    unittest.main()
