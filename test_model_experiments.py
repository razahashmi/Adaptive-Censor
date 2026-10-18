"""Numerical export checks and annotation-trust guards for model experiments."""
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest

import numpy as np

from labeling import read_json,write_json
from provenance import file_hash


TRAINING=bool(importlib.util.find_spec('sklearn'))


class ModelExperimentTests(unittest.TestCase):
    def test_semantic_calibration_preserves_annotation_trust_and_crop_geometry(self):
        from unittest.mock import Mock
        from calibrate_roi import calibrate,POSITIVE,NEGATIVE,COVERED_POSITIVE,COVERED_NEGATIVE
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);weight=root/'clip/onnx/vision_model_quantized.onnx'
            weight.parent.mkdir(parents=True);weight.write_bytes(b'fixture')
            rows=[];X=[]
            for title in range(5):
                for index in range(8):
                    label=index%2;vector=np.zeros(512,dtype=np.float32);vector[0 if label else 1]=1
                    X.append(vector);rows.append(dict(title=f'Title {title}',split='train',label=label,labeler='codex',detector_class='FEMALE_BREAST_EXPOSED'))
            np.savez(root/'features.npz',embeddings=X)
            data=dict(rows=rows,feature_identity=file_hash(root/'features.npz'),clip_identity=file_hash(weight),crop_method='Raw detector box with 25% margin')
            write_json(root/'labels.json',data);models=Mock(root=root)
            positive=np.zeros(512);positive[0]=1;negative=np.zeros(512);negative[1]=1
            models.text_embeddings.return_value=np.array([positive]*len(POSITIVE)+[negative]*len(NEGATIVE))
            with self.assertRaisesRegex(ValueError,'at least 40'):calibrate(root/'labels.json',root/'bad.json',models)
            head=calibrate(root/'labels.json',root/'head.json',models,allow_ai_labels=True)
            self.assertEqual(head['training_recall'],1)
            self.assertEqual(head['training_negative_rejection'],1)
            self.assertEqual(head['supported_classes'],['FEMALE_BREAST_EXPOSED'])
            self.assertEqual(head['labelers'],{'codex':40})
            self.assertEqual(head['negative_confirmations'],3)
            self.assertNotIn('oof_recall',head)
            models.text_embeddings.return_value=np.array([positive]*len(COVERED_POSITIVE)+[negative]*len(COVERED_NEGATIVE))
            covered=calibrate(root/'labels.json',root/'covered.json',models,True,'covered-anatomy')
            self.assertEqual(covered['training_recall'],1)
            self.assertEqual(covered['training_negative_rejection'],1)
            self.assertEqual(covered['prototypes']['negative'],list(COVERED_NEGATIVE))
            data['feature_views']=['body','local'];write_json(root/'labels.json',data)
            with self.assertRaisesRegex(ValueError,'reordered'):calibrate(root/'labels.json',root/'bad.json',models,True)
            data.pop('feature_views');data['crop_method']='Legacy segment mask';write_json(root/'labels.json',data)
            with self.assertRaisesRegex(ValueError,'raw detector crops'):calibrate(root/'labels.json',root/'bad.json',models,True)

    def test_invalid_anatomy_parameters_cannot_be_used_for_decisions(self):
        from roi_gate import validate_head
        head=dict(mean=[0]*512,scale=[1]*512,coef=[1]*512,intercept=0,decision_threshold=0)
        validate_head(head)
        for malformed in (dict(head,decision_threshold=float('nan')),dict(head,scale=[0]*512),dict(head,coef=[1]),dict(head,feature_views=['local','local']),dict(head,negative_confirmations=2),dict(head,view_thresholds={'local':float('nan')})):
            with self.assertRaises(ValueError):validate_head(malformed)

    def test_region_veto_requires_distinct_nearby_matching_negative_frames(self):
        from unittest.mock import Mock
        from roi_gate import AnatomyGate
        from frame_cache import BoundedCache
        gate=AnatomyGate.__new__(AnatomyGate);gate.models=Mock();gate.embedding_cache=BoundedCache(1024)
        gate.head=dict(mean=[0],scale=[1],coef=[1],intercept=0,decision_threshold=1,negative_confirmations=3)
        gate.models.clip_embeddings.side_effect=lambda pictures,*args,**kwargs:np.zeros((len(pictures),1))
        frames=[(i/12,np.full((20,20,3),i,dtype=np.uint8)) for i in range(4)]
        def candidate(i,shot=0,box=None):
            return (i,shot,dict(box=box or [2,2,8,8],score=.9,**{'class':'FEMALE_BREAST_EXPOSED'}),[1,2,3],.9,i/12,'raw')
        for candidates in ([candidate(0)]*3,[candidate(0),candidate(1)],
                           [candidate(0),candidate(1),candidate(2,shot=1)],
                           [candidate(0),candidate(1),candidate(2,box=[12,12,8,8])]):
            self.assertTrue(all(len(r[3])==3 for r in gate.refine(frames,candidates)))
        self.assertTrue(all(len(r[3])==0 for r in gate.refine(frames,[candidate(i) for i in range(3)])))

    def test_positive_body_view_preserves_anatomy_despite_negative_weighted_score(self):
        from unittest.mock import Mock
        from roi_gate import AnatomyGate,validate_head
        from frame_cache import BoundedCache
        gate=AnatomyGate.__new__(AnatomyGate);gate.models=Mock();gate.embedding_cache=BoundedCache(2048*4)
        local=np.zeros(512);local[0]=-2;body=np.zeros(512);body[0]=1
        coef=np.zeros(1024);coef[0]=.75;coef[512]=.25
        gate.head=dict(mean=[0]*1024,scale=[1]*1024,coef=coef.tolist(),intercept=0,decision_threshold=0,
                       feature_views=['local','body'],head_type='semantic-prototype',view_weights={'local':.75,'body':.25},view_thresholds={'local':0,'body':0})
        validate_head(gate.head)
        gate.models.clip_embeddings.return_value=np.array([local,body])
        pixels=np.zeros((100,200,3),dtype=np.uint8);pixels[40:60,100:120]=255
        candidate=(0,0,dict(box=[100,40,20,20],score=.9,**{'class':'FEMALE_BREAST_EXPOSED'}),[1,2,3],.9,0,'raw')
        self.assertEqual(gate.refine([(0,pixels)],[candidate]),[candidate])

    def test_boundary_context_and_owned_regions_share_one_semantic_decision(self):
        from unittest.mock import Mock,patch
        from analysis_engine import Models,ScanConfig
        from core import compile_plan,default_policy
        from roi_gate import AnatomyGate
        from frame_cache import BoundedCache
        gate=AnatomyGate.__new__(AnatomyGate);gate.models=Mock();gate.embedding_cache=BoundedCache(1024)
        gate.head=dict(mean=[0],scale=[1],coef=[1],intercept=0,decision_threshold=1,negative_confirmations=3)
        gate.models.clip_embeddings.side_effect=lambda pictures,*args,**kwargs:np.zeros((len(pictures),1))
        frames=[(t,np.full((20,20,3),i,dtype=np.uint8)) for i,t in enumerate((5.9,6.,6.1))]
        hit=dict(box=[2,2,8,8],score=.9,**{'class':'FEMALE_BREAST_EXPOSED'})
        nude=Mock();nude.detect.return_value=[hit];nude.onnx_session.get_providers.return_value=['CPUExecutionProvider']
        models=Models.__new__(Models);models.backends={};models.nude=Mock(return_value=nude);models.roi_gate=Mock(return_value=gate)
        with patch('analysis_engine.native_frames',return_value=frames),patch('analysis_engine.refine_anatomy',return_value=None):
            events=models._anatomy_events(Path('unused'),6,12,14,ScanConfig(speech=False,scenes=False,anatomy_roi='fixture'),lambda entry:None)
        self.assertTrue(events)
        self.assertTrue(all(e['review_required'] for e in events))
        self.assertEqual(compile_plan(events,default_policy(),14),[])
        self.assertEqual(models.roi_gate.call_count,1)

    def test_roi_reloads_and_refuses_changed_clip_weights(self):
        from analysis_engine import Models
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);weight=root/'clip/onnx/vision_model_quantized.onnx'
            weight.parent.mkdir(parents=True);weight.write_bytes(b'original fixture')
            write_json(root/'roi-models/r.json',dict(clip_identity=file_hash(weight),mean=[0]*512,scale=[1]*512,coef=[1]*512,intercept=0,decision_threshold=0))
            models=Models.__new__(Models);models.root=root;models.sessions={};models.lock=threading.RLock()
            first=models.roi_gate('r')
            self.assertIs(first,models.roi_gate('r'))
            weight.write_bytes(b'changed and larger fixture')
            with self.assertRaisesRegex(RuntimeError,'CLIP weights changed'):models.roi_gate('r')

    @unittest.skipUnless(TRAINING,'training needs .train-venv')
    def test_compact_tree_export_matches_library_predictions(self):
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        from nonlinear_heads import export_tree,project_inputs,tree_scores
        from learned_runtime import head_outputs
        rng=np.random.default_rng(7);X=rng.normal(size=(160,6));y=(X[:,0]*X[:,1]>0).astype(int)
        scaler=StandardScaler().fit(X);pca=PCA(n_components=6).fit(scaler.transform(X))
        projection=dict(mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),center=pca.mean_.tolist(),components=pca.components_.tolist())
        transformed=project_inputs(projection,X)
        np.testing.assert_allclose(transformed,pca.transform(scaler.transform(X)),atol=1e-10,rtol=0)
        model=HistGradientBoostingClassifier(max_iter=25,max_leaf_nodes=7,min_samples_leaf=5,random_state=7).fit(transformed,y)
        head=export_tree(model);head.update(candidate_threshold=-100,decision_threshold=0)
        head=json.loads(json.dumps(head))
        scores=model.decision_function(transformed)
        np.testing.assert_allclose(tree_scores(head,transformed),scores,atol=1e-10,rtol=0)
        times=np.arange(len(X),dtype=float)+.125
        entries,events=head_outputs(dict(grid=.25,projection=projection,heads={'violence':head}),times,X)
        np.testing.assert_allclose([e['score'] for e in entries],scores,atol=1e-10,rtol=0)
        self.assertEqual([e['sample_time'] for e in events],times[scores>=0].tolist())

    @unittest.skipUnless(TRAINING,'training needs .train-venv')
    def test_roi_ai_annotations_need_explicit_opt_in_and_stay_experimental(self):
        from roi_gate import train
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);rows=[];embeddings=[]
            for title in range(5):
                for index in range(8):
                    label=index%2
                    rows.append(dict(id=str(len(rows)),title=f'title {title}',split='train',label=label,labeler='codex'))
                    embeddings.append([label*2-1,0.])
            np.savez(root/'features.npz',embeddings=np.array(embeddings,dtype=np.float32))
            write_json(root/'labels.json',dict(rows=rows,feature_identity=file_hash(root/'features.npz'),clip_identity='fixture'))
            with self.assertRaisesRegex(ValueError,'Review at least 40'):train(root/'labels.json',root/'model.json')
            train(root/'labels.json',root/'model.json',allow_ai_labels=True)
            head=read_json(root/'model.json')
            self.assertTrue(head['experimental'])
            self.assertTrue(head['ai_labels_allowed'])
            self.assertEqual(head['labelers'],{'codex':40})
            self.assertEqual(head['operating_point'],'protect-recall')
            self.assertGreaterEqual(head['oof_recall'],.97)
            self.assertGreaterEqual(head['oof_negative_rejection'],.8)
            self.assertNotIn('promoted',head)


if __name__=='__main__':unittest.main()
