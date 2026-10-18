"""Contracts for the optional score-vector veto and unchanged detector output."""
import copy
import unittest
from unittest.mock import Mock
import numpy as np
from anatomy_runtime import LABELS,postprocess
from roi_gate import AnatomyGate,validate_head
from frame_cache import BoundedCache


def head():
    coef=np.zeros(len(LABELS));coef[16]=-1
    return dict(feature_kind='detector-scores',class_labels=list(LABELS),anatomy_model='640m',
                mean=[0.]*18,scale=[1.]*18,coef=coef.tolist(),intercept=0.,decision_threshold=-.1,
                negative_confirmations=3,supported_classes=['FEMALE_BREAST_EXPOSED'])


class DetectorScoreTests(unittest.TestCase):
    def candidates(self,n=5):
        frames=[(i/12,np.full((30,40,3),i,dtype=np.uint8)) for i in range(n)]
        scores=np.zeros(18);scores[3]=.8;scores[16]=.2
        hit={'class':'FEMALE_BREAST_EXPOSED','score':.8,'box':[5,5,8,8],'class_scores':scores.tolist()}
        return frames,[(i,0,copy.deepcopy(hit),[0,.1,.2],.8,i/12,'full picture') for i in range(n)]

    def gate(self):
        gate=AnatomyGate.__new__(AnatomyGate);gate.models=Mock();gate.head=head()
        return gate

    def test_class_scores_preserve_default_postprocessing(self):
        rows=np.zeros((22,3),dtype=np.float32)
        rows[:4]=np.array([[30,30,60],[30,30,60],[10,10,10],[10,10,10]])
        rows[7,0]=.8;rows[9,0]=.1;rows[7,1]=.7;rows[9,2]=.9
        args=(rows[None],0,0,1,1,100,100,100,100)
        baseline=postprocess(*args);enriched=postprocess(*args,include_class_scores=True)
        self.assertEqual([{k:v for k,v in h.items() if k!='class_scores'} for h in enriched],baseline)
        self.assertEqual(len(enriched),2)
        for h in enriched:
            self.assertEqual(len(h['class_scores']),18)
            self.assertEqual(LABELS[int(np.argmax(h['class_scores']))],h['class'])
            self.assertEqual(max(h['class_scores']),h['score'])

    def test_veto_needs_distinct_frames_without_any_new_model_calls(self):
        gate=self.gate();frames,items=self.candidates(2)
        self.assertEqual(gate.refine(frames,items),items)
        duplicated=items*3
        self.assertEqual(gate.refine(frames,duplicated),duplicated)
        frames,items=self.candidates(3);result=gate.refine(frames,items)
        self.assertTrue(all(r[2]['roi_rejected'] and not r[3] for r in result))
        self.assertTrue(all(len(r[3])==3 and 'roi_rejected' not in r[2] for r in items))
        self.assertEqual(gate.models.mock_calls,[])

    def test_contextual_or_positive_duplicate_protects_its_frame(self):
        for origin in ('contextual crop','full picture'):
            gate=self.gate();frames,items=self.candidates(3)
            alternative=list(copy.deepcopy(items[1]));alternative[6]=origin
            alternative[2]['class_scores'][16]=.01
            items.append(tuple(alternative))
            self.assertEqual(gate.refine(frames,items),items)

    def test_missing_invalid_or_mismatched_scores_keep_supported_detection(self):
        for value in (None,[0]*17,[float('nan')]*18,[-1]*18,[2]*18,[0]*18):
            gate=self.gate();frames,items=self.candidates(3)
            if value is None:items[1][2].pop('class_scores')
            else:items[1][2]['class_scores']=value
            result=gate.refine(frames,items)
            self.assertFalse(any(r[2].get('roi_rejected') for r in result))

    def test_votes_do_not_cross_shots_or_distant_frames(self):
        gate=self.gate();frames,items=self.candidates(3)
        items[2]=(2,1,*items[2][2:]);self.assertEqual(gate.refine(frames,items),items)
        frames,items=self.candidates(3);frames[2]=(1.,frames[2][1])
        self.assertEqual(gate.refine(frames,items),items)

    def test_head_rejects_changed_labels_nonfinite_and_wrong_model(self):
        validate_head(head())
        for key,value in [('class_labels',list(reversed(LABELS))),('coef',[0]*17),('scale',[0]*18),
                          ('intercept',float('nan')),('decision_threshold',None),('anatomy_model','320n'),
                          ('feature_kind','unknown'),('negative_confirmations',1)]:
            h=head();h[key]=value
            with self.assertRaises(ValueError):validate_head(h)

    def test_combined_check_needs_negative_body_context_and_reuses_embeddings(self):
        for body,expected in ((-2.,True),(2.,False)):
            gate=self.gate();gate.embedding_cache=BoundedCache(32*1024)
            coef=np.zeros(1024);coef[0]=coef[512]=1.
            gate.head=dict(mean=[0.]*1024,scale=[1.]*1024,coef=coef.tolist(),intercept=0.,decision_threshold=0.,
                           feature_views=['local','body'],view_weights={'local':1.,'body':1.},
                           view_thresholds={'local':0.,'body':0.},negative_confirmations=3,
                           supported_classes=['FEMALE_BREAST_EXPOSED'],detector_head=head())
            frames,items=self.candidates(3)
            embeddings=np.zeros((6,512));embeddings[::2,0]=2.;embeddings[1::2,0]=body
            gate.models.clip_embeddings.return_value=embeddings
            result=gate.refine(frames,items)
            self.assertEqual([bool(r[2].get('roi_rejected')) for r in result],[expected]*3)
            gate.refine(frames,items)
            self.assertEqual(gate.models.clip_embeddings.call_count,1)

    def test_combined_head_requires_compatible_weights_classes_and_body_view(self):
        h=dict(mean=[0.]*1024,scale=[1.]*1024,coef=[0.]*1024,intercept=0.,decision_threshold=0.,
               feature_views=['local','body'],view_weights={'local':1.,'body':1.},
               view_thresholds={'local':0.,'body':0.},negative_confirmations=3,head_type='semantic-prototype',
               supported_classes=['FEMALE_BREAST_EXPOSED'],anatomy_identity='fixture',detector_head=head())
        h['detector_head']['anatomy_identity']='fixture';validate_head(h)
        for key,value in [('anatomy_identity','changed'),('supported_classes',['BUTTOCKS_EXPOSED']),('feature_kind','clip')]:
            invalid=copy.deepcopy(h);invalid['detector_head'][key]=value
            with self.assertRaises(ValueError):validate_head(invalid)

    def test_semantic_checks_cannot_pool_weak_votes_across_contrasts(self):
        gate=self.gate();gate.embedding_cache=BoundedCache(32*1024)
        coef=np.zeros(1024);coef[0]=.75;coef[512]=.25
        gate.head=dict(mean=[0.]*1024,scale=[1.]*1024,coef=coef.tolist(),intercept=0.,decision_threshold=0.,
                       feature_views=['local','body'],view_weights={'local':.75,'body':.25},
                       view_thresholds={'local':0.,'body':0.},negative_confirmations=3,
                       supported_classes=['FEMALE_BREAST_EXPOSED'])
        secondary=copy.deepcopy(gate.head);secondary['coef']=np.roll(coef,1).tolist();gate.head['secondary_head']=secondary
        embeddings=np.zeros((6,512))
        for i,(first,second) in enumerate(((-1,1),(-1,-1),(1,-1))):
            embeddings[2*i:2*i+2,0]=first;embeddings[2*i:2*i+2,1]=second
        gate.models.clip_embeddings.return_value=embeddings
        frames,items=self.candidates(3)
        self.assertEqual(gate.refine(frames,items),items)
        self.assertEqual(gate.models.clip_embeddings.call_count,1)
        embeddings[:2,1]=-1;gate.embedding_cache=BoundedCache(32*1024)
        result=gate.refine(frames,items)
        self.assertTrue(all(r[2].get('roi_rejected') for r in result))

    def test_composition_requires_the_same_encoder_views_and_anatomy_weights(self):
        from compose_roi import compose
        h=dict(mean=[0.]*1024,scale=[1.]*1024,coef=[0.]*1024,intercept=0.,decision_threshold=0.,
               feature_views=['local','body'],view_weights={'local':.75,'body':.25},
               view_thresholds={'local':0.,'body':0.},negative_confirmations=3,head_type='semantic-prototype',
               supported_classes=['FEMALE_BREAST_EXPOSED'],clip_identity='clip',anatomy_identity='nude',anatomy_model='640m')
        composed=compose(h,h);self.assertNotIn('secondary_head',h)
        self.assertEqual(composed['secondary_head'],h)
        for key,value in [('clip_identity','changed'),('anatomy_identity','changed'),('negative_confirmations',5)]:
            changed=copy.deepcopy(h);changed[key]=value
            with self.assertRaises(ValueError):compose(h,changed)

    def test_legacy_null_threshold_demotes_supported_regions_without_an_error(self):
        gate=self.gate();gate.embedding_cache=BoundedCache(16*1024)
        gate.head=dict(mean=[0.]*512,scale=[1.]*512,coef=[0.]*512,intercept=0.,
                       decision_threshold=None,negative_confirmations=3)
        frames,items=self.candidates(3)
        gate.models.clip_embeddings.return_value=np.zeros((3,512))
        result=gate.refine(frames,items)
        self.assertTrue(all(r[2].get('roi_rejected') for r in result))


if __name__=='__main__':unittest.main()
