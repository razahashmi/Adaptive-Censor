"""Decision validity, review gating and chunk ownership; no model downloads."""
from pathlib import Path
import unittest
from unittest.mock import Mock, patch
import numpy as np

from core import compile_plan, default_policy, validate_events
from analysis_engine import (Models, ScanConfig, sample_burst, merge_scene_events, anatomy_support, refine_anatomy,
                             anatomy_proposals, native_frames, shot_ids, anatomy_segments)
from scene_verifier import CATEGORIES, validate_result, SceneVerifier, supported_evidence


def complete_result(tag=None):
    return dict(summary='Test', tags=[tag if tag and tag['category']==category else
        dict(category=category,state='absent',frames=[],reason='No visible evidence') for category in CATEGORIES])

def fake_burst(path,center):
    return [(center+d,np.zeros((12,12,3),dtype=np.uint8)) for d in [-.16,-.08,0,.08,.16]],2

def dense(times,size=100,color=0):
    return [(round(float(t),4),np.full((size,size,3),color,dtype=np.uint8)) for t in times]

def anatomy_model(results):
    model=Mock();model.onnx_session.get_providers.return_value=['CPUExecutionProvider']
    model.detect.side_effect=results
    return model


class VerifierTests(unittest.TestCase):
    def test_contradictory_or_unsupported_positive_labels_need_review(self):
        for label,reason in [('gun_discharge','Woman holds a gun and looks upward.'),
                             ('gun_discharge','No muzzle flash or smoke is visible.'),
                             ('person_striking_person','Man stands near a person lying still.'),
                             ('severe_bloody_injury','Woman looks distressed and clutches her neck.')]:
            self.assertFalse(supported_evidence(label,reason))
        self.assertTrue(supported_evidence('gun_discharge','Man fires a pistol, muzzle flash visible.'))
        self.assertTrue(supported_evidence('forceful_grab_or_drag','Man grips and pulls her hair.'))
        self.assertTrue(supported_evidence('person_striking_person',"A person's hand forcefully presses against another person's face."))

    def test_contact_recoil_paraphrases_need_actual_contact_with_a_person(self):
        self.assertTrue(supported_evidence('person_striking_person','Person swings bat, then recoils after contact with another person.'))
        self.assertTrue(supported_evidence('person_striking_person',"An arm extends, making contact with another person's head, followed by a recoil motion."))
        for text in ['A person swings a bat during practice.',
                     'A person makes contact with another person while dancing.',
                     'An arm extends without contact with another person, then recoils.',
                     'A person swings a bat, contact with a ball causes recoil.']:
            self.assertFalse(supported_evidence('person_striking_person',text))

    def test_crop_refinement_remaps_regions_and_keeps_temporal_threshold(self):
        proposal=dict(score=.3,box=[620,300,20,20],**{'class':'FEMALE_BREAST_EXPOSED'})
        burst=[(1+i*.08,np.zeros((360,640,3),dtype=np.uint8)) for i in range(5)]
        # The clamped 320px context starts at (320,40); crop boxes must map back.
        detection=dict(proposal,score=.8,box=[300,260,20,20])
        model=Mock();model.detect.return_value=[detection]
        anchor,votes=refine_anatomy(model,proposal,burst,2)
        self.assertEqual(anchor['box'],proposal['box'])
        self.assertEqual(len(votes),5)
        self.assertEqual(model.detect.call_args.args[0].shape,(320,320,3))
        for changes in [dict(score=.54),dict(box=[20,20,20,20]),{'class':'BUTTOCKS_EXPOSED'}]:
            model.detect.return_value=[dict(detection,**changes)]
            self.assertIsNone(refine_anatomy(model,proposal,burst,2))
        model.detect.return_value=[detection]
        self.assertIsNone(refine_anatomy(model,proposal,[(1,p) for _,p in burst],2))
        model.detect.side_effect=[[detection],[detection],[],[],[]]
        self.assertIsNone(refine_anatomy(model,proposal,burst,2))

    def test_neighbor_proposals_are_bounded_deduplicated_and_need_fresh_evidence(self):
        hit=dict(score=.3,box=[20,20,20,20],**{'class':'FEMALE_BREAST_EXPOSED'})
        samples=[(0,[hit]),(.5,[]),(1,[dict(hit,score=.4)]),(2,[dict(hit,box=[60,60,20,20])])]
        proposals=anatomy_proposals(samples,1)
        self.assertEqual(len(proposals),1)
        self.assertEqual(proposals[0][0],1)
        model=Mock();model.detect.return_value=[]
        burst=[(.5+i*.08,np.zeros((100,100,3),dtype=np.uint8)) for i in range(5)]
        self.assertIsNone(refine_anatomy(model,proposals[0][1],burst,2))
    def test_repeated_frames_reuse_cache_but_changed_concept_does_not(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            verifier=SceneVerifier(Path(folder))
            client=Mock();verifier.client=client
            import json
            client.post.return_value.json.return_value={'choices':[{'finish_reason':'stop','message':{'content':json.dumps(complete_result())}}]}
            frames=[(1.,np.zeros((12,12,3),dtype=np.uint8))]
            with patch.object(verifier,'start'):
                first=verifier.verify(frames)
                second=verifier.verify(frames)
                verifier.verify(frames,'spider')
            self.assertFalse(first['cache_hit'])
            self.assertTrue(second['cache_hit'])
            self.assertEqual(client.post.call_count,2)
            verifier.close()

    def test_invalid_results_never_become_decisions(self):
        good = dict(category='violence', state='present', frames=[1], reason='Visible strike')
        for update in [dict(frames=[20]), dict(frames=[True]), dict(frames=[1,1]),
                       dict(state='absent'), dict(frames=[]), dict(category='unknown')]:
            result=complete_result(good)
            result['tags'][CATEGORIES.index('violence')]=dict(good,**update)
            with self.assertRaises(ValueError):
                validate_result(result,3)
        self.assertTrue(all(t['frames']==[] for t in validate_result(complete_result(),3)['tags']))
        for tags in ([],[good],complete_result()['tags'][:-1]):
            with self.assertRaisesRegex(ValueError,'omitted category'):
                validate_result(dict(summary='Incomplete',tags=tags),3)

    def test_review_gate_covers_legacy_clip_and_uncertain_events(self):
        policy=default_policy()
        for details in [dict(source='CLIP experimental'), dict(review_required=True),
                        dict(source='CLIP experimental',review_required=False)]:
            event=dict(start=1,end=2,category='sexual_activity',**details)
            self.assertEqual(compile_plan([event],policy,3),[])
            self.assertTrue(compile_plan([dict(event,confirmed=True)],policy,3))
        with self.assertRaises(ValueError):
            validate_events([dict(event,confirmed='true')],3)

    def test_only_owned_verified_samples_are_committed_and_uncertainty_is_retained(self):
        frames=[(float(t),np.zeros((12,12,3),dtype=np.uint8)) for t in range(4,14)]
        with patch('analysis_engine.prepare_environment',return_value=Path('unused')):
            models=Models()
        verifier=Mock()
        verifier.verify.return_value=dict(tags=[dict(category='violence',state='present',frames=[0,3,9],reason='strike'),
            dict(category='gore',state='uncertain',frames=[4],reason='Obscured injury'),
            dict(category='custom',state='present',frames=[5],reason='Ignored empty custom')])
        verifier.check_frame.side_effect=lambda f,c,*a,**kw:dict(state='uncertain' if c=='gore' else 'present' if f[2][0]==7 else 'absent',label='unclear' if c=='gore' else 'person_striking_person',observation='Frame evidence')
        with patch('analysis_engine.sample_frames',return_value=frames),patch('analysis_engine.sample_burst',side_effect=fake_burst), patch.object(models,'verifier',return_value=verifier), patch.object(models,'scene_scores',side_effect=AssertionError('CLIP must not run')):
            result=models.analyze(Path('unused'),6,12,20,ScanConfig(speech=False,nudity=False,scene_filtering='automatic'),False)
        self.assertEqual([(e['category'],e['sample_time'],e['review_required']) for e in result],[('violence',7,False),('gore',8,True)])
        self.assertTrue(all(6<=e['sample_time']<12 for e in result))

    def test_scene_decisions_require_review_by_default(self):
        self.assertEqual(ScanConfig().scene_filtering,'review')
        frames=[(1.,np.zeros((12,12,3),dtype=np.uint8))]
        with patch('analysis_engine.prepare_environment',return_value=Path('unused')):
            models=Models()
        verifier=Mock();verifier.verify.return_value=dict(tags=[dict(category='violence',state='present',frames=[0],reason='Visible action')])
        verifier.check_frame.return_value=dict(state='present',label='person_striking_person',observation='Visible strike')
        with patch('analysis_engine.sample_frames',return_value=frames),patch('analysis_engine.sample_burst',side_effect=fake_burst),patch.object(models,'verifier',return_value=verifier):
            result=models.analyze(Path('unused'),0,6,10,ScanConfig(speech=False,nudity=False),False)
        self.assertTrue(result[0]['review_required'])
        self.assertEqual(compile_plan(result,default_policy(),10),[])

    def test_anatomy_needs_distinct_consistent_regions_and_unlocalized_claim_needs_review(self):
        frames=[(1.,np.zeros((100,100,3),dtype=np.uint8))]
        native=dense([.84,.92,1,1.08,1.16])
        with patch('analysis_engine.prepare_environment',return_value=Path('unused')):
            models=Models()
        verifier=Mock()
        hit={'class':'FEMALE_BREAST_EXPOSED','score':.95,'box':[20,20,20,20]}
        for scene_present,box_present,supported in ((False,True,False),(True,False,False),(False,True,True)):
            verifier.verify.return_value=complete_result(dict(category='nudity',state='present' if scene_present else 'absent',
                frames=[0] if scene_present else [],reason='Candidate'))
            verifier.check_frame.return_value=dict(state='present',label='exposed_female_nipple',observation='Visible nipple')
            anatomy=anatomy_model([[hit] if supported or (box_present and i==2) else [] for i in range(5)])
            with patch('analysis_engine.sample_frames',return_value=frames),patch('analysis_engine.native_frames',return_value=native),\
                    patch('analysis_engine.sample_burst',side_effect=fake_burst),\
                    patch.object(models,'verifier',return_value=verifier),patch.object(models,'nude',return_value=anatomy):
                result=models.analyze(Path('unused'),0,6,10,ScanConfig(speech=False,scene_filtering='automatic'),False)
            plan=compile_plan(result,default_policy(),10)
            if supported:
                self.assertTrue(plan)
                self.assertTrue(all(e.get('region') for e in result))
            else:
                self.assertTrue(result)
                self.assertEqual(plan,[])

    def test_anatomy_votes_reject_duplicates_different_objects_classes_and_weak_scores(self):
        hit=dict(score=.8,box=[20,20,20,20],**{'class':'FEMALE_BREAST_EXPOSED'})
        samples=[(1+i*.08,[dict(hit)]) for i in range(5)]
        self.assertEqual(len(anatomy_support(hit,samples)),5)
        self.assertEqual(len(anatomy_support(hit,[(1,[hit])]*5)),1)
        for change in [dict(score=.4),dict(box=[70,70,20,20]),{'class':'BUTTOCKS_EXPOSED'}]:
            mixed=[samples[0],samples[1]]+[(1+i*.08,[dict(hit,**change)]) for i in range(2,5)]
            self.assertEqual(len(anatomy_support(hit,mixed)),2)

    def test_unlocalized_anatomy_suggestions_never_call_burst_vlm_or_filter_automatically(self):
        frames=[(float(t),np.zeros((100,100,3),dtype=np.uint8)) for t in range(4,14)]
        with patch('analysis_engine.prepare_environment',return_value=Path('unused')):
            models=Models()
        verifier=Mock();verifier.verify.return_value=complete_result(dict(category='nudity',state='present',
            frames=[0,3,4,9],reason='Visible anatomy proposal'))
        anatomy=Mock();anatomy.detect.return_value=[]
        anatomy.onnx_session.get_providers.return_value=['CPUExecutionProvider']
        with patch('analysis_engine.sample_frames',return_value=frames),patch('analysis_engine.native_frames',return_value=frames),\
                patch.object(models,'verifier',return_value=verifier),\
                patch.object(models,'nude',return_value=anatomy),patch('analysis_engine.sample_burst',side_effect=AssertionError('No anatomical proposal needs native frames')):
            result=models.analyze(Path('unused'),6,12,20,ScanConfig(speech=False,scene_filtering='automatic'),False)
        verifier.check_frame.assert_not_called()
        self.assertEqual({e['sample_time'] for e in result},{7.,8.})
        self.assertTrue(all(e['review_required'] and e['source']=='Qwen sequence proposal' for e in result))
        self.assertEqual(compile_plan(result,default_policy(),20),[])
        self.assertTrue(compile_plan([dict(e,confirmed=True) for e in result],default_policy(),20))

    def test_native_bursts_are_distinct_and_keep_the_target(self):
        import tempfile
        from core import make_demo
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'sample.mp4';make_demo(path)
            for center in (0,2,11.95):
                frames,index=sample_burst(path,center)
                times=[t for t,_ in frames]
                self.assertEqual(len(times),5)
                self.assertEqual(len(set(times)),5)
                self.assertLess(abs(times[index]-center),.05)
            frames,index=sample_burst(path,2)
            self.assertTrue(all(.07<b[0]-a[0]<.09 for a,b in zip(frames,frames[1:])))

    def test_merge_does_not_bridge_clean_gap_or_uncertain_evidence(self):
        base=dict(category='violence',confidence=1,source='Qwen verified frame',evidence='gun_discharge',review_required=False,note='Flash')
        events=[dict(base,start=1,end=1.5),dict(base,start=1.5,end=2),dict(base,start=2.5,end=3),dict(base,start=3,end=3.5,review_required=True)]
        result=merge_scene_events(events)
        self.assertEqual([(e['start'],e['end']) for e in result],[(1,2),(2.5,3),(3,3.5)])

    def test_trace_keeps_rejected_candidates_for_candidate_recall(self):
        frames=[(float(t),np.zeros((100,100,3),dtype=np.uint8)) for t in range(4,14)]
        with patch('analysis_engine.prepare_environment',return_value=Path('unused')):
            models=Models()
        verifier=Mock();verifier.verify.return_value=complete_result(dict(category='violence',state='present',frames=[3],reason='Possible strike'))
        verifier.check_frame.return_value=dict(state='absent',label='none',observation='No contact')
        weak={'class':'FEMALE_BREAST_EXPOSED','score':.3,'box':[20,20,20,20]}
        anatomy=Mock();anatomy.detect.side_effect=lambda pixels:[weak] if pixels.shape==(100,100,3) else []
        anatomy.onnx_session.get_providers.return_value=['CPUExecutionProvider']
        trace=[]
        with patch('analysis_engine.sample_frames',return_value=frames),patch('analysis_engine.native_frames',return_value=frames),\
                patch('analysis_engine.sample_burst',side_effect=fake_burst),\
                patch.object(models,'verifier',return_value=verifier),patch.object(models,'nude',return_value=anatomy):
            result=models.analyze(Path('unused'),6,12,20,ScanConfig(speech=False,scene_filtering='automatic'),False,trace=trace)
        self.assertEqual(compile_plan(result,default_policy(),20),[])
        stages={(e['stage'],e['category']) for e in trace}
        self.assertIn(('scene_sequence','violence'),stages)
        self.assertIn(('scene_evidence','violence'),stages)
        self.assertIn(('anatomy_raw','nudity'),stages)
        self.assertTrue(all(6<=e['sample_time']<12 for e in trace))
        self.assertEqual([e['state'] for e in trace if e['stage']=='scene_evidence'],['absent']*3)

    def test_native_frames_use_global_frame_numbers(self):
        import tempfile
        from core import make_demo
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'sample.mp4';make_demo(path)
            whole=[t for t,_ in native_frames(path,0,3,12)]
            part=[t for t,_ in native_frames(path,1.03,2.5,12)]
            self.assertTrue(all(abs(b-a-1/12)<.002 for a,b in zip(whole,whole[1:])))
            self.assertTrue(set(part)<=set(whole) and len(part)>=16)

    def test_shot_ids_split_hard_cuts_only(self):
        frames=dense(range(6),color=40)+dense(range(6,9),color=200)
        frames[1]=(frames[1][0],frames[1][1]+np.uint8(6))
        self.assertEqual(shot_ids(frames),[0]*6+[1]*3)

    def _dense_anatomy(self,times,detections,shots=None):
        """Run only the anatomy pass over synthetic dense samples."""
        with patch('analysis_engine.prepare_environment',return_value=Path('unused')):
            models=Models()
        native=dense(times)
        if shots:
            native=[(t,np.full((100,100,3),40 if s==0 else 220,dtype=np.uint8)) for (t,_),s in zip(native,shots)]
        anatomy=anatomy_model(detections)
        with patch('analysis_engine.native_frames',return_value=native),patch.object(models,'nude',return_value=anatomy):
            return models.analyze(Path('unused'),0,6,10,ScanConfig(speech=False,scenes=False),False)

    def test_dense_samples_accept_a_brief_flash_inside_one_shot(self):
        hit={'class':'FEMALE_BREAST_EXPOSED','score':.9,'box':[20,20,20,20]}
        times=[1+i/12 for i in range(10)]
        result=self._dense_anatomy(times,[[hit] if i in (4,5,6) else [] for i in range(10)])
        accepted=[e for e in result if not e['review_required']]
        self.assertEqual(len(accepted),1)
        self.assertAlmostEqual(accepted[0]['start'],times[4]-1/24,places=3)
        self.assertAlmostEqual(accepted[0]['end'],times[6]+1/24,places=3)
        self.assertTrue(compile_plan(result,default_policy(),10))

    def test_dense_support_never_crosses_a_cut(self):
        hit={'class':'FEMALE_BREAST_EXPOSED','score':.9,'box':[20,20,20,20]}
        times=[1+i/12 for i in range(10)]
        result=self._dense_anatomy(times,[[hit] if i in (3,4,5) else [] for i in range(10)],shots=[0]*5+[1]*5)
        self.assertTrue(result)
        self.assertTrue(all(e['review_required'] for e in result))
        self.assertEqual(compile_plan(result,default_policy(),10),[])

    def test_segments_chain_moving_regions_and_split_by_shot_length_and_acceptance(self):
        hit=lambda x:{'class':'FEMALE_BREAST_EXPOSED','score':.9,'box':[x,20,20,20]}
        frames=dense([i/12 for i in range(14)])
        found=[(i,0,hit(20+i),[1,2,3],.9,i/12,'full picture') for i in range(8)]
        found.append((9,1,hit(28),[1,2,3],.9,.75,'full picture'))
        found.append((10,1,hit(28),[1],.9,.83,'full picture'))
        events=anatomy_segments(frames,found,1/12)
        self.assertEqual([(e['samples'],e['review_required']) for e in events],[(6,False),(2,False),(1,False),(1,True)])
        first=events[0]['region']
        self.assertLessEqual(first[0],(20-4)/100)
        self.assertGreaterEqual(first[0]+first[2],(25+20+4)/100-1e-9)

    @staticmethod
    def _moving_texture(count, step=(4, 2), size=(360, 640)):
        """Frames of one textured scene panning by `step` pixels per sample."""
        import cv2
        rng=np.random.default_rng(5)
        world=cv2.GaussianBlur((rng.random((size[0]+200,size[1]+200))*255).astype(np.uint8),(0,0),2)
        frames=[]
        for i in range(count):
            x,y=100-step[0]*i,100-step[1]*i
            gray=world[y:y+size[0],x:x+size[1]]
            frames.append((round(1+i/12,4),np.repeat(gray[:,:,None],3,2)))
        return frames

    def test_median_flow_follows_texture_and_refuses_flat_pictures(self):
        import cv2
        from analysis_engine import median_flow
        frames=self._moving_texture(2)
        gray=[cv2.cvtColor(p,cv2.COLOR_RGB2GRAY) for _,p in frames]
        moved=median_flow(gray[0],gray[1],[200,100,60,60])
        self.assertAlmostEqual(moved[0],204,delta=1)
        self.assertAlmostEqual(moved[1],102,delta=1)
        self.assertAlmostEqual(moved[2],60,delta=3)
        flat=np.zeros((360,640),dtype=np.uint8)
        self.assertIsNone(median_flow(flat,flat,[200,100,60,60]))
        self.assertIsNone(median_flow(gray[0],gray[1],[630,100,60,60]))

    def test_tracking_bridges_gaps_between_detections_and_limits_edges(self):
        from analysis_engine import track_anatomy
        frames=self._moving_texture(24)
        shots=[0]*20+[1]*4
        anchor=lambda i:(i,shots[i],{'class':'FEMALE_BREAST_EXPOSED','score':.9,'box':[200+4*i,100+2*i,60,60]},[1,2,3],.9,frames[i][0],'full picture')
        tracked=track_anatomy(frames,shots,[anchor(0),anchor(1),anchor(8)],lambda j:True,1/12)
        indices=sorted(item[0] for item in tracked)
        self.assertEqual(indices[:6],[2,3,4,5,6,7])  # the whole 0.5 s gap between detections
        self.assertTrue(all(item[6]=='tracked' and len(item[3])==3 for item in tracked))
        box=next(item[2]['box'] for item in tracked if item[0]==5)
        self.assertAlmostEqual(box[0],220,delta=2)
        after=[i for i in indices if i>8]
        self.assertEqual(after,list(range(9,15)))  # edge extension stops after 0.5 s
        late=track_anatomy(frames,shots,[anchor(17)],lambda j:True,1/12)
        self.assertEqual(sorted(item[0] for item in late),[11,12,13,14,15,16,18,19])  # never into shot 1
        owned=track_anatomy(frames,shots,[anchor(0),anchor(8)],lambda j:j>=4,1/12)
        self.assertTrue(owned and all(item[0]>=4 for item in owned))

    def test_tracking_turns_detections_with_a_gap_into_one_continuous_mask(self):
        frames=self._moving_texture(16)
        hit=lambda i:[{'class':'FEMALE_BREAST_EXPOSED','score':.9,'box':[200+4*i,100+2*i,60,60]}]
        detections=[hit(i) if i in (0,1,2,10,11,12) else [] for i in range(16)]
        with patch('analysis_engine.prepare_environment',return_value=Path('unused')):
            models=Models()
        plans={}
        for tracking in (False,True):
            anatomy=anatomy_model(list(detections))
            with patch('analysis_engine.native_frames',return_value=frames),patch.object(models,'nude',return_value=anatomy):
                result=models.analyze(Path('unused'),0,6,10,ScanConfig(speech=False,scenes=False,anatomy_tracking=tracking),False)
            policy=default_policy();policy.update(pre_padding=0,post_padding=0)
            plans[tracking]=compile_plan(result,policy,10)
        gap=(frames[3][0],frames[9][0])
        covered=lambda plan,t:any(p['start']<=t<p['end'] for p in plan)
        self.assertFalse(covered(plans[False],(gap[0]+gap[1])/2))
        # Region boxes change as the mask moves, so a continuous mask can span several plan items.
        self.assertTrue(all(a['end']==b['start'] for a,b in zip(plans[True],plans[True][1:])))
        self.assertAlmostEqual(plans[True][0]['start'],frames[0][0]-1/24,places=3)
        self.assertAlmostEqual(plans[True][-1]['end'],frames[15][0]+1/24,places=3)  # 0.5 s tracked edge

    def test_tracking_stops_at_a_matching_semantically_rejected_region(self):
        from analysis_engine import track_anatomy
        frames=self._moving_texture(16);shots=[0]*16
        anchor=lambda i:(i,0,{'class':'FEMALE_BREAST_EXPOSED','score':.9,'box':[200+4*i,100+2*i,60,60]},[1,2,3],.9,frames[i][0],'raw')
        rejected=anchor(4)
        tracked=track_anatomy(frames,shots,[anchor(0),anchor(8)],lambda j:True,1/12,rejected=[rejected])
        indices={item[0] for item in tracked}
        self.assertNotIn(4,indices)
        self.assertTrue({1,2,3,5,6,7}.issubset(indices))
        elsewhere=tuple([4,0,dict(rejected[2],box=[10,10,60,60]),*rejected[3:]])
        other=track_anatomy(frames,shots,[anchor(0),anchor(8)],lambda j:True,1/12,rejected=[elsewhere])
        self.assertIn(4,{item[0] for item in other})

    def test_roi_rejections_survive_tracking_merging_and_policy_padding(self):
        frames=self._moving_texture(16)
        hit=lambda i:[{'class':'FEMALE_BREAST_EXPOSED','score':.9,'box':[200+4*i,100+2*i,60,60]}]
        detections=[hit(i) for i in range(16)]
        class Gate:
            head={}
            def refine(self,frames,items):
                result=[]
                for item in items:
                    if 5<=item[0]<=7:
                        i,shot,hit,_,score,t,note=item
                        item=(i,shot,dict(hit,roi_rejected=True),[],score,t,note+' + ROI review')
                    result.append(item)
                return result
        with patch('analysis_engine.prepare_environment',return_value=Path('unused')):
            models=Models()
        anatomy=anatomy_model(detections)
        with patch('analysis_engine.native_frames',return_value=frames),patch.object(models,'nude',return_value=anatomy),patch.object(models,'roi_gate',return_value=Gate()):
            events=models.analyze(Path('unused'),0,6,10,ScanConfig(speech=False,scenes=False,anatomy_roi='fixture'),False)
        plan=compile_plan(events,default_policy(),10)
        covered=lambda t:any(p['start']<=t<p['end'] for p in plan)
        self.assertTrue(covered(frames[4][0]) and covered(frames[8][0]))
        self.assertTrue(all(not covered(frames[i][0]) for i in (5,6,7)))

    def test_verifier_failure_propagates_without_coverage(self):
        with patch('analysis_engine.prepare_environment',return_value=Path('unused')):
            models=Models()
        verifier=Mock();verifier.verify.side_effect=RuntimeError('malformed output')
        with patch('analysis_engine.sample_frames',return_value=[(1,np.zeros((12,12,3),dtype=np.uint8))]), patch.object(models,'verifier',return_value=verifier):
            with self.assertRaisesRegex(RuntimeError,'malformed'):
                models.analyze(Path('unused'),0,6,12,ScanConfig(speech=False,nudity=False),False)


if __name__=='__main__':
    unittest.main()
