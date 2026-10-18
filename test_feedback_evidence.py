from copy import deepcopy
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from core import default_policy, make_demo, preview_frame
from feedback_evidence import Evidence, REVIEW_PROJECT, TRAIN_PROJECT, source_stamp
from labeling import DEFINITIONS_REVISION, LABEL_CATEGORIES, read_json, router as label_router, write_json
from playback_feedback import Report, Store, router
from provenance import file_hash


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name);self.source=self.root/'source.mp4';self.source.write_bytes(b'fake source for storage tests')
        self.calls=[]
        def encoder(source,start,end,destination):
            self.calls.append((source,start,end));destination.write_bytes(b'fake encoded evidence')
            return dict(duration=end-start,fps=24.,has_audio=True)
        self.keeper=Evidence(self.root,self.resolve,encoder=encoder)
        self.addCleanup(self.keeper.close)
        self.report=dict(id=str(uuid4()),status='confirmed',category='violence',kind='missed_content',window=[2.,6.],
            session_id='a'*32,context=dict(title='Example Show S01E02',source_ref=source_stamp(self.source),live=False))

    def resolve(self, report):
        if not self.source.is_file():raise HTTPException(404,'Source removed')
        return self.source

    def ready(self):
        self.keeper.retain(self.report)
        self.keeper.jobs[self.report['id']].result(timeout=5)
        self.assertEqual(self.keeper.status(self.report['id'])['status'],'ready')

    def task(self):
        self.ready();return self.keeper.make_task(self.report,'Example Show',.2,3.5)

    def labels(self):
        task=self.task()['task'];path=self.root/'labeling'/REVIEW_PROJECT/'labels'/f"{task['window']}.json"
        labels=dict(window=task['window'],labeler='human',definitions_revision=DEFINITIONS_REVISION,seconds_spent=4,updated='2026-10-15T01:00:00',
            categories={c:dict(status='reviewed',intervals=[[.5,1.]] if c=='violence' else [],ambiguous=[]) for c in LABEL_CATEGORIES})
        write_json(path,labels);return task,path

    def test_retention_is_idempotent_and_survives_removed_source(self):
        self.ready();self.keeper.retain(self.report);self.assertEqual(len(self.calls),1)
        self.source.unlink();clip,info=self.keeper.verified_clip(self.report['id'])
        self.assertEqual(file_hash(clip),info['sha256']);self.assertEqual(info['offset'],2)
        again=Evidence(self.root,self.resolve);self.addCleanup(again.close)
        self.assertEqual(again.status(self.report['id'])['status'],'ready')

    def test_clip_corruption_and_changed_source_are_rejected(self):
        self.ready();clip,_=self.keeper.verified_clip(self.report['id']);clip.write_bytes(b'changed evidence')
        with self.assertRaises(HTTPException):self.keeper.verified_clip(self.report['id'])
        self.source.write_bytes(b'different source')
        with self.assertRaises(HTTPException):self.keeper.retain(self.report)

    def test_interrupted_job_can_be_retried_and_failed_encoder_cleans_partial(self):
        folder=self.keeper.directory(self.report['id']);write_json(folder/'evidence.json',dict(status='saving'))
        self.assertEqual(self.keeper.status(self.report['id'])['status'],'interrupted')
        def fail(source,start,end,destination):
            destination.write_bytes(b'partial');raise RuntimeError('private decoder diagnostic')
        self.keeper.encoder=fail;self.keeper.retain(self.report);self.keeper.jobs[self.report['id']].result(timeout=5)
        state=self.keeper.status(self.report['id']);self.assertEqual(state['status'],'error')
        self.assertNotIn('private',state['message']);self.assertFalse((folder/'clip.part.mp4').exists())

    def test_storage_budget_is_enforced_after_encoding(self):
        self.keeper.budget=1;self.keeper.retain(self.report);self.keeper.jobs[self.report['id']].result(timeout=5)
        self.assertEqual(self.keeper.status(self.report['id'])['status'],'error')
        self.assertFalse((self.keeper.directory(self.report['id'])/'clip.mp4').exists())

    def test_capture_rejects_source_changes_during_encoding(self):
        encoder=self.keeper.encoder
        def changing(source,start,end,destination):
            info=encoder(source,start,end,destination);source.write_bytes(b'new source');return info
        self.keeper.encoder=changing;self.keeper.retain(self.report);self.keeper.jobs[self.report['id']].result(timeout=5)
        self.assertEqual(self.keeper.status(self.report['id'])['status'],'error')

    def test_duplicate_requests_share_one_bounded_job_and_removal_waits(self):
        entered=threading.Event();release=threading.Event();encoder=self.keeper.encoder
        def slow(*args):
            entered.set();release.wait(3);return encoder(*args)
        self.keeper.encoder=slow
        try:
            self.keeper.retain(self.report);self.assertTrue(entered.wait(2))
            self.assertTrue(self.keeper.busy_session(self.report['session_id']))
            self.keeper.retain(self.report)
            with self.assertRaises(HTTPException):self.keeper.remove(self.report['id'])
        finally:release.set()
        self.keeper.jobs[self.report['id']].result(timeout=5);self.assertEqual(len(self.calls),1)
        self.assertFalse(self.keeper.busy_session(self.report['session_id']))

    def test_precise_task_is_dev_only_empty_and_cannot_overwrite_labels(self):
        task=self.task()['task'];folder=self.root/'labeling'/REVIEW_PROJECT
        project=read_json(folder/'project.json');window=project['windows'][0]
        self.assertEqual(window['target'],[.2,3.5]);self.assertEqual(window['split'],'dev')
        self.assertTrue(window['model_exposed']);self.assertFalse(window['evaluation_eligible'])
        self.assertFalse((folder/'labels'/f"{task['window']}.json").exists())
        self.assertEqual(self.keeper.make_task(self.report,'Example Show',.2,3.5)['task']['window'],task['window'])
        with self.assertRaises(HTTPException):self.keeper.make_task(self.report,'Different title',.2,3.5)
        with self.assertRaises(HTTPException):self.keeper.make_task(self.report,'Example Show',0,10)

    def test_promotion_needs_human_labels_known_train_title_and_no_heldout_conflict(self):
        self.task()
        with self.assertRaises(HTTPException):self.keeper.promote(self.report)
        task,path=self.labels()
        for split in ('val','test','dev',None):
            write_json(self.root/'labeling/library.json',dict(splits={'example show':split} if split else {}))
            with self.assertRaises(HTTPException):self.keeper.promote(self.report)
        write_json(self.root/'labeling/library.json',dict(splits={'example show':'train'}))
        write_json(self.root/'labeling/heldout/project.json',dict(windows=[dict(title='Example.Show',split='test')]))
        with self.assertRaises(HTTPException):self.keeper.promote(self.report)
        (self.root/'labeling/heldout/project.json').unlink()
        result=self.keeper.promote(self.report)
        self.assertFalse(result['trained']);self.assertFalse(result['model_changed'])
        self.assertEqual(result,self.keeper.promote(self.report))
        promoted=read_json(self.root/'labeling'/TRAIN_PROJECT/'project.json')
        self.assertEqual(len(promoted['windows']),1);self.assertEqual(promoted['windows'][0]['split'],'train')
        self.assertTrue(read_json(self.root/'labeling'/TRAIN_PROJECT/'labels'/f"{task['window']}.json")['drafts_shown'])
        self.assertEqual(read_json(path)['updated'],'2026-10-15T01:00:00')

    def test_alias_conflict_and_protected_source_title_cannot_be_bypassed(self):
        self.labels();library=self.root/'labeling/library.json'
        write_json(library,dict(splits={'example show':'train','Example.Show':'test'}))
        with self.assertRaises(HTTPException):self.keeper.promote(self.report)
        write_json(library,dict(splits={'example show':'train','protected title':'val'}))
        changed=deepcopy(self.report);changed['context']['title']='Protected Title S01E01'
        with self.assertRaises(HTTPException):self.keeper.promote(changed)

    def test_old_definition_labels_cannot_enter_training(self):
        task,path=self.labels();labels=read_json(path);labels['definitions_revision']=0;write_json(path,labels)
        write_json(self.root/'labeling/library.json',dict(splits={'example show':'train'}))
        with self.assertRaisesRegex(HTTPException,'older category definitions'):self.keeper.promote(self.report)

    def test_removing_retained_clip_preserves_explicitly_copied_labeling_task(self):
        task=self.task()['task'];self.keeper.remove(self.report['id'])
        self.assertEqual(self.keeper.status(self.report['id'])['status'],'removed')
        self.assertTrue((self.root/'labeling'/REVIEW_PROJECT/'clips'/f"{task['window']}.mp4").is_file())
        self.assertEqual(self.keeper.status(self.report['id'])['task']['window'],task['window'])

    def test_label_api_records_prior_model_exposure_even_if_client_omits_it(self):
        task=self.task()['task'];app=FastAPI();app.include_router(label_router(self.root));client=TestClient(app)
        categories={c:dict(status='reviewed',intervals=[],ambiguous=[]) for c in LABEL_CATEGORIES}
        response=client.put(f"/api/labeling/{REVIEW_PROJECT}/labels/{task['window']}",json=dict(categories=categories,drafts_shown=False))
        self.assertEqual(response.status_code,200,response.text);self.assertTrue(response.json()['drafts_shown'])
        clip=self.root/'labeling'/REVIEW_PROJECT/'clips'/f"{task['window']}.mp4"
        clip.write_bytes(b'changed clip')
        self.assertEqual(client.get(f"/api/labeling/{REVIEW_PROJECT}/clip/{task['window']}").status_code,409)
        self.assertEqual(client.put(f"/api/labeling/{REVIEW_PROJECT}/labels/{task['window']}",json=dict(categories=categories)).status_code,409)

    def test_report_creation_keeps_retention_optional_and_evidence_remains_readable(self):
        snapshot=dict(id='a'*32,title='Example Show',duration=12,live=False,retained_start=0,config={},generation=0,
            updated=0,status='complete',coverage=[[0,12]],events=[])
        def context(_):return dict(snapshot=deepcopy(snapshot),source_ref=source_stamp(self.source))
        app=FastAPI();app.include_router(router(self.root,context,lambda row:self.source.is_file(),self.keeper));client=TestClient(app)
        payload=dict(id=str(uuid4()),session_id='a'*32,kind='missed_content',position=4,generation=0,snapshot_updated=0,
            policy=default_policy(),viewer_action=dict(visual='allow',audio='allow',blur=0))
        response=client.post('/api/feedback',json=payload);self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(response.json()['evidence']['status'],'none');self.assertEqual(self.calls,[])
        path='/api/feedback/'+payload['id'];self.assertEqual(client.post(path+'/evidence').status_code,200)
        self.keeper.jobs[payload['id']].result(timeout=5);self.source.unlink()
        self.assertEqual(client.get(path+'/evidence').content,b'fake encoded evidence')
        row=client.get(path).json();self.assertFalse(row['media_available']);self.assertEqual(row['evidence']['status'],'ready')
        self.assertEqual(client.delete(path+'/evidence').status_code,200)

    def test_lost_response_retry_does_not_recreate_evidence_that_was_explicitly_removed(self):
        snapshot=dict(title='Example',duration=12,live=False,retained_start=0,config={},generation=0,
                      updated=0,status='complete',coverage=[[0,12]],events=[])
        def context(_):return dict(snapshot=deepcopy(snapshot),source_ref=source_stamp(self.source))
        app=FastAPI();app.include_router(router(self.root,context,lambda row:True,self.keeper));client=TestClient(app)
        payload=dict(id=str(uuid4()),session_id='a'*32,kind='missed_content',position=3,generation=0,snapshot_updated=0,
            policy=default_policy(),viewer_action=dict(visual='allow',audio='allow',blur=0),retain_clip=True)
        self.assertEqual(client.post('/api/feedback',json=payload).status_code,200)
        self.keeper.jobs[payload['id']].result(timeout=5)
        self.assertEqual(client.delete('/api/feedback/'+payload['id']+'/evidence').status_code,200)
        row=client.post('/api/feedback',json=payload).json()
        self.assertEqual(row['evidence']['status'],'removed');self.assertEqual(len(self.calls),1)

    def test_old_pending_request_fingerprint_survives_new_optional_field(self):
        snapshot=dict(title='Example',duration=12,live=False,retained_start=0,config={},generation=0,updated=0,
                      status='complete',coverage=[[0,12]],events=[])
        request=Report(id=uuid4(),session_id='a'*32,kind='missed_content',position=3,generation=0,snapshot_updated=0,
                       policy=default_policy(),viewer_action=dict(visual='allow',audio='allow',blur=0))
        store=Store(self.root);row=store.add(request,lambda _:dict(snapshot=deepcopy(snapshot),source_ref=source_stamp(self.source)))
        legacy=request.model_dump(mode='json');legacy.pop('retain_clip')
        import hashlib
        expected=hashlib.sha256(json.dumps(legacy,sort_keys=True).encode()).hexdigest()
        self.assertEqual(store.existing(str(request.id),expected)['id'],row['id'])


class RealEvidenceEncodingTests(unittest.TestCase):
    def test_retained_native_frames_match_the_original_time_and_reopen_without_source(self):
        import io
        import numpy as np
        from PIL import Image
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'demo.mp4';make_demo(source)
            report=dict(id=str(uuid4()),status='confirmed',category='violence',kind='missed_content',window=[1.2,3.7],
                        context=dict(title='Generated demo',source_ref=source_stamp(source),live=False))
            keeper=Evidence(root,lambda _:source)
            try:
                keeper.retain(report);keeper.jobs[report['id']].result(timeout=20)
                clip,info=keeper.verified_clip(report['id'])
                self.assertAlmostEqual(info['duration'],2.5,delta=.15)
                image=lambda path,t:np.asarray(Image.open(io.BytesIO(preview_frame(path,t,[])))).astype(float)
                self.assertLess(np.abs(image(source,2.2)-image(clip,1)).mean(),12)
                source.unlink();self.assertTrue(keeper.verified_clip(report['id'])[0].is_file())
            finally:keeper.close()


if __name__=='__main__':unittest.main()
