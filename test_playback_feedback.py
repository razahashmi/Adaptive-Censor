from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from core import default_policy
from playback_feedback import router


class FeedbackTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.exists = True
        self.context = dict(snapshot=dict(title='Fixture title', duration=90., live=False, retained_start=0,
            config={'scenes':True, 'scene_filtering':'review'}, generation=2, updated=8, status='ready',
            coverage=[[0,60]], events=[dict(id='match-1',start=32,end=35,category='violence',
                confidence=.9,source='fixture',enabled=True)]), analysis_identity='pipeline-identity',
            source_ref=dict(session_id='a'*32,filename='source.mp4',size=100,mtime_ns=123))
        self.client = self.make_client()

    def make_client(self):
        def context(session_id):
            if not self.exists or session_id != 'a'*32:
                raise HTTPException(404, 'Media session not found')
            return deepcopy(self.context)
        app=FastAPI();app.include_router(router(self.root, context, lambda row:self.exists and not row['context']['live']))
        return TestClient(app)

    def payload(self, **changes):
        return dict(id=str(uuid4()),session_id='a'*32,kind='wrongly_censored',position=34.25,generation=2,
            snapshot_updated=8,policy=default_policy(),viewer_action=dict(visual='blur',audio='allow',blur=35),**changes)

    def test_one_click_report_preserves_context_without_modifying_detections(self):
        before=deepcopy(self.context)
        response=self.client.post('/api/feedback',json=self.payload())
        self.assertEqual(response.status_code,200,response.text)
        row=response.json()
        self.assertEqual(row['window'],[24.25,36.25])
        self.assertFalse(row['verified_label']);self.assertEqual(row['status'],'pending')
        self.assertEqual(row['context']['analysis_identity'],'pipeline-identity')
        self.assertEqual(row['context']['events_at_receipt'][0]['id'],'match-1')
        self.assertTrue(row['context']['analyzed_at_receipt'])
        self.assertEqual(row['context']['plan_at_receipt'][0]['visual'],'blur')
        self.assertEqual(self.context,before)

    def test_report_survives_server_recreation_and_source_removal(self):
        payload=self.payload();row=self.client.post('/api/feedback',json=payload).json()
        self.exists=False;client=self.make_client()
        self.assertEqual(client.get('/api/feedback/'+row['id']).json()['media_available'],False)
        self.assertEqual(client.post('/api/feedback',json=payload).json()['id'],row['id'])
        self.assertEqual(client.get('/api/feedback').json()['total'],1)
        self.assertEqual(client.post('/api/feedback',json=self.payload()).status_code,404)

    def test_duplicate_retries_are_idempotent_including_concurrent_requests(self):
        payload=self.payload()
        with ThreadPoolExecutor(max_workers=4) as pool:
            results=list(pool.map(lambda _:self.client.post('/api/feedback',json=payload),range(8)))
        self.assertTrue(all(row.status_code==200 for row in results))
        self.assertEqual(self.client.get('/api/feedback').json()['total'],1)
        payload['kind']='missed_content'
        self.assertEqual(self.client.post('/api/feedback',json=payload).status_code,409)

    def test_optional_details_revision_conflicts_and_undo(self):
        row=self.client.post('/api/feedback',json=self.payload()).json();path='/api/feedback/'+row['id']
        changed=self.client.patch(path,json=dict(revision=1,category='violence',note='<script>text only</script>',status='confirmed'))
        self.assertEqual(changed.status_code,200)
        self.assertFalse(changed.json()['verified_label'])
        self.assertEqual(self.client.patch(path,json=dict(revision=1,note='stale')).status_code,409)
        self.assertEqual(self.client.get('/api/feedback?status=confirmed').json()['total'],1)
        self.assertEqual(self.client.patch(path,json=dict(revision=2,status='withdrawn')).status_code,200)
        self.assertEqual(self.client.get('/api/feedback/export').text,'')
        self.assertEqual(self.client.get('/api/feedback?status=withdrawn').json()['total'],1)

    def test_export_and_pagination_keep_reports_separate_from_labels(self):
        for _ in range(3):self.client.post('/api/feedback',json=self.payload())
        page=self.client.get('/api/feedback?limit=2').json();self.assertEqual(page['total'],3)
        next_page=self.client.get('/api/feedback?limit=2&offset=2').json()
        self.assertEqual(len(next_page['reports']),1)
        self.assertNotIn(next_page['reports'][0]['id'],[r['id'] for r in page['reports']])
        response=self.client.get('/api/feedback/export')
        rows=[json.loads(line) for line in response.text.splitlines()]
        self.assertEqual(len(rows),3);self.assertTrue(all(not r['verified_label'] for r in rows))
        self.assertFalse((self.root/'labeling').exists())
        self.assertIn('attachment',response.headers['content-disposition'])

    def test_live_expiration_and_changed_analysis_are_explicit(self):
        self.context['snapshot'].update(live=True,retained_start=40,generation=3,updated=12,coverage=[[40,60]])
        row=self.client.post('/api/feedback',json=self.payload()).json()
        self.assertFalse(row['media_available'])
        self.assertTrue(row['context']['context_changed_since_playback'])
        self.assertFalse(row['context']['analyzed_at_receipt'])
        self.assertEqual(row['position'],34.25)  # Never move the reported time to the current buffer.

    def test_bounds_and_payload_validation(self):
        for field,value in [('position',-1),('position',91),('kind','anything'),('id','../escape'),
                            ('session_id','../escape'),('generation',-1)]:
            payload=self.payload();payload[field]=value
            self.assertEqual(self.client.post('/api/feedback',json=payload).status_code,422,(field,value))
        payload=self.payload();payload['policy']['categories']['unknown']=payload['policy']['categories'].pop('nudity')
        self.assertEqual(self.client.post('/api/feedback',json=payload).status_code,422)
        payload=self.payload();payload['policy']['categories']['nudity']['blur']=float('nan')
        self.assertEqual(self.client.post('/api/feedback',content=json.dumps(payload),headers={'Content-Type':'application/json'}).status_code,422)
        payload=self.payload();payload['url']='https://private.example/token'
        self.assertEqual(self.client.post('/api/feedback',json=payload).status_code,422)
        row=self.client.post('/api/feedback',json=self.payload()).json()
        self.assertEqual(self.client.patch('/api/feedback/'+row['id'],json=dict(revision=1,note='x'*1001)).status_code,422)
        self.assertEqual(self.client.get('/api/feedback?limit=1000').status_code,422)

    def test_window_clamps_to_finite_media_boundaries(self):
        payload=self.payload();payload['position']=0
        self.assertEqual(self.client.post('/api/feedback',json=payload).json()['window'],[0,2])
        payload=self.payload();payload['position']=90
        self.assertEqual(self.client.post('/api/feedback',json=payload).json()['window'],[80,90])


if __name__=='__main__':unittest.main()
