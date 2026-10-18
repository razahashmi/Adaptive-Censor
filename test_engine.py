"""Timing, coverage, cancellation, API and word-matching regression tests."""
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from analysis_engine import Engine, MediaJob, ScanConfig, keyword_events


class CoverageTests(unittest.TestCase):
    def test_seek_hole_does_not_become_analyzed(self):
        job = MediaJob('id', 'test', Path('unused'), 31, False, active=True)
        job.chunks = {0: [], 3: [], 4: [], 5: []}
        self.assertEqual(job.coverage(), [[0, 6], [18, 31]])
        job.position = 7
        self.assertEqual(job.next_chunk(), 1)
        job.position = 20
        self.assertIsNone(job.next_chunk())

    def test_lookahead_is_bounded(self):
        job = MediaJob('id', 'test', Path('unused'), 3600, False, active=True)
        job.config = ScanConfig(lookahead=15)
        job.chunks = {0: [], 1: [], 2: []}
        self.assertIsNone(job.next_chunk())
        job.position = 6
        self.assertEqual(job.next_chunk(), 3)

    def test_speech_model_change_invalidates_chunks_but_preserves_manual_review(self):
        with patch('analysis_engine.Models') as fake:
            fake.return_value.root=None
            engine=Engine()
        try:
            job=MediaJob('test','test',Path('unused'),12,False)
            job.chunks={0:[]};job.timings=[.4]
            job.manual=[{'id':'manual'}];job.ignored={'manual','automatic'}
            engine.configure(job,ScanConfig(speech_model='small.en'))
            self.assertEqual(job.generation,1)
            self.assertEqual(job.chunks,{})
            self.assertEqual(job.timings,[])
            self.assertEqual(job.ignored,{'manual'})
        finally:
            engine.close()

    def test_workers_analyze_different_chunks_concurrently(self):
        import threading
        lock, active, seen = threading.Lock(), [], []
        both_running = threading.Event()

        class FakeModels:
            def __init__(self, **kwargs):
                self.workers = kwargs.get('workers')

            def analyze(self, path, start, *args):
                with lock:
                    active.append(start); seen.append(start)
                    if len(active) == 2:
                        both_running.set()
                both_running.wait(2)
                time.sleep(.05)
                with lock:
                    active.remove(start)
                return []
        with patch('analysis_engine.Models', FakeModels):
            engine = Engine(workers=2)
        try:
            self.assertEqual(engine.models.workers, 2)
            job = MediaJob('test', 'test', Path('unused'), 24, False, active=True)
            with engine.condition:
                engine.jobs[job.id] = job
                engine.condition.notify_all()
            self.assertTrue(both_running.wait(3))
            deadline = time.monotonic() + 5
            while len(job.chunks) < 4 and time.monotonic() < deadline:
                time.sleep(.05)
            self.assertEqual(sorted(job.chunks), [0, 1, 2, 3])
            self.assertEqual(sorted(seen), [0, 6, 12, 18])  # no chunk analyzed twice
            self.assertEqual(job.running, set())
        finally:
            engine.close()

    def test_generation_change_cannot_commit_stale_work(self):
        import threading
        started, release = threading.Event(), threading.Event()
        class FakeModels:
            def __init__(self, **kwargs):
                pass

            def analyze(self, *args):
                started.set(); release.wait(2); return [{"old": True}]
        with patch('analysis_engine.Models', FakeModels):
            engine = Engine()
        try:
            job = MediaJob('test', 'test', Path('unused'), 12, False, active=True)
            with engine.condition:
                engine.jobs[job.id] = job
                engine.condition.notify_all()
            self.assertTrue(started.wait(2))
            engine.configure(job, ScanConfig(phrases=['changed']))
            with engine.condition:
                job.active = False
            release.set()
            time.sleep(.05)
            self.assertEqual(job.chunks, {})
        finally:
            release.set(); engine.close()


class PhraseTests(unittest.TestCase):
    def test_exception_and_word_boundaries(self):
        words = [('classic', 0, .4, .9), ('bad', 1, 1.2, .9), ('word', 1.3, 1.5, .8), ('bad', 2, 2.3, .7)]
        matches = keyword_events(words, ['ass', 'bad', 'bad word'], ['bad word'])
        self.assertEqual([(e['start'], e['end']) for e in matches], [(2, 2.3)])

    def test_long_pause_breaks_phrase(self):
        self.assertEqual(keyword_events([('bad', 0, .2, 1), ('word', 3, 4, 1)], ['bad word'], []), [])

    def test_nonfinite_and_oversized_config_rejected(self):
        for values in [{'lookahead': float('nan')}, {'fps': 20}, {'phrases': ['word'] * 301}, {'device': 'remote'}, {'speech_model':'untrusted/model'}]:
            with self.assertRaises(ValueError):
                ScanConfig(**values)


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        os.environ['ADAPTIVE_CENSOR_WORKDIR'] = cls.temp.name
        from fastapi.testclient import TestClient
        import server
        cls.server = server
        cls.client = TestClient(server.app)
        cls.client.__enter__()
        cls.headers = {'x-censor-token': cls.client.get('/api/bootstrap').json()['token']}

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)
        cls.temp.cleanup()

    def test_cross_origin_and_missing_token_are_rejected(self):
        self.assertEqual(self.client.post('/api/demo').status_code, 403)
        headers = dict(self.headers, origin='https://untrusted.example')
        self.assertEqual(self.client.post('/api/demo', headers=headers).status_code, 403)
        self.assertEqual(self.client.get('/api/bootstrap', headers={'host':'untrusted.example'}).status_code, 400)

    def test_bootstrap_offers_only_compatible_region_models(self):
        from labeling import write_json
        from provenance import file_hash
        with tempfile.TemporaryDirectory() as temporary,patch.object(self.server,'ROOT',Path(temporary)):
            root=Path(temporary);weight=root/'clip/onnx/vision_model_quantized.onnx'
            weight.parent.mkdir(parents=True);weight.write_bytes(b'fixture')
            head=dict(mean=[0]*512,scale=[1]*512,coef=[1]*512,intercept=0,decision_threshold=.5,clip_identity=file_hash(weight))
            write_json(root/'roi-models/good.json',head)
            write_json(root/'roi-models/stale.json',dict(head,clip_identity='old'))
            write_json(root/'roi-models/incomplete.json',dict(clip_identity=file_hash(weight)))
            write_json(root/'roi-models/list.json',[])
            self.assertEqual(self.client.get('/api/bootstrap').json()['anatomy_roi_models'],[])
            write_json(root/'roi-models/review-options.json',{name:file_hash(root/f'roi-models/{name}.json') for name in ('good','stale','incomplete','list')})
            self.assertEqual(self.client.get('/api/bootstrap').json()['anatomy_roi_models'],['good'])
            weight.write_bytes(b'changed fixture')
            self.assertEqual(self.client.get('/api/bootstrap').json()['anatomy_roi_models'],[])

    def test_demo_range_request_and_manual_validation(self):
        response = self.client.post('/api/demo', headers=self.headers)
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertEqual(len(data['events']), 3)
        source = self.client.get(f"/api/media/{data['id']}/source", headers={'range': 'bytes=0-127'})
        self.assertEqual(source.status_code, 206)
        self.assertEqual(len(source.content), 128)
        invalid = self.client.post(f"/api/media/{data['id']}/events", headers=self.headers,
                                   json={'start':10,'end':2,'category':'violence'})
        self.assertEqual(invalid.status_code, 422)
        deleted = self.client.delete(f"/api/media/{data['id']}", headers=self.headers)
        self.assertEqual(deleted.status_code, 200)

    def test_review_confirmation_is_persisted_and_unknown_ids_rejected(self):
        data=self.client.post('/api/demo',headers=self.headers).json()
        job=self.server.engine.jobs[data['id']]
        event=job.manual[0];event['review_required']=True
        url=f"/api/media/{job.id}/review"
        confirmed=self.client.post(url,headers=self.headers,json=dict(id=event['id'],enabled=True,confirmed=True))
        self.assertEqual(confirmed.status_code,200)
        self.assertTrue(confirmed.json()['events'][0]['confirmed'])
        import json
        saved=json.loads((job.path.parent/'state.json').read_text())
        self.assertIn(event['id'],saved['confirmed'])
        undone=self.client.post(url,headers=self.headers,json=dict(id=event['id'],enabled=True,confirmed=False))
        self.assertFalse(undone.json()['events'][0]['confirmed'])
        self.assertEqual(self.client.post(url,headers=self.headers,json=dict(id='missing',enabled=True)).status_code,404)
        self.client.delete(f'/api/media/{job.id}',headers=self.headers)


class PersistenceTests(unittest.TestCase):
    def test_cached_chunks_and_manual_corrections_survive_restart(self):
        from core import make_demo
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {'ADAPTIVE_CENSOR_WORKDIR': temp}):
            path=Path(temp)/'sessions-v2'/'fixture'/'source.mp4'
            make_demo(path)
            first=Engine()
            job=first.add(path,'Persistence fixture')
            job.chunks[0]=[{'id':'one','start':1,'end':2,'category':'language','confidence':1}]
            job.manual=[{'id':'manual','start':3,'end':4,'category':'violence','confidence':1}]
            job.ignored.add('one')
            job.confirmed.add('one')
            first.persist(job,0); first.close()
            second=Engine()
            try:
                restored=second.jobs[job.id]
                self.assertEqual(restored.chunks,job.chunks)
                self.assertEqual(restored.manual,job.manual)
                self.assertEqual(restored.ignored,{'one'})
                self.assertEqual(restored.confirmed,{'one'})
                self.assertFalse(restored.active)
            finally:
                second.close()


if __name__ == '__main__':
    unittest.main()
