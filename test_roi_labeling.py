from pathlib import Path
import tempfile
import unittest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from labeling import read_json,write_json
from roi_labeling import router
from provenance import file_hash


class RegionLabelingTests(unittest.TestCase):
    def test_changed_crop_cannot_be_shown_as_the_original_training_example(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);folder=root/'roi/human-1';folder.mkdir(parents=True)
            image=folder/'roi-0000.jpg';image.write_bytes(b'original')
            write_json(folder/'dataset.json',dict(review_api=1,rows=[dict(id='roi-0000',split='train')],
                       images_sha256={image.name:file_hash(image)}))
            app=FastAPI();app.include_router(router(root));client=TestClient(app)
            self.assertEqual(client.get('/api/roi/human-1/images/roi-0000.jpg').status_code,200)
            image.write_bytes(b'changed')
            self.assertEqual(client.get('/api/roi/human-1/images/roi-0000.jpg').status_code,409)

    def test_saved_choice_survives_reload_and_stale_revision_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);folder=root/'roi/human-1'
            write_json(folder/'dataset.json',dict(review_api=1,feature_views=['local','body'],rows=[
                dict(id='roi-0000',title='Example',time=2.,split='train',label=None,labeler=None),
                dict(id='roi-0001',title='Example',time=4.,split='train',label=None,labeler=None)]))
            (folder/'roi-0000.jpg').write_bytes(b'fixture')
            app=FastAPI();app.include_router(router(root));client=TestClient(app)
            data=client.get('/api/roi/human-1').json()
            payload=dict(label=1,revision=data['revision'],dataset_sha256=data['dataset_sha256'])
            saved=client.put('/api/roi/human-1/labels/roi-0000',json=payload)
            self.assertEqual(saved.status_code,200)
            data2=client.get('/api/roi/human-1').json()
            self.assertEqual(data2['rows'][0]['label'],1);self.assertTrue(data2['rows'][0]['reviewed'])
            self.assertEqual(read_json(folder/'reviewed.json')['rows'][0]['labeler'],'human')
            self.assertEqual(client.put('/api/roi/human-1/labels/roi-0000',json=dict(payload,label=0)).status_code,409)
            payload.update(revision=data2['revision'],label=True)
            self.assertEqual(client.put('/api/roi/human-1/labels/roi-0001',json=payload).status_code,400)
            self.assertEqual(client.put('/api/roi/human-1/labels/roi-0001',json={
                k:v for k,v in payload.items() if k!='label'}).status_code,400)
            payload['label']=None
            self.assertEqual(client.put('/api/roi/human-1/labels/roi-0001',json=payload).status_code,200)
            self.assertEqual(client.get('/api/roi-projects').json()[0]['reviewed'],2)
            self.assertEqual(client.get('/api/roi/human-1/images/roi-9999.jpg').status_code,404)
            self.assertEqual(client.get('/api/roi/human-1/images/roi-0000.jpg').status_code,200)
            original=read_json(folder/'dataset.json');original['rows'][0]['time']=3.;write_json(folder/'dataset.json',original)
            self.assertEqual(client.get('/api/roi/human-1').status_code,409)

    def test_legacy_galleries_and_nontraining_rows_cannot_be_changed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);path=root/'roi/old/dataset.json'
            write_json(path,dict(rows=[]))
            app=FastAPI();app.include_router(router(root));client=TestClient(app)
            self.assertEqual(client.get('/api/roi-projects').json(),[])
            self.assertEqual(client.get('/api/roi/old').status_code,404)
            write_json(path,dict(review_api=1,rows=[dict(id='roi-0000',split='val')]))
            self.assertEqual(client.get('/api/roi/old').status_code,400)
