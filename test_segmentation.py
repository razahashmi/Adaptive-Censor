import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from fastapi import FastAPI
from fastapi.testclient import TestClient
from labeling import read_json,write_json
from provenance import file_hash
from segmentation_labels import router,render_strokes
from segmentation_data import export,validate_splits
from segmentation_evaluation import evaluate


class MaskTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.folder,self.data=self.fixture('mask-train-fixture','train','Alpha')
        app=FastAPI();app.include_router(router(self.root));self.client=TestClient(app)
    def tearDown(self):self.tmp.cleanup()
    def fixture(self,name,split,title):
        folder=self.root/'segmentation'/name;folder.mkdir(parents=True)
        rid='mask-00000';Image.new('RGB',(20,10),'gray').save(folder/(rid+'.jpg'))
        Image.new('L',(20,10),255).save(folder/(rid+'-draft.png'))
        row=dict(id=rid,title=title,display=title,split=split,time=1.,group=0,width=20,height=10,
            image_sha256=file_hash(folder/(rid+'.jpg')),draft_sha256=file_hash(folder/(rid+'-draft.png')),source_identity='source',prompt_boxes=[{'hidden':'hint'}])
        data=dict(version=1,name=name,split=split,rows=[row],evaluation_scope='Fixture')
        write_json(folder/'dataset.json',data);return folder,data
    def payload(self,name='mask-train-fixture',**kwargs):
        folder=self.root/'segmentation'/name
        return dict(dataset_sha256=file_hash(folder/'dataset.json'),revision=file_hash(folder/'labels/mask-00000.json'),
                    status='reviewed',strokes=[],draft_included=False,**kwargs)
    def save(self,name='mask-train-fixture',payload=None):
        return self.client.put('/api/segmentation/'+name+'/labels/mask-00000',json=payload or self.payload(name))
    def test_paint_erase_and_invalid_coordinates(self):
        strokes=[dict(mode='paint',radius=.1,points=[[.2,.5],[.8,.5]]),dict(mode='erase',radius=.1,points=[[.5,.5]])]
        mask=np.asarray(render_strokes((20,10),strokes));self.assertGreater(mask.sum(),0);self.assertEqual(mask[0,0],0);self.assertEqual(mask[4,9],0)
        with self.assertRaises(ValueError):render_strokes((20,10),[dict(mode='paint',radius=True,points=[[.5,.5]])])
        with self.assertRaises(ValueError):render_strokes((20,10),[dict(mode='paint',radius=.1,points=[[1.1,.5]])])
    def test_save_ack_revision_and_stale_write_protect_gold(self):
        payload=self.payload();result=self.save(payload=payload);self.assertEqual(result.status_code,200)
        self.assertEqual(result.json()['label']['positive_pixels'],0)
        identity=file_hash(self.folder/'mask-00000-gold.png')
        payload['strokes']=[dict(mode='paint',radius=.1,points=[[.5,.5]])]
        self.assertEqual(self.save(payload=payload).status_code,409)
        self.assertEqual(file_hash(self.folder/'mask-00000-gold.png'),identity)
    def test_draft_exposure_persists_after_clear_and_blocks_blind_export(self):
        folder,data=self.fixture('mask-test-fixture','test','Beta')
        result=self.client.get('/api/segmentation/mask-test-fixture');self.assertNotIn('prompt_boxes',result.json()['rows'][0])
        self.assertFalse((folder/'exposure/mask-00000.json').exists())
        self.assertEqual(self.client.get('/api/segmentation/mask-test-fixture/images/mask-00000/draft').status_code,200)
        self.assertEqual(self.save('mask-test-fixture').status_code,200)
        label=read_json(folder/'labels/mask-00000.json');self.assertTrue(label['drafts_shown'])
        with self.assertRaisesRegex(ValueError,'without model-draft'):evaluate(folder)
        result=export(self.root,['mask-test-fixture'],self.root/'export',drafts=True);self.assertEqual(result['mask-test-fixture']['exported'],0)
    def test_changed_source_invalid_payload_and_unsure(self):
        self.assertEqual(self.client.put('/api/segmentation/mask-train-fixture/labels/mask-00000',json={'status':'reviewed'}).status_code,409)
        payload=self.payload();payload['strokes']=[dict(mode='paint',radius=.1,points=[[-1,0]])]
        self.assertEqual(self.save(payload=payload).status_code,400)
        payload=self.payload();payload['status']='unsure';self.assertEqual(self.save(payload=payload).status_code,200)
        self.assertIsNone(read_json(self.folder/'labels/mask-00000.json')['positive_pixels'])
        with self.assertRaisesRegex(ValueError,'No definite'):evaluate(self.folder)
        (self.folder/'mask-00000.jpg').write_bytes(b'changed');self.assertEqual(self.save().status_code,409)
    def test_pseudo_labels_export_only_for_training_and_coco_roundtrip(self):
        self.fixture('mask-val-fixture','val','Gamma');self.fixture('mask-test-fixture','test','Beta')
        result=export(self.root,['mask-train-fixture','mask-val-fixture','mask-test-fixture'],self.root/'export',drafts=True)
        self.assertEqual([v['exported'] for v in result.values()],[1,0,0])
        data=read_json(self.root/'export/mask-train-fixture.json');self.assertEqual(data['images'][0]['labeler'],'model')
        self.assertEqual(data['annotations'][0]['segmentation']['counts'],[0,200]);self.assertTrue((self.root/'export'/data['images'][0]['file_name']).is_file())
        with self.assertRaises(FileExistsError):export(self.root,['mask-train-fixture'],self.root/'export',drafts=True)
    def test_title_leakage_rejected_and_missing_test_labels_not_scored(self):
        _,test=self.fixture('mask-test-fixture','test','ALPHA')
        with self.assertRaisesRegex(ValueError,'Title crosses'):validate_splits([self.data,test])
        with self.assertRaisesRegex(ValueError,'Incomplete'):evaluate(self.folder)
    def test_human_blind_mask_counts_and_tamper_detection(self):
        self.assertEqual(self.save().status_code,200);report=evaluate(self.folder)
        self.assertEqual(report['counts']['fp'],200);self.assertEqual(report['clean_pixel_disturbance'],1)
        self.assertIsNone(report['coverage']);self.assertFalse(report['qualification_passed'])
        (self.folder/'mask-00000-gold.png').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'Reviewed mask file changed'):evaluate(self.folder)

    def test_title_counts_do_not_inherit_previous_title_totals(self):
        import shutil
        row=dict(self.data['rows'][0],id='mask-00001',title='Beta',display='Beta',group=1)
        for suffix in ('.jpg','-draft.png'):shutil.copyfile(self.folder/('mask-00000'+suffix),self.folder/('mask-00001'+suffix))
        self.data['rows'].append(row);write_json(self.folder/'dataset.json',self.data)
        self.assertEqual(self.save().status_code,200)
        payload=self.payload();payload['revision']=None
        self.assertEqual(self.client.put('/api/segmentation/mask-train-fixture/labels/mask-00001',json=payload).status_code,200)
        report=evaluate(self.folder);self.assertEqual(report['counts']['fp'],400)
        self.assertEqual([v['counts']['fp'] for v in report['by_title'].values()],[200,200])

    def test_ai_test_label_is_not_exported_as_independent_ground_truth(self):
        folder,_=self.fixture('mask-test-fixture','test','Beta');self.assertEqual(self.save('mask-test-fixture').status_code,200)
        label=read_json(folder/'labels/mask-00000.json');label['labeler']='model';write_json(folder/'labels/mask-00000.json',label)
        result=export(self.root,['mask-test-fixture'],self.root/'export',drafts=True)
        self.assertEqual(result['mask-test-fixture']['exported'],0)

    def test_label_definition_is_bound_before_evaluation(self):
        self.assertEqual(self.save().status_code,200)
        label=read_json(self.folder/'labels/mask-00000.json');label['definition_sha256']='outdated'
        write_json(self.folder/'labels/mask-00000.json',label)
        with self.assertRaisesRegex(ValueError,'Mask definition changed'):evaluate(self.folder)

    def test_blind_subset_excludes_exposed_frame_and_its_nearby_sibling(self):
        import shutil
        folder,data=self.fixture('mask-test-blind','test','Held')
        for i,group in [(1,0),(2,1)]:
            rid=f'mask-{i:05d}'
            for suffix in ('.jpg','-draft.png'):shutil.copyfile(folder/('mask-00000'+suffix),folder/(rid+suffix))
            data['rows'].append(dict(data['rows'][0],id=rid,group=group))
        write_json(folder/'dataset.json',data)
        for row in data['rows']:
            payload=self.payload('mask-test-blind');payload['revision']=None
            self.assertEqual(self.client.put('/api/segmentation/mask-test-blind/labels/'+row['id'],json=payload).status_code,200)
        self.client.get('/api/segmentation/mask-test-blind/images/mask-00000/draft')
        with self.assertRaisesRegex(ValueError,'without model-draft exposure'):evaluate(folder)
        report=evaluate(folder,blind_groups_only=True)
        self.assertEqual(report['scored_frames'],1);self.assertEqual(report['exposure_excluded_frames'],2)
        self.assertEqual(report['exposure_excluded_groups'],1);self.assertEqual(report['counts']['fp'],200)
        self.assertEqual(report['clean_frame_false_mask_rate'],1);self.assertEqual(report['positive_titles'],0)
        with self.assertRaisesRegex(ValueError,'evaluation splits only'):evaluate(self.folder,blind_groups_only=True)


if __name__=='__main__':unittest.main()
