import tempfile
import unittest
from pathlib import Path
from PIL import Image
from fastapi import FastAPI
from fastapi.testclient import TestClient
from labeling import read_json, write_json
from provenance import file_hash
from segmentation_labels import router
from expand_mask_data import separated, combine_masks, combine_video, disagreement_negatives


class ExpansionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
    def tearDown(self): self.temp.cleanup()
    def dataset(self, name, split='train', title='Alpha', color='gray'):
        folder=self.root/'segmentation'/name; folder.mkdir(parents=True)
        rid='mask-00000'; Image.new('RGB',(20,10),color).save(folder/(rid+'.jpg'))
        Image.new('L',(20,10),255).save(folder/(rid+'-draft.png'))
        row=dict(id=rid,title=title,display=title,split=split,group=0,time=8.,width=20,height=10,
                 source_project=name,window=name,source_identity='clip',prompt_boxes=[{}],
                 image_sha256=file_hash(folder/(rid+'.jpg')),draft_sha256=file_hash(folder/(rid+'-draft.png')))
        data=dict(version=1,name=name,split=split,rows=[row],evaluation_scope='Fixture')
        write_json(folder/'dataset.json',data); return folder,data
    def client(self):
        app=FastAPI(); app.include_router(router(self.root)); return TestClient(app)
    def test_selection_is_separated_and_title_capped(self):
        rows=[dict(title=t,source=t,time=x,rank=1.,fresh=t=='B')
              for t in ['A','B'] for x in [40.,42.,70.,100.,130.,160.]]
        selected=separated(rows,{'B':[34.]},6,per_title=3)
        self.assertEqual(len(selected),6); self.assertEqual(selected[0]['title'],'B')
        self.assertNotIn(40.,[r['time'] for r in selected if r['title']=='B'])
        for title in ['A','B']:
            times=[r['time'] for r in selected if r['title']==title]
            self.assertEqual(len(times),3)
            self.assertTrue(all(abs(a-b)>=20 for i,a in enumerate(times) for b in times[i+1:]))
    def test_consolidation_preserves_exposure_and_deduplicates(self):
        folder,_=self.dataset('mask-old'); self.dataset('mask-copy')
        self.client().get('/api/segmentation/mask-old/images/mask-00000/draft')
        before=file_hash(folder/'dataset.json')
        batch=combine_masks(self.root,'mask-combined','train',['mask-old','mask-copy'],'identity')
        self.assertEqual(len(batch['rows']),1)
        self.assertEqual(batch['skipped']['duplicate_picture'],1)
        marker=read_json(self.root/'segmentation/mask-combined/exposure/mask-00000.json')
        self.assertTrue(marker['inherited']); self.assertEqual(before,file_hash(folder/'dataset.json'))
        self.assertEqual(batch,combine_masks(self.root,'mask-combined','train',['mask-old','mask-copy'],'identity'))
        listing=self.client().get('/api/segmentation').json()
        self.assertEqual(next(r for r in listing if r['name']=='mask-combined')['supersedes'],['mask-old','mask-copy'])
        target=self.root/'segmentation/mask-combined'
        response=self.client().put('/api/segmentation/mask-combined/labels/mask-00000',json=dict(
            dataset_sha256=file_hash(target/'dataset.json'),revision=None,status='reviewed',strokes=[],draft_included=False))
        self.assertEqual(response.status_code,200)
        self.assertTrue(self.client().get('/api/segmentation/mask-combined/labels/mask-00000').json()['label']['drafts_shown'])
        self.assertFalse((folder/'labels/mask-00000.json').exists())
        with self.assertRaises(ValueError): combine_masks(self.root,'mask-combined','train',['mask-old'],'changed')
    def test_completed_frames_stay_in_original_queue(self):
        folder,_=self.dataset('mask-done'); self.dataset('mask-open',color='white')
        response=self.client().put('/api/segmentation/mask-done/labels/mask-00000',json=dict(
            dataset_sha256=file_hash(folder/'dataset.json'),revision=None,status='reviewed',strokes=[],draft_included=False))
        self.assertEqual(response.status_code,200); gold=file_hash(folder/'mask-00000-gold.png')
        label=file_hash(folder/'labels/mask-00000.json')
        batch=combine_masks(self.root,'mask-pending','train',['mask-done','mask-open'],'id')
        self.assertEqual(len(batch['rows']),1); self.assertEqual(batch['skipped']['already_reviewed'],1)
        self.assertEqual(label,file_hash(folder/'labels/mask-00000.json'))
        self.assertEqual(gold,file_hash(folder/'mask-00000-gold.png'))
    def test_blind_queue_omits_exposed_frames_and_rejects_wrong_split(self):
        self.dataset('mask-seen','test','Beta'); self.dataset('mask-blind','test','Gamma','white')
        self.client().get('/api/segmentation/mask-seen/images/mask-00000/draft')
        batch=combine_masks(self.root,'mask-held','test',['mask-seen','mask-blind'],'id')
        self.assertEqual(len(batch['rows']),1); self.assertEqual(batch['skipped']['exposed_held_out'],1)
        with self.assertRaises(ValueError): combine_masks(self.root,'mask-wrong','train',['mask-blind'],'id')
    def video(self,name,split='train',title='Alpha',content=b'clip'):
        folder=self.root/'labeling'/name; (folder/'clips').mkdir(parents=True)
        rid='window-'+name; (folder/'clips'/(rid+'.mp4')).write_bytes(content)
        write_json(folder/'project.json',dict(name=name,windows=[dict(id=rid,split=split,title=title)]))
    def test_video_combines_pending_only_and_checks_splits(self):
        self.video('one'); self.video('two',content=b'other')
        combined=combine_video(self.root,'batch',['one','two'],{'splits':{'alpha':'train'}})
        self.assertEqual(len(combined['windows']),2)
        self.assertEqual(combined,combine_video(self.root,'batch',['one','two'],{'splits':{'alpha':'train'}}))
        self.video('bad','test','Beta')
        with self.assertRaises(ValueError): combine_video(self.root,'invalid',['bad'],{'splits':{'beta':'test'}})
    def test_changed_frame_cannot_enter_batch(self):
        folder,_=self.dataset('mask-original'); (folder/'mask-00000.jpg').write_bytes(b'changed')
        with self.assertRaises(ValueError): combine_masks(self.root,'mask-batch','train',['mask-original'],'id')
    def test_disagreements_are_only_training_collection_hints(self):
        base=dict(title='A',source='A',time=40.,rank=.9,kind='possible-content',included=False)
        rows=[dict(base,evidence={'tags':[{'category':'nudity','state':'absent'}]}),
              dict(base,time=70.,evidence={'tags':[{'category':'nudity','state':'uncertain'}]}),
              dict(base,time=100.,error='decode failed')]
        selected=disagreement_negatives(rows,{},3)
        self.assertEqual(len(selected),1); self.assertEqual(selected[0]['kind'],'lookalike')
        self.assertEqual(rows[0]['kind'],'possible-content'); self.assertFalse(rows[0]['included'])


if __name__=='__main__': unittest.main()
