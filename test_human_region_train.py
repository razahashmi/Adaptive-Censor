"""Synthetic fixtures exercise calibration isolation, never production labels."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
from labeling import read_json,write_json
from provenance import file_hash


@unittest.skipUnless(importlib.util.find_spec('sklearn'),'Training dependencies unavailable')
class HumanRegionTrainingTests(unittest.TestCase):
    def fixture(self,folder):
        rows=[];features=[];rng=np.random.default_rng(8)
        for title in range(16):
            for i in range(4):
                value=i%2
                rows.append(dict(id=f'roi-{len(rows):04}',title=f'Title {title}',project='fixture',
                                 window=f'window-{title}-{i}',split='train',label=value,labeler='human',
                                 reviewed=True,detector_class='FEMALE_BREAST_EXPOSED'))
                features.append(rng.normal(0,.05,1024)+(2*value-1)*.5)
        x=np.asarray(features,dtype=np.float32);np.savez_compressed(folder/'features.npz',embeddings=x)
        data=dict(review_api=1,rows=rows,feature_views=['local','body'],clip_identity='fixture-clip',
                  anatomy_model='640m',anatomy_identity='fixture-anatomy',feature_identity=file_hash(folder/'features.npz'))
        write_json(folder/'dataset.json',data)
        write_json(folder/'reviewed.json',dict(data,dataset_sha256=file_hash(folder/'dataset.json')))
        return x,rows

    def test_calibration_titles_do_not_enter_fitting_or_refitting(self):
        from human_region_train import train
        from select_windows import normalize_title
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);x,rows=self.fixture(folder)
            result=train(folder/'reviewed.json',folder/'head.json')
            self.assertFalse(set(result['fit_titles'])&set(result['calibration_titles']))
            fit=np.array([normalize_title(r['title']) in result['fit_titles'] for r in rows])
            np.testing.assert_allclose(result['mean'],x[fit].astype(np.float64).mean(0),atol=1e-12)
            self.assertFalse(result['calibration_refit']);self.assertFalse(result['ai_labels_allowed'])
            self.assertEqual(result['negative_confirmations'],3)
            self.assertEqual(result['supported_classes'],['FEMALE_BREAST_EXPOSED'])
            self.assertGreaterEqual(result['calibration_recall'],.97)
            with self.assertRaises(FileExistsError):train(folder/'reviewed.json',folder/'head.json')

    def test_incomplete_or_nonhuman_review_cannot_create_region_model(self):
        from human_region_train import train
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);self.fixture(folder);path=folder/'reviewed.json';data=read_json(path)
            for field,value in [('reviewed',False),('labeler','codex')]:
                altered=read_json(path);altered['rows'][0][field]=value;write_json(path,altered)
                with self.assertRaisesRegex(ValueError,'Complete every crop'):train(path,folder/'head.json')
                self.assertFalse((folder/'head.json').exists());write_json(path,data)

    def test_balanced_partition_is_deterministic_and_does_not_weaken_support(self):
        from human_region_train import balanced_calibration,train
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);x,rows=self.fixture(folder)
            held=balanced_calibration(rows)
            self.assertEqual(held,balanced_calibration(list(reversed(rows))))
            for value in (0,1):
                ts={r['title'].lower() for r in rows if r['label']==value}
                self.assertGreaterEqual(len(ts&held),3);self.assertGreaterEqual(len(ts-held),3)
            result=train(folder/'reviewed.json',folder/'balanced.json',class_balanced=True)
            fit=np.array([r['title'].lower() in result['fit_titles'] for r in rows])
            np.testing.assert_allclose(result['mean'],x[fit].astype(np.float64).mean(0),atol=1e-12)
            self.assertFalse(result['partition_uses_model_scores']);self.assertFalse(result['calibration_refit'])
            altered=[dict(r,split='test') for r in rows]
            with self.assertRaisesRegex(ValueError,'training titles only'):balanced_calibration(altered)
            rare=[dict(r,detector_class='RARE-'+str(i//4)) for i,r in enumerate(rows)]
            with self.assertRaisesRegex(ValueError,'six training titles'):balanced_calibration(rare)


if __name__=='__main__':unittest.main()
