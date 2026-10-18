import copy
import json
from pathlib import Path
import tempfile
import unittest
from public_benchmark import validate_suite,read,check_pack,identity,run

class PublicBenchmarkTests(unittest.TestCase):
    def suite(self):return read(Path(__file__).parent/'samples/public-suite.json')

    def test_unknown_categories_stay_unreviewed_and_drafts_are_explicit(self):
        suite=validate_suite(self.suite())
        self.assertEqual(suite['review']['status'],'assistant_draft')
        self.assertFalse(suite['review']['independent'])
        organ=next(c for c in suite['cases'] if c['id']=='organ-prop')
        self.assertNotIn('gore',organ['reviewed'])
        self.assertEqual(next(c for c in suite['cases'] if c['id']=='action')['reviewed'],[])

    def test_bad_annotations_sources_and_duplicate_cases_are_rejected(self):
        original=self.suite()
        for change in ('duplicate','outside','unreviewed','missing_source'):
            suite=copy.deepcopy(original)
            if change=='duplicate':suite['cases'].append(suite['cases'][0])
            if change=='outside':suite['cases'][0]['events']['kissing']=[[1,99]]
            if change=='unreviewed':suite['cases'][0]['events']['gore']=[[1,2]]
            if change=='missing_source':suite['cases'][0]['source']='unknown'
            with self.assertRaises(ValueError):validate_suite(suite)

    def test_changed_media_is_rejected_before_scoring(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);suite=self.suite()
            manifest={}
            for case in suite['cases']:
                path=folder/(case['id']+'.mp4');path.write_bytes(b'test fixture')
                manifest[path.name]=dict(sha256=identity(path))
            (folder/'FILES.json').write_text(json.dumps(manifest))
            check_pack(folder,suite)
            (folder/'kissing.mp4').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'differs'):check_pack(folder,suite)

    def test_drafts_cannot_silently_become_an_accuracy_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);suite=self.suite();manifest={}
            for case in suite['cases']:
                path=folder/(case['id']+'.mp4');path.write_bytes(b'test fixture');manifest[path.name]=dict(sha256=identity(path))
            (folder/'FILES.json').write_text(json.dumps(manifest));(folder/'suite.json').write_text(json.dumps(suite))
            with self.assertRaisesRegex(ValueError,'drafts'):run(folder,folder/'result.json')
            self.assertFalse((folder/'result.json').exists())

if __name__=='__main__':unittest.main()
