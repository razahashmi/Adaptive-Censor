from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from build_source_release import build
from windows_bootstrap import extract_source,check_archive,inspect_python

class WindowsBootstrapTests(unittest.TestCase):
    def fixture(self,work):
        import json
        root=work/'source';root.mkdir()
        (root/'README.md').write_text('Public example')
        (root/'release-files.json').write_text(json.dumps(dict(version='0.3.0-preview.1',files=['README.md','release-files.json'])))
        return build(root,work/'dist')

    def test_source_checksum_is_checked_before_any_installation(self):
        with tempfile.TemporaryDirectory() as temporary:
            work=Path(temporary);archive=self.fixture(work)
            with self.assertRaisesRegex(ValueError,'checksum'):extract_source(archive,work/'install','0'*64)
            self.assertFalse((work/'install').exists())

    def test_reopening_is_safe_and_source_edits_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            work=Path(temporary);archive=self.fixture(work)
            app,_=extract_source(archive,work/'install')
            self.assertEqual(extract_source(archive,work/'install')[0],app)
            (app/'README.md').write_text('User edit')
            with self.assertRaisesRegex(ValueError,'no file was overwritten'):extract_source(archive,work/'install')
            self.assertEqual((app/'README.md').read_text(),'User edit')

    def test_python_version_and_architecture_must_both_match(self):
        import json
        for version,bits,accepted in [([3,13],64,True),([3,12],64,False),([3,13],32,False)]:
            result=subprocess.CompletedProcess([],0,json.dumps(dict(version=version,bits=bits,executable='python.exe')),'')
            with patch('windows_bootstrap.subprocess.run',return_value=result):
                self.assertEqual(bool(inspect_python(['python'])),accepted)

if __name__=='__main__':unittest.main()
