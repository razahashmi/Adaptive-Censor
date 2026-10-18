import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from build_source_release import audit_text, build, safe_name, verify, MEDIA, sha, REVIEWED_MEDIA_SOURCES


class ReleaseTests(unittest.TestCase):
    def fixture(self, path):
        root=path/'repo';root.mkdir()
        for name,text in {'module.py':'print("example")\n','PUBLIC.md':'Public introduction\n',
                          'README.md':'Private development notes\n','LICENSE':'MIT\n'}.items():
            (root/name).write_text(text,encoding='utf-8')
        spec=dict(version='0.3.0-preview.1',files=['module.py','README.md','PUBLIC.md','LICENSE','release-files.json'],
                  overrides={'README.md':'PUBLIC.md'})
        (root/'release-files.json').write_text(json.dumps(spec),encoding='utf-8')
        return root

    def licensed_fixture(self,path):
        root=self.fixture(path)
        assets={
            'docs/media/adaptive-censor-demo.mp4':b'\x00\x00\x00\x18ftypisom\r\n\x00fixture',
            'docs/media/adaptive-censor-preview.gif':b'GIF89a\r\n\x00fixture',
            'docs/media/adaptive-censor-poster.png':b'\x89PNG\r\n\x1a\nfixture'}
        for name,data in assets.items():
            target=root/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        (root/'docs/MEDIA_CREDITS.md').write_text('Anna Tarazevich / Pexels; Tears of Steel / Blender Foundation; separate licenses')
        media=dict(license='MIT AND LicenseRef-Pexels AND CC-BY-3.0',sources=REVIEWED_MEDIA_SOURCES,
            credits_file='docs/MEDIA_CREDITS.md',changes='Synthetic test fixture, not film footage',
            demonstration='Test fixture',assets={})
        for name,data in assets.items():
            ids=list(REVIEWED_MEDIA_SOURCES) if name.endswith('.mp4') else ['kissing'] if name.endswith('.gif') else []
            media['assets'][name]=dict(sha256=sha(data),bytes=len(data),sources=ids,
                licenses=sorted({'MIT'}|{REVIEWED_MEDIA_SOURCES[key]['license'] for key in ids}))
        (root/'media-assets.json').write_text(json.dumps(media))
        spec=json.loads((root/'release-files.json').read_text())
        spec['files']+=list(assets)+['media-assets.json','docs/MEDIA_CREDITS.md']
        (root/'release-files.json').write_text(json.dumps(spec))
        return root,assets

    def test_reproducible_allowlist_ignores_private_files_and_substitutes_public_readme(self):
        with tempfile.TemporaryDirectory() as temp:
            work=Path(temp);root=self.fixture(work)
            (root/'private.mp4').write_bytes(b'not distributable')
            (root/'.runtime').mkdir();(root/'.runtime'/'report.json').write_text('private')
            (root/'unlisted.py').write_text('private code')
            first=build(root,work/'one');second=build(root,work/'two')
            self.assertEqual(first.read_bytes(),second.read_bytes())
            with zipfile.ZipFile(first) as bundle:
                names=bundle.namelist();prefix='adaptive-censor-0.3.0-preview.1/'
                self.assertEqual(bundle.read(prefix+'README.md'),b'Public introduction\n')
                self.assertFalse(any('private' in name or 'runtime' in name or 'unlisted' in name for name in names))
            self.assertEqual(verify(first)['files'],6)

    def test_missing_files_and_unsafe_names_fail_closed(self):
        for name in ('../file.py','/file.py','models/private.json','.runtime/report.json','clip.mp4','a\\b.py'):
            with self.assertRaises(ValueError):safe_name(name)
        with tempfile.TemporaryDirectory() as temp:
            root=self.fixture(Path(temp));(root/'module.py').unlink()
            with self.assertRaises((ValueError,FileNotFoundError)):build(root,Path(temp)/'out')

    def test_privacy_scanner_rejects_personal_paths_credentials_and_binary_content_without_echoing_them(self):
        samples=['C:'+chr(92)+'Users'+chr(92)+'Example'+chr(92)+'recording',
                 '/'+'Users'+'/'+'example'+'/'+'recording', 'ghp_'+'a'*30, 'hf_'+'b'*30, '\0binary']
        for sample in samples:
            with self.assertRaises(ValueError) as error:audit_text('example.txt',sample.encode())
            self.assertNotIn(sample,str(error.exception))

    def test_changed_archive_content_and_added_media_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            work=Path(temp);root=self.fixture(work);archive=build(root,work/'out')
            with zipfile.ZipFile(archive) as bundle:
                values={name:bundle.read(name) for name in bundle.namelist()}
            modified=work/'changed.zip'
            with zipfile.ZipFile(modified,'w') as bundle:
                for name,value in values.items():bundle.writestr(name,b'changed' if name.endswith('/module.py') else value)
            with self.assertRaisesRegex(ValueError,'Changed release content'):verify(modified)
            with zipfile.ZipFile(archive,'a') as bundle:
                bundle.writestr('adaptive-censor-0.3.0-preview.1/private.mp4',b'private')
            with self.assertRaises(ValueError):verify(archive)

    def test_broken_python_cannot_be_published_even_with_a_fresh_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            root=self.fixture(Path(temp))
            (root/'module.py').write_text("print('unfinished\n",encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Invalid Python source in module.py'):
                build(root,Path(temp)/'out')

    def test_symlink_archive_entry_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            work=Path(temp);archive=build(self.fixture(work),work/'out')
            with zipfile.ZipFile(archive) as bundle:values={name:bundle.read(name) for name in bundle.namelist()}
            changed=work/'link.zip'
            with zipfile.ZipFile(changed,'w') as bundle:
                for name,value in values.items():
                    info=zipfile.ZipInfo(name);info.create_system=3;info.external_attr=(0o120777 if name.endswith('/module.py') else 0o100644)<<16
                    bundle.writestr(info,value)
            with self.assertRaisesRegex(ValueError,'link or special file'):verify(changed)

    def test_reviewed_licensed_media_roundtrips_without_text_normalization(self):
        with tempfile.TemporaryDirectory() as temp:
            work=Path(temp);root,assets=self.licensed_fixture(work);archive=build(root,work/'out')
            self.assertEqual(verify(archive)['files'],11)
            with zipfile.ZipFile(archive) as bundle:
                for name,value in assets.items():
                    self.assertEqual(bundle.read('adaptive-censor-0.3.0-preview.1/'+name),value)
        for name in ('docs/media/private.mp4','docs/media/another.gif','clip.png'):
            with self.assertRaises(ValueError):safe_name(name)

    def test_media_requires_reviewed_hash_source_license_and_credits(self):
        with tempfile.TemporaryDirectory() as temp:
            work=Path(temp);root,assets=self.licensed_fixture(work)
            path=root/'media-assets.json';original=path.read_text();media=json.loads(original)
            media['license']='unknown';path.write_text(json.dumps(media))
            with self.assertRaisesRegex(ValueError,'licensing/provenance'):build(root,work/'out')
            path.write_text(original)
            target=root/next(iter(assets));target.write_bytes(target.read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError,'reviewed manifest'):build(root,work/'out')
            target.write_bytes(assets[next(iter(assets))])

            media=json.loads(original)
            media['assets']['docs/media/adaptive-censor-poster.png']['sources']=['kissing']
            path.write_text(json.dumps(media))
            with self.assertRaisesRegex(ValueError,'reviewed manifest'):build(root,work/'out')
            path.write_text(original)
            spec=json.loads((root/'release-files.json').read_text());spec['files'].remove('docs/MEDIA_CREDITS.md')
            (root/'release-files.json').write_text(json.dumps(spec))
            with self.assertRaisesRegex(ValueError,'credits and manifest'):build(root,work/'out')

    def test_binary_disguised_as_a_reviewed_asset_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            work=Path(temp);root,_=self.licensed_fixture(work)
            (root/'docs/media/adaptive-censor-demo.mp4').write_bytes(b'not an MP4')
            with self.assertRaisesRegex(ValueError,'Invalid or oversized'):build(root,work/'out')

    def test_each_source_and_each_asset_retains_its_own_license(self):
        with tempfile.TemporaryDirectory() as temp:
            work=Path(temp);root,_=self.licensed_fixture(work)
            path=root/'media-assets.json';original=path.read_text()
            for key,value in [('license','MIT'),('source_sha256','0'*64),
                              ('license_evidence','https://example.com/unreviewed')]:
                media=json.loads(original);media['sources']['film'][key]=value
                path.write_text(json.dumps(media))
                with self.assertRaisesRegex(ValueError,'licensing/provenance'):build(root,work/'out')
            media=json.loads(original);media['sources'].pop('film');path.write_text(json.dumps(media))
            with self.assertRaisesRegex(ValueError,'licensing/provenance'):build(root,work/'out')
            media=json.loads(original)
            media['assets']['docs/media/adaptive-censor-preview.gif']['licenses']=['CC-BY-3.0']
            path.write_text(json.dumps(media))
            with self.assertRaisesRegex(ValueError,'reviewed manifest'):build(root,work/'out')

    def test_public_tree_never_copies_private_history_and_refuses_overwrite(self):
        from prepare_public_repository import prepare
        with tempfile.TemporaryDirectory() as temp:
            work=Path(temp);root=self.fixture(work)
            (root/'.git').mkdir();(root/'.git/private-history').write_text('private history')
            (root/'private.mp4').write_bytes(b'private movie')
            archive=build(root,work/'out');target=work/'public'
            result=prepare(archive,target)
            self.assertTrue(result['clean_history']);self.assertFalse(result['published'])
            self.assertFalse((target/'.git').exists());self.assertFalse((target/'private.mp4').exists())
            self.assertEqual((target/'README.md').read_text(),'Public introduction\n')
            with self.assertRaises(FileExistsError):prepare(archive,target)


if __name__=='__main__':unittest.main()
