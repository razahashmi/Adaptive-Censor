import importlib.metadata
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from model_setup import SPEECH_REVISIONS
from readiness import inspect


class ReadinessTests(unittest.TestCase):
    def version(self,name):
        import re
        for line in Path(__file__).with_name('requirements.txt').read_text().splitlines():
            if line.startswith(name+'=='):return line.split('==')[1]
        raise importlib.metadata.PackageNotFoundError(name)

    def test_missing_models_leave_generated_demo_usable_without_downloads_or_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            result=inspect(Path(temporary),version=self.version)
            self.assertTrue(result['demo_ready']);self.assertEqual(result['status'],'needs_setup')
            self.assertFalse(result['inference_performed']);self.assertFalse(result['downloads_performed'])

    def test_missing_dependency_and_unwritable_storage_are_actionable_without_private_paths(self):
        def missing(name):raise importlib.metadata.PackageNotFoundError(name)
        def blocked(root):raise PermissionError('private diagnostic')
        with tempfile.TemporaryDirectory() as temporary:
            result=inspect(Path(temporary),version=missing,storage_probe=blocked)
            self.assertEqual(result['status'],'blocked');self.assertFalse(result['demo_ready'])
            storage=next(row for row in result['checks'] if row['id']=='storage')
            self.assertEqual(storage['status'],'error');self.assertNotIn(temporary,str(result));self.assertNotIn('private diagnostic',str(result))

    def test_file_presence_is_separate_from_model_execution_and_gpu_package_can_supply_ort(self):
        def gpu(name):
            if name=='onnxruntime':raise importlib.metadata.PackageNotFoundError(name)
            return self.version('onnxruntime' if name=='onnxruntime-gpu' else name)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            speech=root/'models'/'models--Systran--faster-whisper-tiny.en'/'snapshots'/SPEECH_REVISIONS['tiny.en']
            for folder,names in [(speech,['model.bin','config.json','tokenizer.json']),
                                (root/'clip',['onnx/vision_model_quantized.onnx','onnx/text_model_quantized.onnx','tokenizer.json'])]:
                for name in names:
                    path=folder/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'fixture')
            result=inspect(root,version=gpu)
            self.assertEqual(result['status'],'ready');self.assertFalse(result['inference_performed'])
            self.assertIn('GPU ONNX Runtime',next(row for row in result['checks'] if row['id']=='dependencies')['detail'])

    def test_overlapping_runtime_packages_are_reported_with_versions_and_repair_choice(self):
        def overlap(name):
            if name=='onnxruntime':return '1.30.0'
            if name=='onnxruntime-gpu':return '1.23.2'
            return self.version(name)
        with tempfile.TemporaryDirectory() as temporary:
            result=inspect(Path(temporary),version=overlap)
            conflict=next(row for row in result['checks'] if row['id']=='runtime_overlap')
            self.assertEqual(conflict['status'],'warning')
            self.assertIn('onnxruntime 1.30.0',conflict['detail'])
            self.assertIn('onnxruntime-gpu 1.23.2',conflict['detail'])
            self.assertIn('Setup.ps1 -Gpu',conflict['action'])
            dependency=next(row for row in result['checks'] if row['id']=='dependencies')
            self.assertIn('expected 1.23.2',dependency['detail'])


if __name__=='__main__':unittest.main()
