"""Small local installation check. No media reading, downloads or inference."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile

from model_setup import runtime_dir, SPEECH_REVISIONS


def writable(root):
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile(dir=root) as probe:
        probe.write(b'readiness');probe.flush()


def inspect(root=None, *, version=importlib.metadata.version, storage_probe=writable):
    root=Path(root) if root is not None else runtime_dir()
    checks=[]
    def add(key,label,status,detail,action=''):
        checks.append(dict(id=key,label=label,status=status,detail=detail,action=action))
    add('python','Python','ok' if sys.version_info[:2]==(3,13) else 'warning',
        f'{sys.version.split()[0]}. Windows/Python 3.13 is the tested CPU configuration.',
        '' if sys.version_info[:2]==(3,13) else 'Use Python 3.13 for the tested setup.')
    missing=[];different=[];gpu_ort=False
    ort_packages={}
    for name in ('onnxruntime','onnxruntime-gpu'):
        try:ort_packages[name]=version(name)
        except importlib.metadata.PackageNotFoundError:pass
    for line in (Path(__file__).parent/'requirements.txt').read_text(encoding='utf-8').splitlines():
        match=re.fullmatch(r'([A-Za-z0-9_.-]+)==([^\s]+)',line.strip())
        if not match:continue
        name,expected=match.groups()
        try:actual=version(name)
        except importlib.metadata.PackageNotFoundError:
            if name=='onnxruntime':
                try:actual=version('onnxruntime-gpu');gpu_ort=True
                except importlib.metadata.PackageNotFoundError:missing.append(name);continue
            else:missing.append(name);continue
        if actual!=expected:different.append(f'{name} {actual} (expected {expected})')
    add('dependencies','Runtime dependencies','error' if missing else 'warning' if different else 'ok',
        ('Missing: '+', '.join(missing)) if missing else ('Versions differ from the preview: '+', '.join(different)) if different
        else 'Pinned direct dependencies are installed'+(' (GPU ONNX Runtime supplies the runtime module).' if gpu_ort else '.'),
        'Run Setup.ps1 again for your selected CPU/GPU configuration.' if missing or different else '')
    if len(ort_packages)>1:
        add('runtime_overlap','Inference package conflict','warning',
            'Both CPU and GPU ONNX Runtime packages are installed. They share files, so package versions may not describe the loaded module. '+
            'Installed: '+', '.join(f'{name} {actual}' for name,actual in ort_packages.items())+'.',
            'Stop the service, then run Setup.ps1 for CPU or Setup.ps1 -Gpu for NVIDIA. Setup removes the overlapping packages before reinstalling the selected runtime.')
    try:
        import imageio_ffmpeg
        ffmpeg=Path(imageio_ffmpeg.get_ffmpeg_exe())
        if not ffmpeg.is_file():raise OSError('FFmpeg missing')
        add('ffmpeg','Video import and export','ok','FFmpeg executable is available. Generated-media tests verify encoding separately.')
    except (ImportError,OSError,RuntimeError):
        add('ffmpeg','Video import and export','error','FFmpeg is unavailable.','Reinstall the pinned runtime dependencies with Setup.ps1.')
    free=None
    try:
        storage_probe(root);free=shutil.disk_usage(root).free
        add('storage','Local storage','warning' if free<1024**3 else 'ok',
            f'{free/1024**3:.1f} GiB available. Runtime folder accepts writes.',
            'Free space before importing or exporting large videos.' if free<1024**3 else '')
    except OSError:
        add('storage','Local storage','error','The runtime folder cannot be written.','Choose a writable ADAPTIVE_CENSOR_WORKDIR and check free space.')
    speech=root/'models'/'models--Systran--faster-whisper-tiny.en'/'snapshots'/SPEECH_REVISIONS['tiny.en']
    speech_ready=all((speech/name).is_file() and (speech/name).stat().st_size for name in ('model.bin','config.json','tokenizer.json'))
    clip_ready=all((root/'clip'/name).is_file() and (root/'clip'/name).stat().st_size
                   for name in ('onnx/vision_model_quantized.onnx','onnx/text_model_quantized.onnx','tokenizer.json'))
    for key,label,present in [('speech','Base English speech model',speech_ready),('clip','Base visual similarity model',clip_ready)]:
        add(key,label,'ok' if present else 'warning','Files are installed; this check does not run the model.' if present
            else 'Separate model files are missing. The generated controls demo can still run.',
            '' if present else 'Run Setup.ps1 to download the pinned base models.')
    providers=[]
    try:
        import onnxruntime
        providers=onnxruntime.get_available_providers()
        add('compute','Compute options','ok','Loaded ONNX Runtime '+onnxruntime.__version__+'. Installed ONNX providers: '+', '.join(providers)+'. GPU execution is not tested by this check.')
    except (ImportError,OSError,RuntimeError):
        add('compute','Compute options','error','The inference runtime cannot be loaded.','Run Setup.ps1 for CPU, or Setup.ps1 -Gpu for optional NVIDIA support.')
    blocked=any(row['status']=='error' for row in checks)
    return dict(status='blocked' if blocked else 'needs_setup' if not speech_ready or not clip_ready else 'ready',
                demo_ready=not blocked, checks=checks, free_bytes=free,
                inference_performed=False, downloads_performed=False,
                note='Installed files and completed analysis do not establish detection accuracy. Optional scene/learned models have their own setup.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--runtime',type=Path)
    args=parser.parse_args();print(json.dumps(inspect(args.runtime),indent=2))
