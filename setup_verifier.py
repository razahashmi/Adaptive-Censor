"""Explicitly install pinned, official local vision-verifier weights and runtime."""
import hashlib
import argparse
import json
from pathlib import Path
import platform
import time
import urllib.request
import zipfile

from model_setup import prepare_environment

REPO = 'Qwen/Qwen3-VL-4B-Instruct-GGUF'
REVISION = '1cd86afb9a95c410a6038ab3b40d8b578c892266'
MODEL = 'Qwen3VL-4B-Instruct-Q4_K_M.gguf'
PROJECTOR = 'mmproj-Qwen3VL-4B-Instruct-F16.gguf'
BUILD = 'b11146'
WEIGHTS = {
    MODEL: '66358cb18bb6b3b1b6675aa412c7a88ef01d228f481184d13668e5201c730a0a',
    PROJECTOR: '256f3a43bd4205ffef48d6b92715e1e70b5b0e9aef06522584967513a9985331',
}
MODEL_SPECS = {
    '4B': dict(repo=REPO,revision=REVISION,model=MODEL,projector=PROJECTOR,weights=WEIGHTS),
    '8B': dict(repo='Qwen/Qwen3-VL-8B-Instruct-GGUF',revision='f982a07559d4a2f6c8744d840bf6fccab30eea96',
        model='Qwen3VL-8B-Instruct-Q4_K_M.gguf',projector='mmproj-Qwen3VL-8B-Instruct-F16.gguf',
        weights={'Qwen3VL-8B-Instruct-Q4_K_M.gguf':'67d1659bfe71b89d50b45a4ad1a9e5b997e5bb16ce5da66a6a6167abd569e9e2',
                 'mmproj-Qwen3VL-8B-Instruct-F16.gguf':'ca524100ebf825c9a870db1c580d03879e0da0ab2541697e2458e64891cf9d38'}),
}
ANATOMY_SHA256 = '04fe3d77980780c1f8297dc6d7f942fd5b3abe6942a188f742a85241e4f634eb'
ARCHIVES = {
    'llama-b11146-bin-win-cuda-13.4-x64.zip': 'b1866c0ce76bc7bfb0c24b33e9a37e9669f1be18539b12c74ce361f81c41f047',
    'cudart-llama-bin-win-cuda-13.4-x64.zip': '738f8c251ac22b70c3ae6f83a10cf222725df0395246a2cf58f32bdb85fbe668',
}


def checksum(path):
    if not path.is_file():
        return None
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def download(url, destination, expected, headers=None):
    if checksum(destination) == expected:
        return
    print('Downloading:', destination.name, flush=True)
    temporary = destination.with_suffix(destination.suffix + '.partial')
    request=urllib.request.Request(url,headers=headers or {})
    with urllib.request.urlopen(request,timeout=120) as response, temporary.open('wb') as output:
        if 'text/html' in response.headers.get('Content-Type',''):
            raise RuntimeError(f'Download returned HTML instead of {destination.name}')
        while chunk:=response.read(1024*1024):
            output.write(chunk)
    if checksum(temporary) != expected:
        raise RuntimeError(f'Checksum mismatch for {destination.name}; installation not accepted')
    temporary.replace(destination)


def install(model_size='4B'):
    if platform.system() != 'Windows' or platform.machine().lower() not in ('amd64','x86_64'):
        raise RuntimeError('This installer currently supports Windows x64 with NVIDIA GPU only.')
    root=prepare_environment()/'scene-verifier'
    spec=MODEL_SPECS[model_size]
    root.mkdir(exist_ok=True)
    for name,digest in ARCHIVES.items():
        archive=root/name
        download(f'https://github.com/ggml-org/llama.cpp/releases/download/{BUILD}/{name}', archive, digest)
        destination=(root/'llama').resolve()
        destination.mkdir(exist_ok=True)
        with zipfile.ZipFile(archive) as bundle:
            for item in bundle.infolist():
                if not (destination/item.filename).resolve().is_relative_to(destination):
                    raise RuntimeError('Invalid archive path')
            bundle.extractall(destination)
    (root/'models').mkdir(exist_ok=True)
    for name, digest in spec['weights'].items():
        download(f"https://huggingface.co/{spec['repo']}/resolve/{spec['revision']}/{name}", root/'models'/name, digest)
    download(f'https://api.github.com/repos/notAI-tech/NudeNet/releases/assets/176832019?download=1&t={int(time.time())}',
             root/'640m.onnx', ANATOMY_SHA256, {'Accept':'application/octet-stream'})
    import onnxruntime as ort
    options=ort.SessionOptions();options.intra_op_num_threads=2
    ort.InferenceSession(str(root/'640m.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
    (root/f'installed-{model_size}.json').write_text(json.dumps(dict(**spec,build=BUILD,archives=ARCHIVES,
        anatomy_sha256=ANATOMY_SHA256),indent=2),encoding='utf-8')
    print('Local verifier installed:',root,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=list(MODEL_SPECS),default='4B')
    install(parser.parse_args().model)
