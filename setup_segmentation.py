"""Explicit pinned download for optional local mask drafting; never at playback."""
from pathlib import Path
import argparse
from labeling import read_json, write_json
from model_setup import prepare_environment
from provenance import file_hash

REPO='facebook/sam2.1-hiera-tiny'
REVISION='de431c4043854a71d8101e17995dfe596bf101a5'
WEIGHTS_SHA256='48c14467e5cf9e51870511feb72c89688e82dd74523142c0538b663e193ac2a7'
FILES=('config.json','preprocessor_config.json','model.safetensors')


def check(folder):
    manifest=read_json(folder/'installation.json',{})
    if manifest.get('repo')!=REPO or manifest.get('revision')!=REVISION:
        raise ValueError('Install the pinned segmentation model explicitly first')
    if set(manifest.get('files',{}))!=set(FILES):raise ValueError('Incomplete segmentation installation')
    for name in FILES:
        if file_hash(folder/name)!=manifest['files'][name]:raise ValueError('Segmentation model file changed: '+name)
    if manifest['files']['model.safetensors']!=WEIGHTS_SHA256:raise ValueError('Unexpected segmentation weights')
    return manifest


def install(root):
    from huggingface_hub import snapshot_download
    folder=root/'segmentation-runtime/sam2.1-tiny'
    snapshot_download(REPO,revision=REVISION,allow_patterns=list(FILES),local_dir=str(folder),max_workers=2)
    identities={name:file_hash(folder/name) for name in FILES}
    if identities['model.safetensors']!=WEIGHTS_SHA256:raise ValueError('Segmentation checkpoint checksum mismatch')
    write_json(folder/'installation.json',dict(repo=REPO,revision=REVISION,files=identities,
               use='Offline draft masks, not a semantic content classifier or accepted censorship'))
    check(folder)
    print('Pinned SAM 2.1 tiny installed and checksum verified (156 MB weights)',flush=True)


if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    install(prepare_environment())
