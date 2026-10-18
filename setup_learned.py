"""Explicitly install pinned training snapshots. Playback never calls this downloader."""
import argparse
from features import REVISIONS
from model_setup import prepare_environment


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--download', action='store_true', help='Explicit permission to download the pinned HF snapshots')
    args=p.parse_args()
    if not args.download:
        raise SystemExit('Use --download to install models explicitly; playback is local-only.')
    from huggingface_hub import snapshot_download
    root=prepare_environment()
    for model, revision in REVISIONS.items():
        snapshot_download(model, revision=revision, cache_dir=root/'hf')
    print('Pinned snapshots installed. Export with the training environment, then run export_learned.py check on reference clips.')


if __name__ == '__main__':
    main()
