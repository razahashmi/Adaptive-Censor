"""Create a new public source tree from the verified archive; never copy old Git history."""
import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile

from build_source_release import verify


def prepare(archive,destination,init_git=False):
    archive=Path(archive).resolve();destination=Path(destination).absolute()
    identity=verify(archive)
    if destination.exists():raise FileExistsError('Choose a new directory; never overwrite an existing checkout')
    if not destination.parent.is_dir():raise ValueError('The destination parent must exist')
    with tempfile.TemporaryDirectory(prefix='adaptive-public-') as temporary:
        staging=Path(temporary)
        with zipfile.ZipFile(archive) as bundle:bundle.extractall(staging)
        source=staging/('adaptive-censor-'+identity['version'])
        if (source/'.git').exists():raise ValueError('Historical repository metadata is forbidden')
        shutil.copytree(source,destination)
    if init_git:
        subprocess.run(['git','init','--initial-branch=main',str(destination)],check=True,capture_output=True)
    return dict(identity,clean_history=True,repository_initialized=init_git,published=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('archive',type=Path)
    p.add_argument('--destination',type=Path,required=True);p.add_argument('--init-git',action='store_true')
    a=p.parse_args();print(prepare(a.archive,a.destination,a.init_git))
