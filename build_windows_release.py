"""Build a small native Windows setup shell plus its verified source archive.

The executable is original MIT C# code using the Windows .NET Framework. It
bundles no Python, Tcl/Tk, inference library, codec or model runtime binaries.
An existing 64-bit Python 3.13 is required for the CPU installation it manages.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile
from build_source_release import verify

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def build(archive,output):
    archive=Path(archive).resolve();output=Path(output).resolve();root=Path(__file__).parent
    identity=verify(archive)
    if sys.platform!='win32':raise RuntimeError('Build the native Windows shell on Windows')
    compiler=Path(os.environ.get('WINDIR','C:/Windows'))/'Microsoft.NET/Framework64/v4.0.30319/csc.exe'
    if not compiler.is_file():raise RuntimeError('The Windows .NET Framework compiler is unavailable')
    output.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='adaptive-native-setup-') as temporary:
        work=Path(temporary);info=work/'setup-build-info.json'
        info.write_text(json.dumps(dict(archive=archive.name,sha256=identity['sha256'],version=identity['version'])),encoding='utf-8')
        exe=work/'Adaptive Censor Setup.exe'
        command=[str(compiler),'/nologo','/target:winexe','/platform:x64','/optimize+',
                 '/r:System.Windows.Forms.dll','/r:System.Drawing.dll','/r:System.Web.Extensions.dll','/out:'+str(exe),
                 '/resource:'+str(info)+',setup-build-info.json',
                 '/resource:'+str(root/'windows_bootstrap.py')+',windows_bootstrap.py',
                 '/resource:'+str(root/'build_source_release.py')+',build_source_release.py',str(root/'windows_setup.cs')]
        subprocess.run(command,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        readme=work/'START_HERE.txt'
        readme.write_text('''Extract the complete ZIP, then open Adaptive Censor Setup.exe.
Requires existing 64-bit Python 3.13 and Windows .NET Framework 4.x.
The source ZIP is checked against the checksum embedded in this build.
Start with the controls demo; dependencies and optional models download separately.
No Python, inference runtime, codecs or model weights are bundled in the setup executable.
The launcher is original MIT code. Media in the source ZIP retain their own terms.
Local setup testing is not independent clean-Windows validation. This build is unsigned.
https://github.com/razahashmi/Adaptive-Censor
''',encoding='utf-8')
        receipt=dict(version=identity['version'],source_sha256=identity['sha256'],exe_sha256=sha(exe),build_tool='Windows .NET Framework C# compiler',
                     signed=False,independent_clean_windows_validated=False,bundled_third_party_runtime=False)
        receipt_file=work/'WINDOWS_BUILD.json';receipt_file.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
        target=output/f"adaptive-censor-{identity['version']}-windows-setup.zip"
        with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
            for path in (exe,archive,readme,receipt_file,root/'LICENSE'):z.write(path,path.name)
        (output/'WINDOWS_SHA256SUMS').write_text(sha(target)+'  '+target.name+'\n',encoding='utf-8')
        return target

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('archive',type=Path);p.add_argument('--output',type=Path,default=Path(__file__).parent/'dist')
    a=p.parse_args();print(build(a.archive,a.output))
