"""Build and audit the explicitly allowlisted public source archive. No Git required."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile

ROOT = Path(__file__).resolve().parent
DENIED = {'.git', '.runtime', '.venv', '.train-venv', '.ci-venv', '.release-env', '.claude',
          '__pycache__', 'benchmarks', 'dist', 'feedback', 'models', 'sessions-v2', 'training-runs'}
EXTENSIONS = {'.py', '.md', '.json', '.mjs', '.css', '.html', '.yml', '.yaml', '.txt', '.ps1', '.cs'}
SPECIAL = {'LICENSE', '.gitignore', '.gitattributes', 'docs/media/adaptive-censor-demo.srt', 'docs/media/adaptive-censor-demo.vtt'}
MEDIA = {'docs/media/adaptive-censor-demo.mp4': (b'ftyp',18*1024**2),
         'docs/media/adaptive-censor-preview.gif': (b'GIF',3*1024**2),
         'docs/media/adaptive-censor-poster.png': (b'\x89PNG\r\n\x1a\n',1024**2)}
REVIEWED_MEDIA_SOURCES = {
    'kissing': dict(title='Close Up Video of People Kissing', attribution='Anna Tarazevich / Pexels',
        source_url='https://www.pexels.com/video/close-up-video-of-people-kissing-5763324/',
        source_download_url='https://videos.pexels.com/video-files/5763324/5763324-hd_1920_1080_24fps.mp4',
        source_sha256='f937d4aeb4e02ac15330f5ff3d10c81089844f4d5acc6dbc3f2d197f16530666',
        license='LicenseRef-Pexels',license_url='https://www.pexels.com/license/',
        license_evidence='https://www.pexels.com/license/',source_intervals_seconds=[[0,6]]),
    'film': dict(title='Tears of Steel (2012)', attribution='(CC) Blender Foundation | mango.blender.org',
        source_url='https://mango.blender.org/',
        source_download_url='https://download.blender.org/demo/movies/ToS/tears_of_steel_720p.mov',
        source_sha256='efa9062d9cdb7a338e40ad530dfdf234806743f29ae6a1a136b97ece4e588e8f',
        license='CC-BY-3.0',license_url='https://creativecommons.org/licenses/by/3.0/',
        license_evidence='https://mango.blender.org/sharing/',source_intervals_seconds=[[57,61],[357.8,360.8]])}
PRIVATE = [
    re.compile(r'(?i)\b[a-z]:[\\/]+Users[\\/]+[^\s"\']+'),
    re.compile(r'/Users/[A-Za-z0-9_. -]+/'),
    re.compile(r'(?i)\b[fgh]:[\\/]+(?:Movies|Seasons)\b'),
    re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}|hf_[A-Za-z0-9]{20,})\b'),
    re.compile(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
    re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def safe_name(value):
    if not isinstance(value, str) or '\\' in value or ':' in value:
        raise ValueError('Release paths must be relative POSIX names')
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(p in ('.', '..') or p in DENIED for p in path.parts):
        raise ValueError(f'Excluded release path: {value}')
    if path.as_posix() != value or (value not in SPECIAL and value not in MEDIA and path.suffix not in EXTENSIONS):
        raise ValueError(f'Unapproved release file type/path: {value}')
    return value


def validate_spec(spec):
    version = spec['version']
    if not re.fullmatch(r'\d+\.\d+\.\d+-preview\.\d+', version):
        raise ValueError('Use an explicit preview version')
    files = spec['files']
    if not isinstance(files, list) or not files or len(files) != len(set(files)):
        raise ValueError('Release file list is empty or duplicated')
    for name in files:
        safe_name(name)
    overrides = spec.get('overrides', {})
    if not set(overrides).issubset(files):
        raise ValueError('Every override target must be in the release list')
    for name in overrides.values():
        safe_name(name)
    return spec


def audit_text(name, data):
    text = data.decode('utf-8-sig')
    if '\0' in text or any(pattern.search(text) for pattern in PRIVATE):
        # Never echo the matched credential or personal path.
        raise ValueError(f'Privacy audit failed for {name}; inspect it locally')
    if name.endswith('.py'):
        try:
            ast.parse(text, filename=name)
        except SyntaxError as error:
            raise ValueError(f'Invalid Python source in {name} at line {error.lineno}') from None


def audit_content(name,data):
    if name not in MEDIA:
        audit_text(name,data)
        return
    signature,limit=MEDIA[name]
    header=data[4:8] if name.endswith('.mp4') else data[:len(signature)]
    if header!=signature or len(data)>limit:
        raise ValueError('Invalid or oversized reviewed media asset: '+name)


def audit_media_manifest(data):
    """A narrow exception for three reviewed licensed assets, never arbitrary media."""
    included=set(data)&set(MEDIA)
    if not included:return
    if included!=set(MEDIA) or 'media-assets.json' not in data or 'docs/MEDIA_CREDITS.md' not in data:
        raise ValueError('Licensed media requires its complete asset set, credits and manifest')
    media=json.loads(data['media-assets.json'])
    required=dict(license='MIT AND LicenseRef-Pexels AND CC-BY-3.0',credits_file='docs/MEDIA_CREDITS.md',
                  sources=REVIEWED_MEDIA_SOURCES)
    if (any(media.get(k)!=v for k,v in required.items()) or set(media.get('assets',{}))!=included
            or not media.get('changes') or not media.get('demonstration')):
        raise ValueError('Missing or unreviewed media licensing/provenance')
    for name in included:
        source_ids=list(REVIEWED_MEDIA_SOURCES) if name.endswith('.mp4') else ['kissing'] if name.endswith('.gif') else []
        licenses=sorted({'MIT'}|{REVIEWED_MEDIA_SOURCES[key]['license'] for key in source_ids})
        if media['assets'][name]!=dict(sha256=sha(data[name]),bytes=len(data[name]),
                                       sources=source_ids,licenses=licenses):
            raise ValueError('Media asset differs from the reviewed manifest: '+name)


def source_bytes(root, name):
    safe_name(name)
    root = root.resolve()
    current = root
    for part in PurePosixPath(name).parts:
        current = current / part
        attributes = getattr(current.lstat(), 'st_file_attributes', 0)
        if current.is_symlink() or attributes & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0):
            raise ValueError(f'Linked publication input: {name}')
    if not current.resolve().is_relative_to(root) or not current.is_file():
        raise ValueError(f'Missing or external publication input: {name}')
    data = current.read_bytes()
    audit_content(name, data)
    return data if name in MEDIA else data.replace(b'\r\n', b'\n')


def build(root, output):
    root, output = Path(root), Path(output)
    spec = validate_spec(json.loads(source_bytes(root, 'release-files.json')))
    data = {name:source_bytes(root, spec.get('overrides', {}).get(name, name)) for name in sorted(spec['files'])}
    for name,value in data.items():audit_content(name,value)
    audit_media_manifest(data)
    manifest = dict(schema_version=1, version=spec['version'],
                    distribution='source-preview-with-licensed-demo' if set(data)&set(MEDIA) else 'source-only-preview',
                    files={name:dict(sha256=sha(value), bytes=len(value)) for name,value in data.items()})
    data['RELEASE_MANIFEST.json'] = (json.dumps(manifest, sort_keys=True, indent=2)+'\n').encode()
    output.mkdir(parents=True, exist_ok=True)
    archive = output / f"adaptive-censor-{spec['version']}-source.zip"
    temporary = archive.with_suffix('.zip.tmp')
    prefix = f"adaptive-censor-{spec['version']}/"
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
            for name, value in sorted(data.items()):
                info = zipfile.ZipInfo(prefix+name, date_time=(2026, 10, 15, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                bundle.writestr(info, value)
        verify(temporary)
        temporary.replace(archive)
    finally:
        temporary.unlink(missing_ok=True)
    (output/'SHA256SUMS').write_text(f'{sha(archive.read_bytes())}  {archive.name}\n', encoding='utf-8')
    return archive


def verify(archive):
    with zipfile.ZipFile(archive) as bundle:
        infos = bundle.infolist()
        names = [info.filename for info in infos]
        if not names or len(names) != len(set(names)) or len(names)>500:
            raise ValueError('Empty, duplicate or oversized release file list')
        prefix = names[0].split('/')[0]+'/'
        if sum(info.file_size for info in infos)>30*1024**2:
            raise ValueError('Source release exceeds its size budget')
        for info in infos:
            if not info.filename.startswith(prefix) or info.is_dir():
                raise ValueError('Release must have one source directory and regular files')
            name = safe_name(info.filename[len(prefix):])
            if stat.S_IFMT(info.external_attr>>16) not in (0, stat.S_IFREG):
                raise ValueError('Archive contains a link or special file')
            audit_content(name, bundle.read(info))
        spec = validate_spec(json.loads(bundle.read(prefix+'release-files.json')))
        if prefix != f"adaptive-censor-{spec['version']}/":
            raise ValueError('Version and archive directory do not match')
        expected = set(spec['files']) | {'RELEASE_MANIFEST.json'}
        if set(names) != {prefix+name for name in expected}:
            raise ValueError('Archive contains unlisted or missing files')
        manifest = json.loads(bundle.read(prefix+'RELEASE_MANIFEST.json'))
        if manifest['version'] != spec['version'] or set(manifest['files']) != set(spec['files']):
            raise ValueError('Invalid file manifest')
        for name, identity in manifest['files'].items():
            value = bundle.read(prefix+name)
            if identity != dict(sha256=sha(value), bytes=len(value)):
                raise ValueError(f'Changed release content: {name}')
        audit_media_manifest({name:bundle.read(prefix+name) for name in spec['files']})
        return dict(version=spec['version'], files=len(expected), sha256=sha(Path(archive).read_bytes()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'dist')
    parser.add_argument('--verify', type=Path)
    args=parser.parse_args()
    if args.verify:
        print(json.dumps(verify(args.verify), indent=2))
    else:
        path=build(ROOT, args.output)
        print(path)
        print(json.dumps(verify(path), indent=2))
