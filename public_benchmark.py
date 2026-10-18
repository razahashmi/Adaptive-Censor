"""Prepare licensed public examples and measure the exact incremental player passes.

The bundled reference is assistant-reviewed development data, not independent
human ground truth. Scoring drafts requires an explicit --allow-draft flag.
No private library, local labels or trained heads are read by this tool.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import threading
import time
import zipfile

ROOT=Path(__file__).resolve().parent
REPO='https://github.com/razahashmi/Adaptive-Censor'

def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def identity(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')

def validate_suite(suite):
    from core import CATEGORIES
    if suite.get('schema_version')!=1 or suite.get('split')!='development':
        raise ValueError('Use the documented public development suite')
    ids=set()
    for case in suite['cases']:
        if case['id'] in ids or not case['id'].replace('-','').isalnum():raise ValueError('Unsafe or duplicate case ID')
        ids.add(case['id'])
        if case['source'] not in suite['sources'] or not all(math.isfinite(t) for t in (case['start'],case['end'])) or not 0<=case['start']<case['end']:raise ValueError('Invalid source interval')
        if any(c not in CATEGORIES for c in case['reviewed']):raise ValueError('Unknown reviewed category')
        if any(c not in case['reviewed'] for c in case['events']):raise ValueError('Gold events need explicit reviewed windows')
        for spans in case['events'].values():
            if any(not isinstance(span,list) or len(span)!=2 or not all(math.isfinite(t) for t in span) or not 0<=span[0]<span[1]<=case['end']-case['start'] for span in spans):raise ValueError('Gold interval outside its clip')
    for source in suite['sources'].values():
        if len(source['sha256'])!=64 or not source['license'] or not source['attribution']:raise ValueError('Source identity and terms required')
    return suite

def prepare(source_dir,destination):
    from core import run_ffmpeg
    suite=validate_suite(read(ROOT/'samples/public-suite.json'))
    destination=Path(destination)
    if destination.exists() and any(destination.iterdir()):raise ValueError('Choose an empty example-pack destination')
    source_dir=Path(source_dir)
    sources={key:source_dir/row['filename'] for key,row in suite['sources'].items()}
    for key,path in sources.items():
        if not path.is_file() or identity(path)!=suite['sources'][key]['sha256']:raise ValueError('Missing or changed licensed source: '+key)
    destination.mkdir(parents=True,exist_ok=True)
    for case in suite['cases']:
        run_ffmpeg(['-y','-ss',str(case['start']),'-i',str(sources[case['source']]),'-t',str(case['end']-case['start']),
                    '-map','0:v:0','-map','0:a:0?','-vf','scale=1280:-2,fps=24','-c:v','libx264','-threads','2','-preset','veryfast',
                    '-crf','20','-pix_fmt','yuv420p','-c:a','aac','-b:a','128k','-map_metadata','-1',str(destination/(case['id']+'.mp4'))])
    write(destination/'suite.json',suite)
    (destination/'CREDITS.md').write_text((ROOT/'samples/README.md').read_text(encoding='utf-8'),encoding='utf-8')
    write(destination/'FILES.json',{p.name:dict(sha256=identity(p),bytes=p.stat().st_size) for p in destination.iterdir() if p.is_file()})
    return destination

def check_pack(folder,suite):
    manifest=read(Path(folder)/'FILES.json')
    for case in suite['cases']:
        path=Path(folder)/(case['id']+'.mp4');expected=manifest.get(path.name,{})
        if not path.is_file() or identity(path)!=expected.get('sha256'):raise ValueError('Example clip differs from its pack manifest: '+case['id'])

def run(folder,output,mode='speech',allow_draft=False):
    import psutil
    from analysis_engine import Models,ScanConfig,CHUNK,media_info,ANALYSIS_REVISION
    from core import CATEGORIES,default_policy
    from evaluate import evaluate
    from provenance import pipeline_identity
    folder=Path(folder);suite=validate_suite(read(folder/'suite.json'));check_pack(folder,suite)
    human=suite['review'].get('status')=='human_reviewed' and bool(suite['review'].get('reviewer'))
    if not human and not allow_draft:raise ValueError('Reference labels are drafts. Review them first or explicitly use --allow-draft for a diagnostic run.')
    config=ScanConfig(speech=True,nudity=mode=='cpu-review',scenes=mode=='cpu-review',scene_mode='review',device='cpu',
                      phrases=['dick'],allow_phrases=[],speech_model='small.en',fps=2)
    peak=[psutil.Process().memory_info().rss];done=threading.Event()
    def monitor():
        while not done.wait(.1):peak[0]=max(peak[0],psutil.Process().memory_info().rss)
    watcher=threading.Thread(target=monitor,daemon=True);watcher.start()
    started=time.perf_counter();models=Models(workers=1);results=[];chunks=[]
    try:
        for case in suite['cases']:
            if mode=='speech' and 'language' not in case['reviewed']:continue
            path=folder/(case['id']+'.mp4');info=media_info(path);events=[];begin=time.perf_counter()
            for index in range(int(__import__('math').ceil(info['duration']/CHUNK))):
                before=time.perf_counter()
                events+=models.analyze(path,index*CHUNK,min((index+1)*CHUNK,info['duration']),info['duration'],config,info['has_audio'])
                chunks.append(time.perf_counter()-before)
            duration=info['duration'];policy=default_policy();gold=[]
            reviewed={c:[[0,duration]] for c in case['reviewed'] if mode!='speech' or c=='language'}
            for category,spans in case['events'].items():
                if category not in reviewed:continue
                for a,b in spans:
                    gold.append(dict(start=a,end=b,category=category,confidence=1,severity=2,enabled=True,source='manual'))
            for c in CATEGORIES:policy['categories'][c]['visual']='hide' if c in reviewed and c!='language' else 'allow'
            metrics=evaluate(events,gold,duration,policy,reviewed)['by_category']
            results.append(dict(id=case['id'],source=case['source'],source_sha256=identity(path),media_seconds=duration,
                wall_seconds=time.perf_counter()-begin,events=events,metrics=metrics))
            print('Measured',case['id'],flush=True)
    finally:done.set();watcher.join()
    elapsed=time.perf_counter()-started;by_category={}
    for c in CATEGORIES:
        rows=[r['metrics'][c] for r in results if r['metrics'][c]['reviewed_seconds']>0]
        sums={key:sum(row[key] for row in rows) for key in ('reviewed_seconds','gold_seconds','gold_events','filter_seconds','matched_seconds','missed_seconds','unnecessary_filter_seconds','events_at_least_95_percent_covered')}
        matched=sums['matched_seconds'];filtered=sums['filter_seconds'];gold=sums['gold_seconds'];clean=sums['reviewed_seconds']-gold
        by_category[c]=dict(**sums,time_precision=matched/filtered if filtered else None,time_recall=matched/gold if gold else None,
            false_filter_seconds_per_clean_hour=sums['unnecessary_filter_seconds']/clean*3600 if clean else None,
            title_support=len({r['source'] for r in results if r['metrics'][c]['reviewed_seconds']>0}),
            pixel_coverage=None,pixel_coverage_reason='No independently reviewed pixel masks in this starter pack',
            evidence='diagnostic_draft' if rows and not human else 'human_reviewed_development' if rows else 'not_represented')
    import numpy as np
    from provenance import dependency_identity
    media=sum(r['media_seconds'] for r in results)
    pipeline=pipeline_identity(models.root,config,1)
    pipeline['assets']={('installed-package/'+Path(key).name if Path(key).is_absolute() else key.replace('\\','/')):value for key,value in pipeline['assets'].items()}
    report=dict(schema_version=1,date='2026-10-18',suite=suite['name'],split='development',mode=mode,
        suite_sha256=identity(folder/'suite.json'),label_review=suite['review'],independent_evaluation=False,representative_accuracy_claim_supported=False,
        config=config.model_dump(),analysis_revision=ANALYSIS_REVISION,pipeline_identity=pipeline,
        environment=dict(os=platform.system(),python=platform.python_version(),cpu=platform.processor(),logical_cores=psutil.cpu_count(),
            ram_gib=round(psutil.virtual_memory().total/1024**3,1),packages=dependency_identity()),
        timing=dict(media_seconds=media,wall_seconds=elapsed,realtime_factor=media/elapsed if elapsed else None,
            cold_first_chunk_seconds=chunks[0] if chunks else None,warm_p95_chunk_seconds=float(np.percentile(chunks[1:] or chunks,95)) if chunks else None,
            peak_process_rss_mib=peak[0]/1024**2,includes_decode_and_policy=True,sustained_live_test=False),
        by_category=by_category,results=results,backends=models.backends,
        interpretation='A tiny, already inspected development set. Unconfirmed scene suggestions do not filter automatically. Draft labels are not independent human truth. Event timing is not pixel coverage. No automatic model qualification or 95% claim follows from these results.')
    write(output,report);return report

def bundle(folder,destination):
    folder=Path(folder);suite=validate_suite(read(folder/'suite.json'));check_pack(folder,suite)
    names={case['id']+'.mp4' for case in suite['cases']}|{'suite.json','FILES.json','CREDITS.md'}
    if {p.name for p in folder.iterdir()}!=names:raise ValueError('Unlisted example-pack files')
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as z:
        for name in sorted(names):z.write(folder/name,'adaptive-censor-public-examples/'+name)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--source-dir',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a=sub.add_parser('run');a.add_argument('--pack',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--mode',choices=['speech','cpu-review'],default='speech');a.add_argument('--allow-draft',action='store_true')
    a=sub.add_parser('bundle');a.add_argument('--pack',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.command=='prepare':print(prepare(args.source_dir,args.output))
    elif args.command=='bundle':bundle(args.pack,args.output)
    else:
        r=run(args.pack,args.output,args.mode,args.allow_draft);print(json.dumps(dict(timing=r['timing'],by_category=r['by_category']),indent=2))
