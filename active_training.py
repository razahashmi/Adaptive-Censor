"""Retrieve and screen fresh training moments using human positives and confusers.

Mining scores and vision-model screening never become human annotations. Weak
sources do not fill quotas, and validation/test shows never enter this queue.
"""
import argparse
from collections import defaultdict
from pathlib import Path
import time
import numpy as np

from human_calibration import collect
from labeling import LABEL_CATEGORIES, add_windows,project_dir,read_json,write_json
from model_setup import prepare_environment
from provenance import digest,file_hash
from select_windows import embedding_path,normalize_title,occupied


def screening_frames(path,start,end):
    from analysis_engine import native_frames,decoded_range
    try:
        return native_frames(path,start,end,1)
    except RuntimeError:
        # Some variable-rate sources skip every implied 1fps frame index.
        # Use real decoded timestamps; never duplicate or synthesize pictures.
        _,frames=decoded_range(path,start,end)
        if not frames:raise
        times=np.array([t for t,_ in frames])
        selected=sorted({int(np.abs(times-t).argmin()) for t in np.arange(start+.5,end,1.)})
        return [frames[i] for i in selected]


def prepare(root,projects,name,limit=24,screen_limit=40):
    from analysis_engine import Models,native_frames
    destination=project_dir(root,name)
    if destination.exists():raise FileExistsError('Never replace a training review queue')
    rows=collect([project_dir(root,n) for n in projects]);models=Models(workers=1)
    library=read_json(root/'labeling/library.json');taken=occupied(list((root/'labeling').glob('*/project.json')))
    positive=defaultdict(list);negative=defaultdict(list);records=[];started=time.perf_counter()
    private=root/'benchmarks'/name;private.mkdir(parents=True,exist_ok=True)
    anchor_hashes={f'{folder.name}/{w["id"]}':file_hash(folder/'labels'/f"{w['id']}.json") for folder,w,_ in rows}
    try:
        for folder,w,label in rows:
            moments={}
            for c in LABEL_CATEGORIES:
                rule=label['categories'][c]
                if rule['status']!='reviewed':continue
                times=[(a+b)/2 for a,b in rule['intervals']][:3]
                if not times and not rule['ambiguous']:times=[sum(w['target'])/2]
                for t in times:moments.setdefault(t,[]).append((c,bool(rule['intervals'])))
            pictures=[];tags=[]
            for t,items in moments.items():
                frames=native_frames(folder/'clips'/f"{w['id']}.mp4",max(0,t-.05),min(w['clip_duration'],t+.1),30)
                picture=min(frames,key=lambda f:abs(f[0]-t));pictures.append(picture);tags.append(items)
            if not pictures:continue
            vectors=models.clip_embeddings(pictures,'cpu',fit='squash')
            for v,items in zip(vectors,tags):
                for c,is_positive in items:(positive if is_positive else negative)[c].append(v)
        proposals=[];cache_hashes={}
        for number,f in enumerate(library['files'],1):
            title=normalize_title(f['title']);source=f['source']
            if library['splits'].get(title)!='train' or not Path(source).is_file():continue
            path=embedding_path(root,source)
            if not path.is_file():continue
            with np.load(path) as data:times=data['times'];vectors=data['embeddings'].astype(np.float32)
            cache_hashes[source]=file_hash(path)
            for c in LABEL_CATEGORIES:
                if len(positive[c])<2 or not negative[c]:continue
                matches=vectors@np.asarray(positive[c]).T
                pos=np.sort(matches,axis=1)[:,-2:].mean(1)
                neg=(vectors@np.asarray(negative[c]).T).max(1)
                margin=pos-neg
                # Fixed evidence floors; retrieval is a sampling heuristic.
                eligible=np.flatnonzero((pos>=.72)&(margin>=.015))
                for i in eligible[np.argsort(margin[eligible])[-8:]][::-1]:
                    start=max(0,round(float(times[i])-6,1))
                    if any(abs(start-other)<16 for other in taken.get(source,[])):continue
                    proposals.append(dict(category=c,title=title,display=f.get('display',title),episode=f['episode'],
                                          source=source,start=start,retrieval_similarity=float(pos[i]),margin=float(margin[i])))
            if number%100==0:print(f'Retrieval: inspected {number}/{len(library["files"])} inventory entries',flush=True)
        buckets={c:sorted([p for p in proposals if p['category']==c],key=lambda p:-p['margin']) for c in LABEL_CATEGORIES}
        previous=read_json(private/'screening.json',{})
        if previous and previous.get('anchors')!=anchor_hashes:
            raise ValueError('Human retrieval anchors changed; choose a new collection name')
        screened=previous.get('screened',[])
        chosen=[p for p in screened if p.get('included')]
        per_title=defaultdict(int);attempted=list(screened)
        for p in chosen:per_title[p['title']]+=1
        verifier=models.verifier('4B')
        def record_screening():
            write_json(private/'screening.json',dict(screened=screened,anchors=anchor_hashes,
                       library_sha256=file_hash(root/'labeling/library.json'),embedding_sha256=cache_hashes,
                       scope='Training collection hints, not assigned labels or accuracy evidence'))
        while len(screened)<screen_limit and any(buckets.values()) and len(chosen)<limit:
            for c in ('gore','nudity','sexual_activity','violence','kissing'):
                bucket=buckets[c]
                proposal=None
                while bucket:
                    p=bucket.pop(0)
                    if per_title[p['title']]>=2 or any(p['source']==a['source'] and abs(p['start']-a['start'])<20 for a in attempted):continue
                    proposal=p;break
                if proposal is None:continue
                attempted.append(proposal)
                try:
                    frames=screening_frames(Path(proposal['source']),proposal['start'],proposal['start']+12)
                except (RuntimeError,ValueError) as error:
                    screened.append(dict(proposal,included=False,kind='decode-failure',error=str(error)))
                    record_screening()
                    print(f'Skipped unavailable sample: {proposal["title"]} at {proposal["start"]:.1f}s',flush=True)
                    if len(screened)>=screen_limit:break
                    continue
                evidence=verifier.verify(frames[:20])
                positive_tags=[tag for tag in evidence['tags'] if tag['state']=='present' and tag['category'] in LABEL_CATEGORIES and tag['frames']]
                # A highly similar proposal explicitly rejected by the screener
                # is a useful confusing negative, not a human-assigned negative.
                confuser=not positive_tags and any(tag['category']==c and tag['state']=='absent' for tag in evidence['tags'])
                take=bool(positive_tags) or (confuser and sum(r['kind']=='confuser' for r in chosen)<6)
                decision=dict(proposal,evidence=evidence,included=take,
                              kind='possible-content' if positive_tags else 'confuser' if confuser else 'weak')
                screened.append(decision)
                if take:chosen.append(decision);per_title[proposal['title']]+=1
                record_screening()
                print(f'Screened {len(screened)}/{screen_limit}; retained {len(chosen)} targets ({c})',flush=True)
                if len(screened)>=screen_limit or len(chosen)>=limit:break
        if not chosen:raise ValueError('No proposals passed retrieval and screening; no empty queue published')
        for folder,w,_ in rows:
            if file_hash(folder/'labels'/f"{w['id']}.json")!=anchor_hashes[f'{folder.name}/{w["id"]}']:
                raise ValueError('Human anchors changed during collection')
        windows=[dict(id=f"active-{p['title'].replace(' ','-')}-{p['episode']}-{int(p['start'])}",
                      title=p['display'],episode=p['episode'],source=p['source'],start=p['start'],length=12,
                      split='train',stratum='training-screened',score=None) for p in chosen]
        project=add_windows(root,name,windows,workers=2)
        write_json(private/'summary.json',dict(project=name,targets=len(project['windows']),titles=len({p['title'] for p in chosen}),
                   possible_content=sum(p['kind']=='possible-content' for p in chosen),confusers=sum(p['kind']=='confuser' for p in chosen),
                   drives=sorted({Path(p['source']).drive for p in chosen}),screened=len(screened),
                   labels_created=False,drafts_published=False,validation_or_test_titles_used=False,
                   elapsed_seconds=round(time.perf_counter()-started,3)))
        print(f'Published {name}: {len(windows)} unlabelled targets; no labels or drafts assigned',flush=True)
    finally:models.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--projects',nargs='+',required=True)
    p.add_argument('--name',required=True);p.add_argument('--limit',type=int,default=24)
    p.add_argument('--screen-limit',type=int,default=40)
    a=p.parse_args();prepare(prepare_environment(),a.projects,a.name,a.limit,a.screen_limit)
