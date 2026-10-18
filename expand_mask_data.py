"""Prepare and consolidate a larger private, title-separated labeling batch."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import shutil
import time
import numpy as np
from PIL import Image
from labeling import NAME, add_windows, read_json, write_json
from model_setup import prepare_environment
from provenance import digest, file_hash
from select_windows import CONFUSERS, NEUTRAL, embedding_path, normalize_title, occupied
from segmentation_data import MaskDrafter, validate_splits
from segmentation_labels import read_label

OLD = {s: f'mask-{s}-20261012-3' for s in ('train', 'val', 'test')}


def separated(rows, taken, limit, per_title=4):
    """Interleave titles, rejecting nearby moments and duplicate source frames."""
    buckets = defaultdict(list)
    for row in rows:
        buckets[row['title']].append(row)
    used = {k: list(v) for k, v in taken.items()}
    selected = []; counts = Counter()
    titles = sorted(buckets, key=lambda t: (-int(any(r.get('fresh') for r in buckets[t])),
                                           -max(r['rank'] for r in buckets[t]), t))
    while len(selected) < limit and any(buckets.values()):
        for title in titles:
            if counts[title] >= per_title:
                buckets[title] = []; continue
            while buckets[title]:
                row = buckets[title].pop(0)
                if any(abs(row['time'] - x - 6) < 20 for x in used.get(row['source'], [])):
                    continue
                selected.append(row); counts[title] += 1
                used.setdefault(row['source'], []).append(row['time'] - 6)
                break
            if len(selected) >= limit: break
    return selected


def combine_masks(root, name, split, names, collection_identity):
    """Copy pending frames into one queue; retain old annotations and exposure."""
    folder = root / 'segmentation' / name
    datasets = [read_json(root / 'segmentation' / n / 'dataset.json') for n in names]
    validate_splits(datasets)
    if any(d['split'] != split for d in datasets):
        raise ValueError('Combined queue must have one split')
    sources = {d['name']: file_hash(root / 'segmentation' / d['name'] / 'dataset.json') for d in datasets}
    if (folder / 'dataset.json').exists():
        saved = read_json(folder / 'dataset.json')
        if saved['split'] != split or saved['collection_identity'] != collection_identity or saved['source_datasets'] != sources:
            raise ValueError('Consolidated queue identity changed')
        return saved
    rows = []; seen = set(); groups = {}; exposed = []; skipped = Counter()
    for data in datasets:
        source = root / 'segmentation' / data['name']
        for row in data['rows']:
            if read_label(source, data, row):
                skipped['already_reviewed'] += 1; continue
            exposure = read_json(source / 'exposure' / f"{row['id']}.json", {})
            if exposure and split != 'train':
                skipped['exposed_held_out'] += 1; continue
            if row['image_sha256'] in seen:
                skipped['duplicate_picture'] += 1; continue
            for filename, expected in [(row['id'] + '.jpg', row['image_sha256']),
                                       (row['id'] + '-draft.png', row['draft_sha256'])]:
                if file_hash(source / filename) != expected:
                    raise ValueError('A source frame or draft changed')
            seen.add(row['image_sha256']); rid = f'mask-{len(rows):05d}'
            group_key = (row.get('source_project'), row.get('window'), row['group'])
            group = groups.setdefault(group_key, len(groups))
            folder.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / (row['id'] + '.jpg'), folder / (rid + '.jpg'))
            shutil.copyfile(source / (row['id'] + '-draft.png'), folder / (rid + '-draft.png'))
            rows.append(dict(row, id=rid, group=group, origin_dataset=data['name'], origin_id=row['id']))
            if exposure: exposed.append(rid)
    if not rows: raise ValueError('No pending mask frames to consolidate')
    data = dict(version=1, name=name, split=split, rows=rows,
                collection_identity=collection_identity, category='nudity',
                source_datasets=sources,
                drafts_are_ground_truth=False, titles=len({r['title'] for r in rows}),
                evaluation_scope='Title-separated enriched development/regression; previously inspected titles, not a fresh final test',
                skipped=dict(skipped), timing_scope='Offline preparation, not sustained playback')
    write_json(folder / 'dataset.json', data)
    for rid in exposed:
        write_json(folder / 'exposure' / (rid + '.json'),
                   dict(dataset_sha256=file_hash(folder / 'dataset.json'), model_draft_seen=True,
                        inherited=True))
    return data


def combine_video(root, name, projects, library):
    folder = root / 'labeling' / name
    sources = {n: file_hash(root / 'labeling' / n / 'project.json') for n in projects}
    if (folder / 'project.json').exists():
        saved = read_json(folder / 'project.json')
        if saved['source_projects'] != sources: raise ValueError('Combined video sources changed')
        return saved
    windows = []; seen = set(); ids = set()
    for project_name in projects:
        source = root / 'labeling' / project_name
        for w in read_json(source / 'project.json')['windows']:
            if w['split'] != 'train' or library['splits'].get(normalize_title(w['title'])) != 'train':
                raise ValueError('Evaluation title cannot enter combined training')
            if (source / 'labels' / (w['id'] + '.json')).exists(): continue
            sha = file_hash(source / 'clips' / (w['id'] + '.mp4'))
            if not sha: raise ValueError('Missing source clip')
            if sha in seen: continue
            if w['id'] in ids: raise ValueError('Duplicate video window id')
            seen.add(sha); ids.add(w['id'])
            (folder / 'clips').mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / 'clips' / (w['id'] + '.mp4'), folder / 'clips' / (w['id'] + '.mp4'))
            windows.append(dict(w, origin_project=project_name, clip_sha256=sha))
    if not windows: raise ValueError('No pending video windows to consolidate')
    data = dict(version=1, name=name, context=2, windows=windows,
                created=time.strftime('%Y-%m-%dT%H:%M:%S'),
                source_projects=sources)
    write_json(folder / 'project.json', data)
    return data


def disagreement_negatives(screened, taken, limit):
    rows=[]
    for row in screened:
        tag=next((t for t in row.get('evidence',{}).get('tags',[]) if t['category']=='nudity'),None)
        if row['kind']=='possible-content' and tag and tag['state']=='absent':
            rows.append(dict(row,kind='lookalike',included=True,
                             selection_rationale='Detector/verifier disagreement; still requires human review'))
    return separated(sorted(rows,key=lambda r:-r['rank']),taken,limit,per_title=3)


def prepare(root, name, drive_collection, limits=(56,12,12), screen_limit=120, screening_seed=None):
    from analysis_engine import Models, decoded_range
    from detectors import NUDITY_CLASSES
    if not NAME.fullmatch(name): raise ValueError('Invalid batch name')
    private = root / 'data-collections' / name
    if (private / 'summary.json').exists():
        print('Existing published batch retained', flush=True); return read_json(private / 'summary.json')
    lib_path = root / 'labeling/library.json'; library = read_json(lib_path)
    fresh_path = root / 'data-collections' / drive_collection / 'mined.json'
    fresh = read_json(fresh_path)
    if not fresh['complete']: raise ValueError('Fresh episode mining is incomplete')
    models = Models(workers=1); started = time.perf_counter()
    taken = occupied((root / 'labeling').glob('*/project.json'))
    selection_path = private / 'selection.json'; previous = read_json(selection_path, {})
    contract = dict(library_sha256=file_hash(lib_path), fresh_sha256=file_hash(fresh_path),
                    code_sha256=file_hash(Path(__file__)), limits=list(limits), screen_limit=screen_limit,
                    offsets=[-.167,0,.167], width=960,
                    old_datasets={s:file_hash(root / 'segmentation' / n / 'dataset.json') for s,n in OLD.items()},
                    sam_sha256=file_hash(root / 'segmentation-runtime/sam2.1-tiny/model.safetensors'),
                    verifier_manifest_sha256=file_hash(root / 'scene-verifier/installed-4B.json'),
                    screening_seed_sha256=file_hash(screening_seed) if screening_seed else None)
    if previous and previous['contract'] != contract:
        raise ValueError('Collection contract changed; use a new batch name')
    try:
        if previous:
            candidates = previous['candidates']; screened = previous['screened']
        elif screening_seed:
            seed=read_json(screening_seed)
            if seed['contract']['library_sha256']!=contract['library_sha256'] or seed['contract']['fresh_sha256']!=contract['fresh_sha256']:
                raise ValueError('Screening seed has different source inputs')
            candidates=seed['candidates']; screened=seed['screened']
            previous=dict(contract=contract,input_sha256=seed['input_sha256'],candidates=candidates,screened=screened)
            write_json(selection_path,previous)
        else:
            neutral = models.text_embeddings(NEUTRAL)
            confuser = models.text_embeddings(CONFUSERS['nudity'])
            candidates = []; seen = set(); inputs = {}
            paths = [fresh_path] + sorted((root / 'labeling').glob('mined*.json'))
            fresh_sources = {e['source'] for e in fresh['episodes']}
            for path in paths:
                inputs[str(path)] = file_hash(path)
                for e in read_json(path).get('episodes', []):
                    source = e['source']; title = normalize_title(e['title'])
                    if source in seen or not Path(source).is_file(): continue
                    seen.add(source); split = library['splits'].get(title)
                    if split not in ('train','val','test'): continue
                    cache = (fresh_path.parent / 'embeddings' / embedding_path(root,source).name
                             if source in fresh_sources else embedding_path(root,source))
                    if not cache.is_file(): continue
                    inputs[str(cache)] = file_hash(cache)
                    with np.load(cache) as arrays:
                        times = arrays['times']; vectors = arrays['embeddings'].astype(np.float32)
                    margins = (vectors @ confuser.T).max(1) - (vectors @ neutral.T).max(1)
                    stat = Path(source).stat()
                    for row in e['rows']:
                        t = row['time']
                        if t < 30 or t > float(times[-1]) - 30: continue
                        if any(abs(t-x-6) < 20 for x in taken.get(source, [])): continue
                        index = int(np.abs(times-t).argmin()); raw = row['nudity']
                        if raw >= .35: kind='possible-content'; rank=float(raw)
                        elif margins[index] >= .03: kind='lookalike'; rank=float(margins[index])
                        else: continue
                        candidates.append(dict(title=title,display=e['display'],episode=e['episode'],source=source,
                            split=split,time=float(t),rank=rank,raw_nudity=float(raw),kind=kind,
                            fresh=source in fresh_sources,size=stat.st_size,mtime_ns=stat.st_mtime_ns))
            candidates.sort(key=lambda r:-r['rank']); screened=[]
            previous=dict(contract=contract,input_sha256=inputs,candidates=candidates,screened=screened)
            write_json(selection_path,previous)
        for filename, sha in previous['input_sha256'].items():
            if file_hash(Path(filename)) != sha: raise ValueError('A selection input changed')
        selected = previous.get('selected')
        if selected is None:
            verifier = models.verifier('4B'); selected={}; attempts={(r['source'],r['proposal_time']) for r in screened}
            def persist(): write_json(selection_path,previous)
            negative_quota=limits[0]-limits[0]*3//4
            disagreement=disagreement_negatives(screened,taken,negative_quota)
            for kind, quota in [('possible-content',limits[0]*3//4),('lookalike',limits[0]-limits[0]*3//4)]:
                good=[r for r in screened if r.get('included') and r['kind']==kind]
                if kind=='lookalike':
                    keys={(r['source'],r['time']) for r in good}
                    good.extend(r for r in disagreement if (r['source'],r['time']) not in keys)
                title_counts=Counter(r['title'] for r in good)
                occupied_now={k:list(v) for k,v in taken.items()}
                for r in screened: occupied_now.setdefault(r['source'],[]).append(r['proposal_time']-6)
                proposals=separated([r for r in candidates if r['split']=='train' and r['kind']==kind],occupied_now,screen_limit)
                for proposal in proposals:
                    budget=screen_limit-(2*(negative_quota-len(disagreement)) if kind=='possible-content' else 0)
                    if len(good)>=quota or len(screened)>=budget: break
                    if title_counts[proposal['title']]>=4: continue
                    if (proposal['source'],proposal['time']) in attempts: continue
                    attempts.add((proposal['source'],proposal['time']))
                    try:
                        _, burst=decoded_range(Path(proposal['source']),proposal['time']-.3,proposal['time']+.3)
                        positions=sorted({min(range(len(burst)),key=lambda i:abs(burst[i][0]-proposal['time']-offset)) for offset in [-.167,0,.167]})
                        frames=[burst[i] for i in positions]
                        if len(frames)!=3: raise ValueError('Need three real screening frames')
                        evidence=verifier.verify(frames)
                        tag=next(t for t in evidence['tags'] if t['category']=='nudity')
                        include=(tag['state']=='present' and bool(tag['frames'])) if kind=='possible-content' else tag['state']=='absent'
                        center=frames[tag['frames'][len(tag['frames'])//2]][0] if include and tag['frames'] else proposal['time']
                        record=dict(proposal,time=float(center),proposal_time=proposal['time'],included=include,evidence=evidence)
                    except (RuntimeError,ValueError) as error:
                        record=dict(proposal,proposal_time=proposal['time'],included=False,error=str(error))
                    screened.append(record)
                    if record['included']: good.append(record); title_counts[proposal['title']]+=1
                    persist(); print(f'Screened {len(screened)}/{screen_limit}: retained {len(good)}/{quota} {kind}',flush=True)
                selected.setdefault('train',[]).extend(good[:quota])
            for split, limit in zip(('val','test'),limits[1:]):
                chosen=[]; local={k:list(v) for k,v in taken.items()}
                for kind, quota in [('possible-content',limit*3//4),('lookalike',limit-limit*3//4)]:
                    picked=separated([r for r in candidates if r['split']==split and r['kind']==kind],local,quota)
                    chosen.extend(picked)
                    for r in picked: local.setdefault(r['source'],[]).append(r['time']-6)
                selected[split]=chosen
            if not selected.get('train'): raise ValueError('No training examples passed local screening')
            previous['selected']=selected; persist()
        for split, groups in selected.items():
            for g in groups:
                stat=Path(g['source']).stat()
                if (stat.st_size,stat.st_mtime_ns)!=(g['size'],g['mtime_ns']): raise ValueError('Selected episode changed')
                if library['splits'][g['title']]!=split: raise ValueError('Source title crossed splits')
        models.close()
        projects={}
        for split,groups in selected.items():
            project_name=f'new-{split}-{name}'; destination=root/'labeling'/project_name
            if (destination/'project.json').exists():
                projects[split]=read_json(destination/'project.json'); continue
            windows=[dict(id=f"batch-{g['title'].replace(' ','-')}-{g['episode']}-{int(g['time'])}",
                          title=g['display'],episode=g['episode'],source=g['source'],split=split,
                          start=round(g['time']-6,3),length=12,stratum='collection:'+g['kind']) for g in groups]
            print(f'Extracting {len(windows)} new {split} clips',flush=True)
            projects[split]=add_windows(root,project_name,windows,workers=2)
        models=Models(workers=1); drafter=MaskDrafter(root); new_names={}
        for split,groups in selected.items():
            new_name=f'mask-{split}-new-{name}'; folder=root/'segmentation'/new_name; new_names[split]=new_name
            if (folder/'dataset.json').exists(): continue
            rows=[]; folder.mkdir(parents=True,exist_ok=True)
            for number,(g,w) in enumerate(zip(groups,projects[split]['windows'])):
                clip=root/'labeling'/projects[split]['name']/'clips'/(w['id']+'.mp4')
                center=g['time']-w['clip_offset']; _,burst=decoded_range(clip,center-.3,center+.3)
                indices=[min(range(len(burst)),key=lambda i:abs(burst[i][0]-center-offset)) for offset in [-.167,0,.167]]
                if len(set(indices))!=3: raise ValueError('Need three distinct native mask frames')
                for index in indices:
                    timestamp,pixels=burst[index]; picture=Image.fromarray(pixels)
                    if picture.width>960: picture=picture.resize((960,round(picture.height*960/picture.width)),Image.Resampling.LANCZOS)
                    pixels=np.asarray(picture); h,width=pixels.shape[:2]; boxes=[]; hits=[]
                    for hit in sorted(models.nude('auto','640m').detect(pixels[:,:,::-1].copy()),key=lambda r:-r['score']):
                        if hit['class'] not in NUDITY_CLASSES or hit['score']<.2: continue
                        x,y,bw,bh=hit['box']; box=[max(0,x),max(0,y),min(width,x+bw),min(h,y+bh)]
                        if min(box[2]-box[0],box[3]-box[1])<8: continue
                        boxes.append(box); hits.append(dict(detector_class=hit['class'],score=float(hit['score']),box=box))
                        if len(boxes)==6: break
                    mask,scores=drafter.predict(pixels,boxes); rid=f'mask-{len(rows):05d}'
                    picture.save(folder/(rid+'.jpg'),quality=95); Image.fromarray(mask.astype(np.uint8)*255).save(folder/(rid+'-draft.png'))
                    rows.append(dict(id=rid,title=g['title'],display=g['display'],split=split,group=number,
                        source_project=projects[split]['name'],window=w['id'],source_identity=file_hash(clip),
                        time=timestamp,source_time=timestamp+w['clip_offset'],width=width,height=h,
                        image_sha256=file_hash(folder/(rid+'.jpg')),draft_sha256=file_hash(folder/(rid+'-draft.png')),
                        prompt_boxes=hits,geometry_scores=scores,draft_labeler='model',semantic_label=None,
                        selection_hint=g['kind'],selection_labeler='model',fresh_episode=g['fresh']))
                print(f'{split}: generated {number+1}/{len(groups)} frame groups',flush=True)
            write_json(folder/'dataset.json',dict(version=1,name=new_name,split=split,rows=rows,model=drafter.identity,
                category='nudity',collection_identity=digest(previous),drafts_are_ground_truth=False,
                evaluation_scope='Model-enriched title-separated development/regression, not fresh final qualification',
                titles=len({r['title'] for r in rows}),code_sha256=file_hash(Path(__file__))))
        identity=digest(previous); combined={}
        for split in selected:
            combined[split]=combine_masks(root,f'mask-{split}-batch-{name}',split,[OLD[split],new_names[split]],identity)
        validate_splits(list(combined.values()))
        video=combine_video(root,f'training-batch-{name}',
                            ['training-targeted-20261011-1',projects['train']['name']],library)
        if file_hash(lib_path)!=contract['library_sha256']: raise ValueError('Library changed')
        summary=dict(name=name,identity=identity,mask_projects={s:dict(name=d['name'],frames=len(d['rows']),titles=d['titles'],
                      groups=len({r['group'] for r in d['rows']}),nonempty_drafts=sum(bool(r['prompt_boxes']) for r in d['rows'])) for s,d in combined.items()},
            new_groups={s:len(g) for s,g in selected.items()},new_frames={s:len(g)*3 for s,g in selected.items()},
            new_training_hints=dict(Counter(g['kind'] for g in selected['train'])),
            fresh_episode_groups=sum(g['fresh'] for gs in selected.values() for g in gs),
            drives=sorted({Path(g['source']).drive for gs in selected.values() for g in gs}),
            video_project=video['name'],video_clips=len(video['windows']),screened=len(screened),
            human_labels_created=False,old_annotations_changed=False,titles_disjoint=True,
            accuracy_gain_measured=False,player_changed=False,seconds=round(time.perf_counter()-started,3))
        write_json(private/'summary.json',summary); print(summary,flush=True); return summary
    finally: models.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',required=True); parser.add_argument('--drive-collection',required=True)
    parser.add_argument('--groups',nargs=3,type=int,default=[56,12,12]); parser.add_argument('--screen-limit',type=int,default=120)
    parser.add_argument('--screening-seed',type=Path)
    args=parser.parse_args(); prepare(prepare_environment(),args.name,args.drive_collection,args.groups,args.screen_limit,args.screening_seed)
