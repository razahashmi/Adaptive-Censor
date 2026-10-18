"""Title-separated mask datasets with explicit model drafts and reviewed labels.

SAM predicts outline geometry from raw detector boxes. Neither box nor mask
establishes that its pixels depict exposed anatomy. No human labels are invented.
"""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import time
import numpy as np
from PIL import Image
from labeling import read_json, write_json, is_complete
from model_setup import prepare_environment
from provenance import digest, file_hash
from select_windows import normalize_title

REVISION=1
SOURCES={'train':['training-actions-human-1','train-drive-expansion-1','training-content-human-2'],
         'val':['val-2'],'test':['test-1','test-2']}


def choose_groups(root,split,limit,library):
    candidates=[];known=set()
    for name in SOURCES[split]:
        folder=root/'labeling'/name;project=read_json(folder/'project.json')
        for w in project['windows']:
            title=normalize_title(w['title'])
            if library['splits'].get(title)!=split or w['split']!=split:
                raise ValueError('Permanent title assignment conflicts with mask collection')
            label=read_json(folder/'labels'/f"{w['id']}.json",{})
            if not is_complete(label):continue
            if split in ('train','val') and label['labeler']!='human':continue
            key=(title,w['source'],w['start'])
            if key in known:continue
            known.add(key)
            rule=label['categories']['nudity'];spans=rule['intervals']
            center=sum(max(spans,key=lambda p:p[1]-p[0]))/2 if spans else sum(w['target'])/2
            candidates.append(dict(title=title,display=w['title'],source_project=name,window=w['id'],
                clip_sha256=file_hash(folder/'clips'/f"{w['id']}.mp4"),label_sha256=file_hash(folder/'labels'/f"{w['id']}.json"),
                split=split,time=center,clip_duration=w['clip_duration'],target=w['target'],
                selection_hint='interval-positive' if spans else 'contrast',selection_labeler=label['labeler']))
    # Positive-enriched development collection; cycle titles, never fill with
    # duplicated pictures or silently change the permanent split.
    buckets=defaultdict(list)
    for r in sorted(candidates,key=lambda r:(r['selection_hint']!='interval-positive',r['window'])):
        buckets[r['title']].append(r)
    ordered=sorted(buckets,key=lambda t:(not any(r['selection_hint']=='interval-positive' for r in buckets[t]),t))
    selected=[]
    while len(selected)<limit and any(buckets.values()):
        for title in ordered:
            if buckets[title]:selected.append(buckets[title].pop(0))
            if len(selected)>=limit:break
    if not selected:raise ValueError('No eligible source clips for '+split)
    return selected


def validate_splits(datasets):
    owned={}
    for dataset in datasets:
        split=dataset['split']
        for row in dataset['rows']:
            if row['split']!=split:raise ValueError('Row has the wrong split')
            title=normalize_title(row['title'])
            if title in owned and owned[title]!=split:raise ValueError('Title crosses training/evaluation splits')
            owned[title]=split
    return owned


class MaskDrafter:
    def __init__(self,root):
        import torch
        from transformers import Sam2Model,Sam2Processor,Sam2Config
        from setup_segmentation import check
        folder=root/'segmentation-runtime/sam2.1-tiny';self.identity=check(folder)
        torch.set_num_threads(4);self.torch=torch;self.device='cuda' if torch.cuda.is_available() else 'cpu'
        # The official checkpoint describes the shared image/video backbone.
        # Select the shared backbone's image interface explicitly. The pinned
        # Transformers release supplies its processor through Torchvision.
        config=read_json(folder/'config.json')
        if config.pop('model_type') not in ('sam2','sam2_video'):raise ValueError('Unexpected SAM architecture')
        self.model=Sam2Model.from_pretrained(folder,config=Sam2Config(**config),local_files_only=True,use_safetensors=True).to(self.device).eval()
        self.processor=Sam2Processor.from_pretrained(folder,local_files_only=True)

    def predict(self,pixels,boxes):
        h,w=pixels.shape[:2]
        if not boxes:return np.zeros((h,w),dtype=bool),[]
        inputs=self.processor(images=Image.fromarray(pixels),input_boxes=[boxes],return_tensors='pt').to(self.device)
        with self.torch.inference_mode():
            outputs=self.model(**inputs,multimask_output=False)
        masks=self.processor.post_process_masks(outputs.pred_masks.cpu(),inputs['original_sizes'])[0].numpy()
        result=np.zeros((h,w),dtype=bool)
        # Constrain each prompted outline to a 10% expanded detector envelope.
        # SAM can otherwise select the entire person from a small anatomy box.
        for mask,box in zip(masks,boxes):
            x1,y1,x2,y2=box;mx=(x2-x1)*.1;my=(y2-y1)*.1
            envelope=np.zeros((h,w),dtype=bool)
            envelope[max(0,int(y1-my)):min(h,int(np.ceil(y2+my))),max(0,int(x1-mx)):min(w,int(np.ceil(x2+mx)))]=True
            result|=np.asarray(mask[0],dtype=bool)&envelope
        return result,outputs.iou_scores.detach().float().cpu().reshape(-1).tolist()


def prepare(root,suffix,limits=(24,8,8),maximum_width=960):
    from analysis_engine import Models,decoded_range
    from detectors import NUDITY_CLASSES
    library=read_json(root/'labeling/library.json');private=root/'segmentation'/('collection-'+suffix)
    plan=dict(version=REVISION,library_sha256=file_hash(root/'labeling/library.json'),
              groups={s:choose_groups(root,s,n,library) for s,n in zip(SOURCES,limits)},
              offsets=[-.167,0,.167],maximum_width=maximum_width,
              code_sha256=file_hash(Path(__file__)),model_sha256=file_hash(root/'segmentation-runtime/sam2.1-tiny/model.safetensors'))
    previous=read_json(private/'plan.json',{})
    if previous and previous!=plan:raise ValueError('Mask collection contract changed; use a new name')
    write_json(private/'plan.json',plan)
    planned=[dict(split=s,rows=rows) for s,rows in plan['groups'].items()];validate_splits(planned)
    models=Models(workers=1);drafter=MaskDrafter(root);datasets=[];started=time.perf_counter()
    try:
        for split,groups in plan['groups'].items():
            name=f'mask-{split}-{suffix}';folder=root/'segmentation'/name
            existing=read_json(folder/'dataset.json',{})
            if existing:
                if existing['collection_identity']!=digest(plan):raise ValueError('Different mask collection already exists')
                datasets.append(existing);print(name+': existing frozen dataset retained',flush=True);continue
            folder.mkdir(parents=True,exist_ok=True);rows=[]
            for group_index,g in enumerate(groups):
                source=root/'labeling'/g['source_project']/'clips'/f"{g['window']}.mp4"
                if file_hash(source)!=g['clip_sha256']:raise ValueError('Mask source clip changed')
                low=max(g['target'][0],g['time']-.3);high=min(g['target'][1],g['time']+.3)
                _,decoded=decoded_range(source,low,high)
                if not decoded:raise ValueError('Source has no real decoded mask frames')
                used=set()
                for offset in plan['offsets']:
                    target=min(high,max(low,g['time']+offset))
                    index=min(range(len(decoded)),key=lambda i:abs(decoded[i][0]-target))
                    if index in used:continue
                    used.add(index);timestamp,pixels=decoded[index];picture=Image.fromarray(pixels)
                    if picture.width>maximum_width:picture=picture.resize((maximum_width,round(picture.height*maximum_width/picture.width)),Image.Resampling.LANCZOS)
                    pixels=np.asarray(picture);h,w=pixels.shape[:2];hits=[];boxes=[]
                    for hit in sorted(models.nude('auto','640m').detect(pixels[:,:,::-1].copy()),key=lambda a:-a['score']):
                        if hit['class'] not in NUDITY_CLASSES or hit['score']<.2:continue
                        x,y,bw,bh=hit['box'];box=[max(0,x),max(0,y),min(w,x+bw),min(h,y+bh)]
                        if box[2]-box[0]<8 or box[3]-box[1]<8:continue
                        boxes.append(box);hits.append(dict(detector_class=hit['class'],score=float(hit['score']),box=box))
                        if len(boxes)>=6:break
                    mask,scores=drafter.predict(pixels,boxes);rid=f'mask-{len(rows):05d}'
                    picture.save(folder/f'{rid}.jpg',quality=95)
                    Image.fromarray(mask.astype(np.uint8)*255).save(folder/f'{rid}-draft.png')
                    rows.append(dict(id=rid,title=g['title'],display=g['display'],split=split,group=group_index,
                        source_project=g['source_project'],window=g['window'],source_identity=g['clip_sha256'],time=timestamp,
                        width=w,height=h,image_sha256=file_hash(folder/f'{rid}.jpg'),draft_sha256=file_hash(folder/f'{rid}-draft.png'),
                        prompt_boxes=hits,geometry_scores=scores,draft_labeler='model',semantic_label=None,
                        selection_hint=g['selection_hint'],selection_labeler=g['selection_labeler']))
                print(f'{name}: {group_index+1}/{len(groups)} groups, {len(rows)} real frames',flush=True)
            dataset=dict(version=REVISION,name=name,split=split,collection_identity=digest(plan),rows=rows,
                category='nudity',labeling='Draft outline geometry only. Human or separately identified AI review must judge target anatomy.',
                evaluation_scope='Title-separated, positive-enriched mask development/regression; previously inspected titles, not a fresh final test',
                model=drafter.identity,code_sha256=file_hash(Path(__file__)),drafts_are_ground_truth=False,
                titles=len({r['title'] for r in rows}),timing_scope='Offline data preparation, not sustained playback speed')
            write_json(folder/'dataset.json',dataset);datasets.append(dataset)
        validate_splits(datasets)
        summary=dict(collection_identity=digest(plan),datasets={d['name']:dict(split=d['split'],frames=len(d['rows']),titles=d['titles'],
                      prompted_frames=sum(bool(r['prompt_boxes']) for r in d['rows'])) for d in datasets},
                     titles_disjoint=True,human_labels_created=False,player_defaults_changed=False,
                     seconds=round(time.perf_counter()-started,3),scope=datasets[0]['evaluation_scope'])
        write_json(private/'summary.json',summary);print(summary,flush=True)
    finally:models.close()


def export(root,names,output,drafts=False):
    """Portable COCO RLE; held-out datasets never fall back to model labels."""
    from segmentation_labels import read_label
    datasets=[read_json(root/'segmentation'/n/'dataset.json') for n in names];validate_splits(datasets)
    output.mkdir(parents=True,exist_ok=True);counts={}
    def rle(mask):
        flat=np.asarray(mask,dtype=np.uint8).ravel(order='F');runs=[];value=0;count=0
        for pixel in flat:
            if int(pixel)==value:count+=1
            else:runs.append(count);count=1;value=int(pixel)
        runs.append(count);return dict(size=list(mask.shape),counts=runs)
    for dataset in datasets:
        folder=root/'segmentation'/dataset['name'];images=[];annotations=[];skipped=0
        for row in dataset['rows']:
            if file_hash(folder/(row['id']+'.jpg'))!=row['image_sha256']:raise ValueError('Export source frame changed')
            label=read_label(folder,dataset,row)
            if label and label['status']=='reviewed':
                filename=row['id']+'-gold.png';labeler=label['labeler'];exposed=label['drafts_shown']
                if dataset['split']!='train' and (exposed or labeler!='human'):skipped+=1;continue
                if labeler!='human' and not drafts:skipped+=1;continue
            elif drafts and dataset['split']=='train':
                filename=row['id']+'-draft.png';labeler='model';exposed=True
                if file_hash(folder/filename)!=row['draft_sha256']:raise ValueError('Export draft mask changed')
            else:skipped+=1;continue
            with Image.open(folder/filename) as image:mask=np.asarray(image.convert('L'))>=128
            if mask.shape!=(row['height'],row['width']):raise ValueError('Export mask shape differs from frame')
            image_id=len(images)+1
            relative=Path('images')/dataset['name']/(row['id']+'.jpg');destination=output/relative
            destination.parent.mkdir(parents=True,exist_ok=True)
            if destination.exists() and file_hash(destination)!=row['image_sha256']:raise ValueError('Export image already exists with different contents')
            import shutil
            shutil.copyfile(folder/(row['id']+'.jpg'),destination)
            images.append(dict(id=image_id,file_name=relative.as_posix(),width=row['width'],height=row['height'],
                          title=row['title'],split=row['split'],labeler=labeler,drafts_shown=exposed,source_identity=row['source_identity']))
            if mask.any():
                yy,xx=np.where(mask);annotations.append(dict(id=len(annotations)+1,image_id=image_id,category_id=1,
                    segmentation=rle(mask),area=int(mask.sum()),bbox=[int(xx.min()),int(yy.min()),int(xx.max()-xx.min()+1),int(yy.max()-yy.min()+1)],
                    iscrowd=0,labeler=labeler))
        destination=output/(dataset['name']+'.json')
        if destination.exists():raise FileExistsError('Never overwrite an exported training/test dataset')
        write_json(destination,dict(images=images,annotations=annotations,categories=[dict(id=1,name='target-anatomy')],
            metadata=dict(split=dataset['split'],dataset_sha256=file_hash(folder/'dataset.json'),pseudo_labels_allowed=drafts,
                          skipped_unreviewed_or_exposed=skipped,scope=dataset['evaluation_scope'])))
        counts[dataset['name']]=dict(exported=len(images),skipped=skipped)
    print(counts,flush=True);return counts


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare');q.add_argument('--name',required=True);q.add_argument('--groups',nargs=3,type=int,default=[24,8,8])
    q=sub.add_parser('export');q.add_argument('--projects',nargs='+',required=True);q.add_argument('--output',type=Path,required=True);q.add_argument('--allow-training-drafts',action='store_true')
    args=p.parse_args();root=prepare_environment()
    if args.command=='prepare':prepare(root,args.name,args.groups)
    else:export(root,args.projects,args.output,args.allow_training_drafts)
