"""Strict mask-geometry scoring against completed blind human frame labels."""
from collections import defaultdict
from pathlib import Path
import argparse
import numpy as np
from PIL import Image
from labeling import read_json,write_json
from provenance import file_hash
from segmentation_labels import read_label
from evaluate_masks import counts,rates


def evaluate(folder,blind_groups_only=False):
    data=read_json(folder/'dataset.json');total=dict(tp=0,fp=0,fn=0,clean=0,gold=0);results=[];unsure=[];titles=defaultdict(lambda:{k:0 for k in total})
    if blind_groups_only and data['split'] not in ('val','test'):
        raise ValueError('Blind group subsets are for evaluation splits only')
    labels=[];blocked=set();excluded=[]
    for row in data['rows']:
        label=read_label(folder,data,row)
        if not label:raise ValueError('Incomplete mask reviews: '+row['id'])
        if label['labeler']!='human' or (label['drafts_shown'] and not blind_groups_only):
            raise ValueError('Independent scoring requires human labels without model-draft exposure')
        if label['drafts_shown']:blocked.add((row['title'],row['group']))
        labels.append(label)
    for row,label in zip(data['rows'],labels):
        if (row['title'],row['group']) in blocked:
            excluded.append(row['id']);continue
        if label['status']=='unsure':unsure.append(row['id']);continue
        if label['status']!='reviewed':raise ValueError('Incomplete mask status')
        if file_hash(folder/(row['id']+'.jpg'))!=row['image_sha256'] or file_hash(folder/(row['id']+'-draft.png'))!=row['draft_sha256']:
            raise ValueError('Mask prediction/source identity changed')
        def load(suffix):
            with Image.open(folder/(row['id']+suffix)) as image:return np.asarray(image.convert('L'))>=128
        predicted=load('-draft.png');gold=load('-gold.png');c=counts(predicted,gold)
        if predicted.shape!=(row['height'],row['width']):raise ValueError('Mask dimensions differ from source frame')
        for k,v in c.items():total[k]+=v;titles[row['title']][k]+=v
        results.append(dict(id=row['id'],title=row['title'],group=row['group'],counts=c,**rates(c)))
    if not results:raise ValueError('No definite independent human masks to score')
    clean_frames=[r for r in results if not r['counts']['gold']]
    return dict(dataset_identity=file_hash(folder/'dataset.json'),split=data['split'],expected_frames=len(data['rows']),
        blind_groups_only=blind_groups_only,exposure_excluded_frames=len(excluded),exposure_excluded_groups=len(blocked),
        exposure_excluded_ids=excluded,positive_frames=sum(bool(r['counts']['gold']) for r in results),
        positive_titles=len({r['title'] for r in results if r['counts']['gold']}),
        clean_frames=len(clean_frames),clean_frames_with_prediction=sum(bool(r['counts']['fp']) for r in clean_frames),
        clean_frame_false_mask_rate=sum(bool(r['counts']['fp']) for r in clean_frames)/len(clean_frames) if clean_frames else None,
        pixel_precision=total['tp']/(total['tp']+total['fp']) if total['tp']+total['fp'] else None,
        scored_frames=len(results),unsure_frames=len(unsure),positive_groups=len({(r['title'],r['group']) for r in results if r['counts']['gold']}),
        counts=total,**rates(total),samples=results,by_title={t:dict(counts=c,**rates(c)) for t,c in titles.items()},
        qualification_passed=False,scope=data['evaluation_scope'],
        limits='Sparse correlated frames measure geometry only; not whole-video event recognition, leakage or an automatic-filter qualification')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--blind-groups-only',action='store_true',help='Exclude entire scene groups if any frame had model-draft exposure')
    a=p.parse_args();from model_setup import prepare_environment
    if a.output.exists():raise FileExistsError('Never overwrite a measured mask report')
    write_json(a.output,evaluate(prepare_environment()/'segmentation'/a.project,a.blind_groups_only))
