"""Prepare diverse raw anatomy proposals from completed human training clips."""
import argparse
from pathlib import Path
import time
import numpy as np

from human_calibration import collect
from labeling import project_dir,write_json
from model_setup import prepare_environment
from provenance import file_hash


def prepare(root,projects,name,limit=80):
    from analysis_engine import Models,native_frames
    from detectors import NUDITY_CLASSES
    from roi_gate import crop,body_crop
    from PIL import Image
    folder=root/'roi'/name
    if folder.exists():raise FileExistsError('Never overwrite a crop review project')
    source_rows=collect([project_dir(root,n) for n in projects])
    # Interleave titles and sample gold/clean targets. Interval labels do not
    # become region labels: every detector box starts unreviewed.
    by_title={}
    for item in source_rows:by_title.setdefault(item[1]['title'],[]).append(item)
    ordered=[]
    while any(by_title.values()):
        for title in sorted(by_title):
            if by_title[title]:ordered.append(by_title[title].pop(0))
    models=Models();rows=[];pictures=[];per_title={};folder.mkdir(parents=True)
    started=time.perf_counter()
    try:
        for source,w,label in ordered:
            if per_title.get(w['title'],0)>=8:continue
            low,high=w['target'];gold=label['categories']['nudity']['intervals']
            times=[(a+b)/2 for a,b in gold]+list(np.linspace(low+.3,high-.3,8))
            previous=[]
            for t in times:
                if any(abs(t-x)<.5 for x in previous):continue
                previous.append(t)
                frames=native_frames(source/'clips'/f"{w['id']}.mp4",max(0,t-.06),min(w['clip_duration'],t+.1),30)
                timestamp,pixels=min(frames,key=lambda f:abs(f[0]-t));height,width=pixels.shape[:2]
                for hit in models.nude('auto','640m').detect(pixels[:,:,::-1].copy()):
                    # Weak raw proposals supply useful human negatives too;
                    # this collection floor never changes playback acceptance.
                    if hit['class'] not in NUDITY_CLASSES or hit['score']<.20:continue
                    x,y,bw,bh=hit['box'];region=[x/width,y/height,bw/width,bh/height]
                    local=crop(pixels,region);body=body_crop(pixels,region)
                    if min(local.shape[:2])<16:continue
                    rid=f'roi-{len(rows):04}'
                    Image.fromarray(local).save(folder/f'{rid}.jpg',quality=95)
                    Image.fromarray(body).save(folder/f'{rid}-body.jpg',quality=95)
                    rows.append(dict(id=rid,title=w['title'],project=source.name,window=w['id'],time=timestamp,
                                     region=region,detector_class=hit['class'],detector_score=float(hit['score']),
                                     source_identity=file_hash(source/'clips'/f"{w['id']}.mp4"),split='train',
                                     label=None,labeler=None,reviewed=False))
                    pictures.extend([(timestamp,local),(timestamp,body)])
                    per_title[w['title']]=per_title.get(w['title'],0)+1
                    if len(rows)>=limit or per_title[w['title']]>=8:break
                if len(rows)>=limit or per_title.get(w['title'],0)>=8:break
            if len(rows)>=limit:break
        if not rows:raise ValueError('No supported anatomy proposals found')
        embeddings=models.clip_embeddings(pictures,'cpu',fit='squash').reshape(len(rows),-1)
        np.savez_compressed(folder/'features.npz',embeddings=embeddings)
        write_json(folder/'dataset.json',dict(review_api=1,rows=rows,feature_views=['local','body'],anatomy_model='640m',
                   anatomy_identity=file_hash(root/'scene-verifier/640m.onnx'),
                   clip_identity=file_hash(root/'clip/onnx/vision_model_quantized.onnx'),
                   feature_identity=file_hash(folder/'features.npz'),crop_method='Raw detector box with 25% margin; matches runtime',
                   images_sha256={p.name:file_hash(p) for p in sorted(folder.glob('roi-*.jpg'))},
                   purpose='Human region labels required. Scene intervals do not label detector boxes.',
                   sampling=dict(target_frames=8,maximum_crops_per_title=8,raw_detector_floor=.20),
                   preparation_seconds=round(time.perf_counter()-started,3)))
        print(f'{len(rows)} unlabelled raw crop pairs across {len(per_title)} training titles; open /roi.html#{name}',flush=True)
    finally:models.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--projects',nargs='+',required=True)
    p.add_argument('--name',required=True);p.add_argument('--limit',type=int,default=80)
    a=p.parse_args();prepare(prepare_environment(),a.projects,a.name,a.limit)
