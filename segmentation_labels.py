"""Authenticated local mask review, with separate drafts and immutable provenance."""
from pathlib import Path
import math
import re
import threading
import hashlib
from PIL import Image,ImageDraw
import numpy as np
from fastapi import APIRouter,Body,HTTPException
from fastapi.responses import FileResponse
from labeling import NAME,read_json,write_json
from provenance import file_hash

ROW=re.compile(r'^mask-\d{5}$');LOCK=threading.RLock()
DEFINITION=('Outline visible exposed breast tissue where a female nipple is visible, exposed genitals or anus, and bare buttocks. '
            'Mark every target in the frame, including targets outside suggested boxes. '
            'Exclude male chests, cleavage, underwear, covered anatomy, faces and unrelated skin. '
            'Choose unsure for anatomy that cannot be judged.')
DEFINITION_SHA256=hashlib.sha256(DEFINITION.encode()).hexdigest()


def validated_strokes(strokes):
    if not isinstance(strokes,list) or len(strokes)>1000:raise ValueError('Too many mask strokes')
    result=[];total=0
    for stroke in strokes:
        if not isinstance(stroke,dict) or stroke.get('mode') not in ('paint','erase'):
            raise ValueError('Choose paint or erase')
        radius=stroke.get('radius')
        if type(radius) not in (float,int) or not math.isfinite(radius) or not .0001<=radius<=.2:
            raise ValueError('Invalid brush size')
        points=stroke.get('points')
        if not isinstance(points,list) or not points:raise ValueError('A stroke needs points')
        total+=len(points)
        if total>20000:raise ValueError('Too many brush points')
        clean=[]
        for point in points:
            if not isinstance(point,list) or len(point)!=2 or any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in point):
                raise ValueError('Brush points must lie inside the image')
            clean.append(list(point))
        result.append(dict(mode=stroke['mode'],radius=float(radius),points=clean))
    return result


def render_strokes(size,strokes,base=None):
    w,h=size
    if type(w) is not int or type(h) is not int or not 1<=w<=2048 or not 1<=h<=2048:
        raise ValueError('Unsupported mask dimensions')
    mask=Image.new('L',size) if base is None else base.convert('L').copy()
    if mask.size!=size:raise ValueError('Draft mask shape differs from frame')
    draw=ImageDraw.Draw(mask)
    for stroke in validated_strokes(strokes):
        radius=max(.5,stroke['radius']*min(w,h));points=[(x*(w-1),y*(h-1)) for x,y in stroke['points']]
        color=255 if stroke['mode']=='paint' else 0
        if len(points)>1:draw.line(points,fill=color,width=max(1,round(radius*2)),joint='curve')
        for x,y in points:draw.ellipse((x-radius,y-radius,x+radius,y+radius),fill=color)
    return mask.point(lambda p:255 if p>=128 else 0)


def read_label(folder,data,row):
    label=read_json(folder/'labels'/f"{row['id']}.json",{})
    if not label:return None
    if label.get('definition_sha256')!=DEFINITION_SHA256:raise ValueError('Mask definition changed; labels need explicit review')
    if label.get('dataset_sha256')!=file_hash(folder/'dataset.json') or label.get('image_sha256')!=row['image_sha256']:
        raise ValueError('Mask labels no longer match this dataset')
    if label['status']=='reviewed' and file_hash(folder/f"{row['id']}-gold.png")!=label.get('mask_sha256'):
        raise ValueError('Reviewed mask file changed')
    exposure=read_json(folder/'exposure'/f"{row['id']}.json",{})
    return dict(label,drafts_shown=bool(label['drafts_shown'] or exposure))


def router(root):
    api=APIRouter(prefix='/api/segmentation')
    def dataset(name):
        if not NAME.fullmatch(name):raise HTTPException(400,'Invalid mask project name')
        folder=Path(root)/'segmentation'/name;data=read_json(folder/'dataset.json',{})
        if data.get('version')!=1 or not data.get('rows'):raise HTTPException(404,'Mask project not found')
        return folder,data
    def sample(name,rid):
        folder,data=dataset(name)
        if not ROW.fullmatch(rid):raise HTTPException(404,'Mask frame not found')
        row=next((r for r in data['rows'] if r['id']==rid),None)
        if row is None:raise HTTPException(404,'Mask frame not found')
        return folder,data,row
    def exposure(folder,row):
        write_json(folder/'exposure'/f"{row['id']}.json",dict(dataset_sha256=file_hash(folder/'dataset.json'),model_draft_seen=True))
    @api.get('')
    def projects():
        results=[]
        for path in sorted((Path(root)/'segmentation').glob('mask-*/dataset.json')):
            folder,data=dataset(path.parent.name)
            results.append(dict(name=folder.name,split=data['split'],total=len(data['rows']),
                supersedes=list(data.get('source_datasets',{})),
                reviewed=sum(bool(read_json(folder/'labels'/f"{r['id']}.json",{})) for r in data['rows'])))
        return results
    @api.get('/{name}')
    def get(name):
        folder,data=dataset(name)
        return dict(name=name,split=data['split'],dataset_sha256=file_hash(folder/'dataset.json'),
            scope=data['evaluation_scope'],definition=DEFINITION,rows=[{k:r[k] for k in ('id','display','time','width','height','split')} for r in data['rows']])
    @api.get('/{name}/labels/{rid}')
    def labels(name,rid):
        folder,data,row=sample(name,rid)
        try:label=read_label(folder,data,row)
        except ValueError as e:raise HTTPException(409,str(e))
        return dict(label=label,revision=file_hash(folder/'labels'/f'{rid}.json'))
    @api.get('/{name}/images/{rid}/{kind}')
    def image(name,rid,kind):
        with LOCK:
            folder,data,row=sample(name,rid)
            if kind not in ('frame','draft','gold'):raise HTTPException(404,'Mask image not found')
            filename=rid+'.jpg' if kind=='frame' else rid+'-'+kind+'.png';path=folder/filename
            if not path.is_file():raise HTTPException(404,'Mask image not found')
            expected=row['image_sha256'] if kind=='frame' else row['draft_sha256'] if kind=='draft' else (read_json(folder/'labels'/f'{rid}.json',{}).get('mask_sha256'))
            if not expected or file_hash(path)!=expected:raise HTTPException(409,'Mask image changed; review needs reconciliation')
            if kind=='draft':exposure(folder,row)
            return FileResponse(path,media_type='image/jpeg' if kind=='frame' else 'image/png')
    @api.put('/{name}/labels/{rid}')
    def save(name,rid,payload:dict=Body(...)):
        with LOCK:
            folder,data,row=sample(name,rid);labelpath=folder/'labels'/f'{rid}.json';identity=file_hash(folder/'dataset.json')
            if payload.get('dataset_sha256')!=identity or payload.get('revision')!=file_hash(labelpath):
                raise HTTPException(409,'Mask labels changed; reload before saving')
            if file_hash(folder/f'{rid}.jpg')!=row['image_sha256']:raise HTTPException(409,'Source frame changed')
            status=payload.get('status');included=payload.get('draft_included')
            if status not in ('reviewed','unsure') or type(included) is not bool or 'strokes' not in payload:
                raise HTTPException(400,'Choose reviewed or unsure and supply explicit mask strokes')
            try:
                strokes=validated_strokes(payload['strokes']);base=None
                if included:
                    path=folder/f'{rid}-draft.png'
                    if file_hash(path)!=row['draft_sha256']:raise ValueError('Draft mask changed')
                    with Image.open(path) as image:base=image.copy()
                mask=render_strokes((row['width'],row['height']),strokes,base)
            except ValueError as e:raise HTTPException(400,str(e))
            if included:exposure(folder,row)
            if status=='reviewed':
                destination=folder/f'{rid}-gold.png';temporary=destination.with_suffix('.part.png')
                mask.save(temporary);temporary.replace(destination);mask_sha=file_hash(destination)
            else:mask_sha=None
            label=dict(id=rid,status=status,labeler='human',split=row['split'],dataset_sha256=identity,
                definition_sha256=DEFINITION_SHA256,
                image_sha256=row['image_sha256'],mask_sha256=mask_sha,strokes=strokes,draft_included=included,
                drafts_shown=bool(read_json(folder/'exposure'/f'{rid}.json',{})),positive_pixels=int(np.count_nonzero(np.asarray(mask))) if status=='reviewed' else None)
            write_json(labelpath,label)
            return dict(label=label,revision=file_hash(labelpath))
    return api
