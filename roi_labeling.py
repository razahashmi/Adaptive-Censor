"""Saved local human labels for detector crops; no download-and-copy workflow."""
from pathlib import Path
import re
import threading
import time

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import FileResponse
from labeling import NAME, read_json, write_json
from provenance import file_hash

IMAGE=re.compile(r'^roi-\d{4,6}(-body)?\.jpg$')
_lock=threading.RLock()


def router(root):
    api=APIRouter()
    def dataset(name):
        if not NAME.fullmatch(name):raise HTTPException(400,'Invalid crop project name')
        folder=Path(root)/'roi'/name;path=folder/'dataset.json'
        if not path.is_file():raise HTTPException(404,'Crop project not found')
        data=read_json(path)
        if data.get('review_api')!=1:raise HTTPException(404,'This gallery uses the legacy file review workflow')
        if any(r.get('split')!='train' for r in data['rows']):raise HTTPException(400,'Crop calibration is training-only')
        identity=file_hash(path);reviewed=read_json(folder/'reviewed.json',{})
        if reviewed and reviewed.get('dataset_sha256')!=identity:
            raise HTTPException(409,'Crop project changed since review; labels need explicit reconciliation')
        return folder,data,reviewed or dict(data,dataset_sha256=identity),identity
    @api.get('/api/roi-projects')
    def projects():
        result=[]
        for path in sorted((Path(root)/'roi').glob('*/dataset.json')):
            if read_json(path).get('review_api')!=1:continue
            folder,data,reviewed,identity=dataset(path.parent.name)
            result.append(dict(name=folder.name,total=len(data['rows']),reviewed=sum(
                r.get('labeler')=='human' and r.get('reviewed',False) for r in reviewed['rows'])))
        return result
    @api.get('/api/roi/{name}')
    def get(name):
        folder,data,reviewed,identity=dataset(name)
        keys=('id','title','time','detector_class','label','labeler','reviewed')
        return dict(name=name,dataset_sha256=identity,revision=file_hash(folder/'reviewed.json'),
                    feature_views=data.get('feature_views',['local']),rows=[{k:r.get(k) for k in keys} for r in reviewed['rows']])
    @api.get('/api/roi/{name}/images/{filename}')
    def image(name,filename):
        folder,data,_,_=dataset(name)
        if not IMAGE.fullmatch(filename) or filename.removesuffix('-body.jpg').removesuffix('.jpg') not in {r['id'] for r in data['rows']}:
            raise HTTPException(404,'Crop image not found')
        path=folder/filename
        if not path.is_file():raise HTTPException(404,'Crop image not found')
        if data.get('images_sha256') and data['images_sha256'].get(filename)!=file_hash(path):
            raise HTTPException(409,'Crop image changed; review needs explicit reconciliation')
        return FileResponse(path,media_type='image/jpeg')
    @api.put('/api/roi/{name}/labels/{crop_id}')
    def save(name,crop_id,payload:dict=Body(...)):
        with _lock:
            folder,data,reviewed,identity=dataset(name)
            if payload.get('dataset_sha256')!=identity or payload.get('revision')!=file_hash(folder/'reviewed.json'):
                raise HTTPException(409,'Saved crop labels changed; reload before saving')
            if 'label' not in payload:raise HTTPException(400,'Choose exposed, covered or unsure')
            value=payload['label']
            if value is not None and (type(value) is not int or value not in (0,1)):
                raise HTTPException(400,'Crop label must be exposed, covered or unsure')
            row=next((r for r in reviewed['rows'] if r['id']==crop_id),None)
            if row is None:raise HTTPException(404,'Crop not found')
            row.update(label=value,labeler='human',reviewed=True,updated=time.strftime('%Y-%m-%dT%H:%M:%S'))
            write_json(folder/'reviewed.json',reviewed)
            return dict(row={k:row.get(k) for k in ('id','label','labeler','reviewed')},revision=file_hash(folder/'reviewed.json'))
    return api
