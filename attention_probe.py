"""A compact attentive video classifier with a frozen, local V-JEPA encoder.

Keeps 8 temporal x 4 x 4 spatial tokens instead of averaging the entire picture.
Its ONNX encoder derives from the installed encoder; no model download needed.
"""
import argparse
import hashlib
from pathlib import Path
import random
import time

import numpy as np

from labeling import LABEL_CATEGORIES, is_complete, project_dir, read_json, write_json
from provenance import digest, file_hash
from frame_cache import BoundedCache


def export_tokens(root):
    import onnx
    from onnx import helper, numpy_helper, TensorProto
    folder=root/'attention-runtime';folder.mkdir(exist_ok=True)
    destination=folder/'tokens.onnx'
    installed=read_json(root/'learned-runtime/manifest.json')['models']['vjepa']
    source=root/'learned-runtime'/installed['file']
    identity=file_hash(source)
    if destination.exists():
        manifest=read_json(folder/'tokens.json')
        if manifest['source_sha256']!=identity or manifest['sha256']!=file_hash(destination):
            raise ValueError('Token encoder changed; choose a new encoder installation')
        return
    model=onnx.load(str(source))
    pooling=next(n for n in model.graph.node if 'embeddings' in n.output)
    if pooling.op_type!='ReduceMean':raise ValueError('Expected installed mean-pooled V-JEPA encoder')
    tokens=pooling.input[0]
    shape=np.array([-1,8,4,4,4,4,1024],dtype=np.int64)
    model.graph.initializer.extend([numpy_helper.from_array(shape,'probe_grid_shape'),
                                  numpy_helper.from_array(np.array([3,5],dtype=np.int64),'probe_pool_axes'),
                                  numpy_helper.from_array(np.array([-1,128,1024],dtype=np.int64),'probe_flat_shape')])
    model.graph.node.extend([
        helper.make_node('Reshape',[tokens,'probe_grid_shape'],['probe_grid']),
        helper.make_node('ReduceMean',['probe_grid','probe_pool_axes'],['probe_grid_pooled'],keepdims=0),
        helper.make_node('Reshape',['probe_grid_pooled','probe_flat_shape'],['grid_tokens'])])
    model.graph.output.extend([helper.make_tensor_value_info('grid_tokens',TensorProto.FLOAT,['batch',128,1024])])
    onnx.checker.check_model(model)
    onnx.save(model,str(destination))
    write_json(folder/'tokens.json',dict(source_sha256=identity,sha256=file_hash(destination),
               token_layout='time8-space4x4-dim1024',code_sha256=file_hash(Path(__file__))))
    print('Exported local spatial token encoder',flush=True)


class Tokens:
    def __init__(self,models):
        self.models=models; self.folder=models.root/'attention-runtime'
        manifest=read_json(self.folder/'tokens.json')
        if manifest['sha256']!=file_hash(self.folder/'tokens.onnx'):raise ValueError('Token encoder identity mismatch')
        ort,options,providers=models.ort('cuda')
        providers=[(providers[0][0],dict(providers[0][1],gpu_mem_limit=6*1024**3)),*providers[1:]]
        self.session=ort.InferenceSession(str(self.folder/'tokens.onnx'),sess_options=options,providers=providers)
        self.cache=BoundedCache(64*1024**2);self.identity=manifest['sha256']
        self.parity_maximum=0.

    def compute(self,path,centers,duration):
        from analysis_engine import native_frames
        from learned_runtime import squash_frames,vjepa_clips
        frames=native_frames(path,max(0,float(min(centers))-1.2),min(duration,float(max(centers))+1.2),8)
        times,pixels=squash_frames(frames); clips=vjepa_clips(times,pixels,centers)
        keys=[(self.identity,hashlib.sha256(c.tobytes()).hexdigest()) for c in clips]
        def infer(indices):
            values=[]
            for start in range(0,len(indices),2):
                batch=np.stack([clips[i] for i in indices[start:start+2]]).astype(np.float16)
                pooled,tokens=self.session.run(['embeddings','grid_tokens'],{'frames':batch})
                error=float(np.max(np.abs(tokens.mean(1)-pooled)))
                self.parity_maximum=max(self.parity_maximum,error)
                if error>2e-4:raise ValueError('Spatial token pooling does not match installed encoder')
                values.extend([t.astype(np.float16) for t in tokens])
            return values
        return np.stack(self.cache.get_many(keys,infer))


def chunk_tokens(encoder,path,start,end,duration):
    from learned_runtime import STEP,GRID
    centers=np.arange(np.floor(start/STEP)*STEP,np.ceil(end/STEP)*STEP+1e-9,STEP)
    centers=centers[centers<=duration]
    tokens=encoder.compute(path,centers,duration)
    times=np.arange(start+GRID/2,end,GRID)
    owners=np.abs(centers[None,:]-times[:,None]).argmin(1)
    return times,tokens,owners,centers


def token_contract(folder,w):
    from player_inputs import contract
    return dict(player=contract(folder,w,['vjepa'],()),
                encoder=file_hash(folder.parent.parent/'attention-runtime/tokens.onnx'),
                code=file_hash(Path(__file__)))


def token_path(folder,w):return folder/'attention-inputs'/f"{w['id']}.npz"


def load_tokens(folder,w):
    path=token_path(folder,w);manifest=read_json(path.with_suffix('.json'),{})
    from player_inputs import compatible_reader
    recorded=manifest.get('contract',{});current=token_contract(folder,w)
    if (manifest.get('identity')!=digest(recorded) or manifest.get('data_sha256')!=file_hash(path)
            or any(recorded.get(k)!=v for k,v in current.items() if k!='player')
            or not compatible_reader(recorded.get('player',{}),current['player'])):
        raise ValueError('Missing or stale spatial-token cache')
    with np.load(path) as data:return {k:data[k] for k in data.files}


def prepare_tokens(folder,w,encoder):
    from analysis_engine import CHUNK
    path=token_path(folder,w)
    if path.exists():return load_tokens(folder,w)
    expected=token_contract(folder,w); duration=w['clip_duration']
    times=[];samples=[];owners=[];centers=[];n=0
    for start in np.arange(0,duration,CHUNK):
        t,x,idx,c=chunk_tokens(encoder,folder/'clips'/f"{w['id']}.mp4",start,min(start+CHUNK,duration),duration)
        times.extend(t);samples.extend(x);owners.extend(idx+n);centers.extend(c);n+=len(c)
    if digest(expected)!=digest(token_contract(folder,w)):raise ValueError('Inputs changed during token extraction')
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,times=times,tokens=samples,owners=owners,centers=centers)
    write_json(path.with_suffix('.json'),dict(identity=digest(expected),contract=expected,data_sha256=file_hash(path)))
    return load_tokens(folder,w)


def build_probe(dim=1024):
    import torch
    class Probe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.query=torch.nn.Parameter(torch.randn(1,1,dim)*.02)
            self.norm=torch.nn.LayerNorm(dim)
            self.attention=torch.nn.MultiheadAttention(dim,16,batch_first=True,dropout=.1)
            self.output=torch.nn.Linear(dim,len(LABEL_CATEGORIES))
        def forward(self,tokens):
            tokens=self.norm(tokens)
            pooled,_=self.attention(self.query.expand(len(tokens),-1,-1),tokens,tokens,need_weights=False)
            return self.output(pooled[:,0])
    return Probe()


class RuntimeProbe:
    def __init__(self,models,metadata):
        import onnxruntime as ort
        self.encoder=Tokens(models)
        if metadata.get('encoder_sha256')!=self.encoder.identity:raise ValueError('Attention encoder changed')
        path=models.root/'attention-runtime'/metadata['file']
        if Path(metadata['file']).name!=metadata['file'] or file_hash(path)!=metadata['sha256']:
            raise ValueError('Attention classifier identity mismatch')
        options=ort.SessionOptions();options.intra_op_num_threads=2
        self.session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
    def matrix(self,path,start,end,duration):
        times,tokens,owners,_=chunk_tokens(self.encoder,path,start,end,duration)
        logits=self.session.run(None,{'tokens':tokens.astype(np.float32)})[0]
        return times,logits[owners]


def train_probe(root,folders,name,epochs=8):
    import torch
    from sklearn.metrics import average_precision_score
    from human_calibration import collect,partitions,event_ids,calibrated_head
    from learned import grid_labels,title_class_weights
    from select_windows import normalize_title
    from provenance import FEATURE_SCHEMA,head_identity
    destination=root/'heads'/f'{name}.json'
    if destination.exists():raise FileExistsError('Choose a new attentive head name')
    rows=collect(folders);split=partitions(rows)
    samples={s:[] for s in split}
    for folder,w,label in rows:
        title=normalize_title(w['title']);subset=next(s for s,titles in split.items() if title in titles)
        data=load_tokens(folder,w);times=data['times']
        y=np.stack([grid_labels(w,label,c,times) for c in LABEL_CATEGORIES],1)
        ids=np.stack([event_ids(w,label,c,times) for c in LABEL_CATEGORIES],1)
        for index in np.flatnonzero((y>=0).any(1)):
            samples[subset].append((data['tokens'][data['owners'][index]],y[index],title,ids[index]))
    if any(not samples[s] for s in samples):raise ValueError('Empty attentive training/tuning/calibration subset')
    torch.manual_seed(7);torch.set_num_threads(4)
    model=build_probe().cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.1)
    yfit=np.stack([y for _,y,_,_ in samples['fit']]);groups=np.array([g for _,_,g,_ in samples['fit']])
    ratios=np.clip((yfit==0).sum(0)/np.maximum((yfit==1).sum(0),1),1,50)
    pos_weight=torch.tensor(ratios,dtype=torch.float32,device='cuda')
    weights=np.ones(yfit.shape,dtype=np.float32)
    for k in range(len(LABEL_CATEGORIES)):
        known=yfit[:,k]>=0
        weights[known,k]=title_class_weights(yfit[known,k],groups[known])
    def run(subset,learn=False,epoch=0):
        model.train(learn);order=np.arange(len(samples[subset]))
        if learn:np.random.default_rng(7+epoch).shuffle(order)
        logits=[];labels=[];event_rows=[];title_rows=[]
        for start in range(0,len(order),64):
            idx=order[start:start+64];batch=[samples[subset][i] for i in idx]
            x=torch.from_numpy(np.stack([t for t,_,_,_ in batch]).astype(np.float32)).cuda()
            y=torch.from_numpy(np.stack([y for _,y,_,_ in batch])).cuda()
            with torch.set_grad_enabled(learn),torch.autocast('cuda',dtype=torch.bfloat16):
                z=model(x).float()
                if learn:
                    loss=torch.nn.functional.binary_cross_entropy_with_logits(z,y.clamp(min=0).float(),pos_weight=pos_weight,reduction='none')
                    weight=torch.from_numpy(weights[idx]).cuda()*(y>=0)
                    loss=(loss*weight).sum()/weight.sum().clamp(min=1)
            if learn:
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step()
            logits.append(z.detach().cpu().numpy());labels.extend([r[1] for r in batch])
            title_rows.extend([r[2] for r in batch]);event_rows.extend([r[3] for r in batch])
        return np.concatenate(logits),np.stack(labels),np.array(title_rows),np.stack(event_rows)
    best=None
    for epoch in range(epochs):
        started=time.perf_counter();run('fit',True,epoch)
        scores,y,_,_=run('tune')
        ap={c:float(average_precision_score(y[y[:,k]>=0,k],scores[y[:,k]>=0,k]))
            for k,c in enumerate(LABEL_CATEGORIES) if len(np.unique(y[y[:,k]>=0,k]))==2}
        mean=float(np.mean(list(ap.values()))) if ap else 0.
        print(f'attention epoch {epoch+1}/{epochs}: tuning AP {mean:.3f}, {ap}, {time.perf_counter()-started:.1f}s',flush=True)
        if best is None or mean>best[0]:best=(mean,epoch+1,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},ap)
    model.load_state_dict(best[2]);scores,y,groups,events=run('calibration')
    heads={}
    for k,c in enumerate(LABEL_CATEGORIES):
        known=y[:,k]>=0;coef=np.zeros(len(LABEL_CATEGORIES));coef[k]=1
        raw=dict(mean=[0]*len(coef),scale=[1]*len(coef),coef=coef.tolist(),intercept=0.)
        heads[c]=calibrated_head(raw,scores[known,k],y[known,k],groups[known],events[known,k])
    model=model.cpu().eval();folder=root/'attention-runtime';path=folder/f'{name}.onnx'
    if path.exists():raise FileExistsError('Never overwrite an attentive export')
    dummy=torch.from_numpy(np.stack([r[0] for r in samples['calibration'][:3]]).astype(np.float32))
    with torch.no_grad():expected=model(dummy).numpy()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        torch.onnx.export(model,(dummy,),str(path),input_names=['tokens'],output_names=['logits'],
                          dynamic_axes={'tokens':{0:'batch'},'logits':{0:'batch'}},opset_version=18,dynamo=False)
    finally:torch.backends.mha.set_fastpath_enabled(True)
    import onnxruntime as ort
    session=ort.InferenceSession(str(path),providers=['CPUExecutionProvider'])
    actual=session.run(None,{'tokens':dummy.numpy()})[0]
    difference=float(np.max(np.abs(actual-expected)))
    if difference>1e-4:raise ValueError('Attentive ONNX export does not match training')
    artifact=dict(kinds=[],grid=.25,temporal_offsets=[],feature_schema=FEATURE_SCHEMA,heads=heads,
                  input_layout='player-spatial-tokens-v1',attention_model=dict(file=path.name,sha256=file_hash(path),
                  encoder_sha256=file_hash(folder/'tokens.onnx')),experimental=True,validation_labels_used=False,
                  fit_titles=sorted(split['fit']),tuning_titles=sorted(split['tune']),calibration_titles=sorted(split['calibration']),
                  training_pool_windows=len(rows),best_epoch=best[1],tuning_average_precision=best[3],
                  onnx_maximum_logit_difference=difference,calibration_refit=False,
                  qualification_note='Sealed training-title calibration; full-player human comparison required')
    write_json(destination,artifact)
    print('Saved attentive candidate',name,'identity',head_identity(artifact),'ONNX difference',difference,flush=True)
    return artifact


def main():
    from analysis_engine import Models
    from model_setup import prepare_environment
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('export-tokens')
    a=sub.add_parser('prepare');a.add_argument('--projects',nargs='+',required=True)
    t=sub.add_parser('train');t.add_argument('--projects',nargs='+',required=True);t.add_argument('--heads',required=True)
    t.add_argument('--epochs',type=int,default=8)
    args=p.parse_args();root=prepare_environment()
    if args.command=='export-tokens':return export_tokens(root)
    if args.command=='train':return train_probe(root,[project_dir(root,n) for n in args.projects],args.heads,args.epochs)
    models=Models(evaluate_unpromoted=True);encoder=Tokens(models);started=time.perf_counter()
    try:
        for name in args.projects:
            folder=project_dir(root,name)
            windows=[w for w in read_json(folder/'project.json')['windows']
                     if w['split']=='train' and is_complete(read_json(folder/'labels'/f"{w['id']}.json",{}))]
            for i,w in enumerate(windows,1):
                prepare_tokens(folder,w,encoder)
                print(f'{name}: {i}/{len(windows)} spatial tokens ({time.perf_counter()-started:.1f}s)',flush=True)
        print('Token-to-pooled parity maximum:',encoder.parity_maximum,flush=True)
    finally:models.close()


if __name__=='__main__':main()
