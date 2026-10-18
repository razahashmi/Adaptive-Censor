"""Fit spatial probes separately from immutable runtime token extraction.

Bfloat16 accelerates fitting only. Tuning and sealed calibration use float32
inference, matching the deployed classifier's numeric path.
"""
import argparse
from pathlib import Path
import time
import numpy as np
from labeling import LABEL_CATEGORIES,read_json,write_json,project_dir
from provenance import file_hash
from attention_probe import load_tokens


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
            pooled,_=self.attention(self.query.expand(tokens.shape[0],-1,-1),tokens,tokens,need_weights=False)
            return self.output(pooled[:,0])
    return Probe()

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
            with torch.set_grad_enabled(learn),torch.autocast('cuda',dtype=torch.bfloat16,enabled=learn):
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
    differences=[]
    for count in (1,3,13):
        real=torch.from_numpy(np.stack([r[0] for r in samples['calibration'][:count]]).astype(np.float32))
        with torch.no_grad():expected=model(real).numpy()
        actual=session.run(None,{'tokens':real.numpy()})[0]
        if actual.shape!=expected.shape:raise ValueError('Attentive export has a static batch dimension')
        differences.append(float(np.max(np.abs(actual-expected))))
    difference=max(differences)
    if difference>1e-4:raise ValueError('Attentive ONNX export does not match training')
    deployed=[]
    for start in range(0,len(samples['calibration']),64):
        real=np.stack([r[0] for r in samples['calibration'][start:start+64]]).astype(np.float32)
        deployed.append(session.run(None,{'tokens':real})[0])
    deployed=np.concatenate(deployed)
    gpu_cpu_difference=float(np.max(np.abs(deployed-scores)))
    # Seal thresholds on the deployed numeric path itself. The fitted weights
    # remain fixed; no validation examples or post-calibration refitting.
    heads={}
    for k,c in enumerate(LABEL_CATEGORIES):
        known=y[:,k]>=0;coef=np.zeros(len(LABEL_CATEGORIES));coef[k]=1
        raw=dict(mean=[0]*len(coef),scale=[1]*len(coef),coef=coef.tolist(),intercept=0.)
        heads[c]=calibrated_head(raw,deployed[known,k],y[known,k],groups[known],events[known,k])
    artifact=dict(training_code_sha256=file_hash(Path(__file__)),kinds=[],grid=.25,temporal_offsets=[],feature_schema=FEATURE_SCHEMA,heads=heads,
                  input_layout='player-spatial-tokens-v1',attention_model=dict(file=path.name,sha256=file_hash(path),
                  encoder_sha256=file_hash(folder/'tokens.onnx')),experimental=True,validation_labels_used=False,
                  fit_titles=sorted(split['fit']),tuning_titles=sorted(split['tune']),calibration_titles=sorted(split['calibration']),
                  training_pool_windows=len(rows),best_epoch=best[1],tuning_average_precision=best[3],
                  onnx_maximum_logit_difference=difference,onnx_verified_batch_sizes=[1,3,13],calibration_refit=False,
                  calibration_inference='deployed ONNX float32',training_gpu_cpu_maximum_difference=gpu_cpu_difference,
                  qualification_note='Sealed training-title calibration; full-player human comparison required')
    write_json(destination,artifact)
    print('Saved attentive candidate',name,'identity',head_identity(artifact),'ONNX difference',difference,flush=True)
    return artifact

if __name__=='__main__':
    from model_setup import prepare_environment
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--projects',nargs='+',required=True);p.add_argument('--heads',required=True)
    p.add_argument('--epochs',type=int,default=8);a=p.parse_args();root=prepare_environment()
    train_probe(root,[project_dir(root,n) for n in a.projects],a.heads,a.epochs)
