"""Experimental anatomy veto using human-labeled exposed/covered ROI embeddings.

It only turns a supported proposal into review; it never creates positives.
Prepare unlabeled crops, review them, then train with title-held-out thresholds.
"""
import argparse
import hashlib
import html
import json
from pathlib import Path
import numpy as np
from labeling import project_dir, read_json, write_json
from learned import thresholds
from model_setup import prepare_environment
from provenance import file_hash
from frame_cache import BoundedCache


def validate_head(head):
    if head.get('feature_kind')=='detector-scores':
        from anatomy_runtime import LABELS
        values=[np.asarray(head[key],dtype=np.float64) for key in ('mean','scale','coef')]
        if (head.get('class_labels')!=list(LABELS) or head.get('anatomy_model')!='640m'
                or any(v.shape!=(len(LABELS),) or not np.isfinite(v).all() for v in values)
                or np.any(values[1]<=0) or not np.isfinite(float(head['intercept']))
                or head.get('decision_threshold') is None or not np.isfinite(float(head['decision_threshold']))
                or head.get('negative_confirmations') not in (3,5)
                or not head.get('supported_classes')
                or any(c not in LABELS for c in head['supported_classes'])):
            raise ValueError('Invalid detector-score anatomy head')
        return
    if head.get('feature_kind','clip')!='clip':
        raise ValueError('Unknown anatomy feature kind')
    views=head.get('feature_views',['local'])
    if not isinstance(views,list) or not views or len(set(views))!=len(views) or any(v not in ('local','body') for v in views):
        raise ValueError('Invalid anatomy feature views')
    values=[np.asarray(head[key],dtype=np.float64) for key in ('mean','scale','coef')]
    if any(v.shape!=(512*len(views),) or not np.isfinite(v).all() for v in values) or np.any(values[1]<=0):
        raise ValueError('Invalid anatomy feature parameters')
    if not np.isfinite(float(head['intercept'])) or (head['decision_threshold'] is not None and not np.isfinite(float(head['decision_threshold']))):
        raise ValueError('Invalid anatomy threshold/intercept')
    confirmations=head.get('negative_confirmations',1)
    if type(confirmations) is not int or confirmations not in (1,3,5):
        raise ValueError('Invalid anatomy negative confirmations')
    if 'view_thresholds' in head:
        thresholds=head['view_thresholds'];weights=head.get('view_weights',{})
        if (not isinstance(thresholds,dict) or not isinstance(weights,dict) or set(thresholds)!=set(views)
                or set(weights)!=set(views) or head.get('head_type')!='semantic-prototype' or float(head['intercept'])!=0
                or any(not np.isfinite(float(thresholds[v])) or not np.isfinite(float(weights[v])) or float(weights[v])<=0 for v in views)):
            raise ValueError('Invalid anatomy view thresholds/weights')
    if 'detector_head' in head:
        detector=head['detector_head']
        if (not isinstance(detector,dict) or detector.get('feature_kind')!='detector-scores'
                or 'body' not in views or 'view_thresholds' not in head
                or detector.get('anatomy_identity')!=head.get('anatomy_identity')
                or detector.get('supported_classes')!=head.get('supported_classes')
                or head.get('negative_confirmations') not in (3,5)):
            raise ValueError('Invalid combined anatomy head')
        validate_head(detector)
    if 'secondary_head' in head:
        secondary=head['secondary_head']
        if (not isinstance(secondary,dict) or secondary.get('feature_kind','clip')!='clip'
                or any(k in secondary for k in ('secondary_head','detector_head'))
                or secondary.get('feature_views',['local'])!=views
                or any(secondary.get(k)!=head.get(k) for k in ('clip_identity','anatomy_identity','anatomy_model','supported_classes','negative_confirmations'))
                or head.get('negative_confirmations') not in (3,5)):
            raise ValueError('Invalid secondary semantic anatomy head')
        validate_head(secondary)


def crop(pixels, box, margin=.25):
    height, width = pixels.shape[:2]
    x, y, w, h = box
    left, top = max(0,int((x-w*margin)*width)), max(0,int((y-h*margin)*height))
    right, bottom = min(width,int(np.ceil((x+w*(1+margin))*width))), min(height,int(np.ceil((y+h*(1+margin))*height)))
    if right <= left or bottom <= top:
        raise ValueError('ROI does not intersect the frame')
    return pixels[top:bottom,left:right]


def body_crop(pixels, box):
    """A square around the proposal, with enough context for torso/clothing cues."""
    height, width = pixels.shape[:2]
    x, y, w, h = box
    size = min(width, height, max(width*.5, 3*max(w*width, h*height)))
    left = int(np.clip((x+w/2)*width-size/2, 0, width-size))
    top = int(np.clip((y+h/2)*height-size/2, 0, height-size))
    return pixels[top:top+int(size), left:left+int(size)]


class AnatomyGate:
    def __init__(self, models, path):
        self.models, self.head = models, read_json(path)
        self.embedding_cache=BoundedCache(32*1024**2)
        if self.head.get('feature_kind','clip')=='clip' and self.head.get('clip_identity') != file_hash(models.root/'clip/onnx/vision_model_quantized.onnx'):
            raise RuntimeError('ROI CLIP weights changed; retrain or disable the gate')
        validate_head(self.head)
        if self.head.get('anatomy_model')=='640m' and self.head.get('anatomy_identity')!=file_hash(models.root/'scene-verifier/640m.onnx'):
            raise RuntimeError('ROI anatomy weights changed; retrain or disable the gate')

    def refine(self, frames, candidates):
        allowed = self.head.get('supported_classes')
        selected = [(n,item) for n,item in enumerate(candidates) if len(item[3]) >= 3
                    and (allowed is None or item[2]['class'] in allowed)]
        if not selected:
            return candidates
        if self.head.get('feature_kind')=='detector-scores':
            negative=self._detector_negatives(selected,self.head)
            return self._apply(frames,candidates,selected,negative)
        height, width = frames[0][1].shape[:2]
        pictures = []
        views = self.head.get('feature_views', ['local'])
        for _, item in selected:
            index, _, hit, *_ = item
            x,y,w,h = hit['box']
            box = [x/width,y/height,w/width,h/height]
            for view in views:
                if view not in ('local','body'):
                    raise ValueError(f'Unknown anatomy feature view: {view}')
                picture = crop(frames[index][1], box) if view=='local' else body_crop(frames[index][1], box)
                pictures.append((frames[index][0], picture))
        keys=[(picture.shape,hashlib.sha256(picture.tobytes()).hexdigest()) for _,picture in pictures]
        def encode(indices):
            values=self.models.clip_embeddings([pictures[i] for i in indices],'cpu',fit='squash')
            return [row.copy() for row in values]
        embeddings=np.stack(self.embedding_cache.get_many(keys,encode)).reshape(len(selected), -1)
        h = self.head
        negative,view_negative=self._semantic_negatives(embeddings,h)
        if 'detector_head' in h:
            negative |= self._detector_negatives(selected,h['detector_head']) & view_negative['body']
        routes=[negative]
        if 'secondary_head' in h:
            routes.append(self._semantic_negatives(embeddings,h['secondary_head'])[0])
        return self._apply(frames,candidates,selected,np.stack(routes))

    @staticmethod
    def _semantic_negatives(embeddings,h):
        views=h.get('feature_views',['local'])
        standardized=(embeddings-np.asarray(h['mean']))/np.asarray(h['scale'])
        scores = standardized@np.asarray(h['coef'])+h['intercept']
        negative=np.ones(len(embeddings),dtype=bool) if h['decision_threshold'] is None else scores<h['decision_threshold']
        # A veto needs agreement from the local evidence and its body context.
        # A positive view preserves the detector's accepted evidence.
        view_negative={}
        if 'view_thresholds' in h:
            for column,view in enumerate(views):
                begin,end=column*512,(column+1)*512
                view_scores=standardized[:,begin:end]@(np.asarray(h['coef'][begin:end])/h['view_weights'][view])
                view_negative[view]=view_scores<h['view_thresholds'][view]
                negative &= view_negative[view]
        return negative,view_negative

    @staticmethod
    def _detector_negatives(selected,head):
        from anatomy_runtime import LABELS
        negative=np.zeros(len(selected),dtype=bool)
        for row,(_,item) in enumerate(selected):
            # Only full-picture scores were calibrated. A contextual crop,
            # missing evidence or a positive duplicate protects that frame.
            scores=np.asarray(item[2].get('class_scores',[]),dtype=np.float64)
            if (item[6]!='full picture' or scores.shape!=(len(LABELS),)
                    or not np.isfinite(scores).all() or np.any((scores<0)|(scores>1))):
                continue
            expected=LABELS.index(item[2]['class'])
            if int(scores.argmax())!=expected or abs(scores[expected]-item[2]['score'])>1e-6:
                continue
            value=((scores-np.asarray(head['mean']))/np.asarray(head['scale']))@np.asarray(head['coef'])+head['intercept']
            negative[row]=value<head['decision_threshold']
        return negative

    def _apply(self,frames,candidates,selected,negative):
        required=self.head.get('negative_confirmations',1)
        routes=np.atleast_2d(negative)
        def confirmed(row,evidence):
            if not evidence[row]:return False
            if required==1:return True
            _,center=selected[row];index,shot,hit,*_=center
            from analysis_engine import box_iou
            neighbors={}
            for other,(_,candidate) in enumerate(selected):
                j,s,proposal,*_=candidate
                if s==shot and abs(frames[j][0]-frames[index][0])<=.25+1e-6 and proposal['class']==hit['class'] and box_iou(proposal['box'],hit['box'])>=.3:
                    neighbors.setdefault(j,[]).append(bool(evidence[other]))
            nearest=sorted(neighbors,key=lambda j:(abs(frames[j][0]-frames[index][0]),j))[:5]
            # Duplicate full/crop proposals cannot supply independent votes.
            return sum(all(neighbors[j]) for j in nearest)>=required
        result = list(candidates)
        for row,(n,item) in enumerate(selected):
            # Each contrast independently needs enough distinct negative frames;
            # unrelated weak votes from different checks cannot be pooled.
            if any(confirmed(row,evidence) for evidence in routes):
                index,shot,hit,_,proposal_score,proposal_time,refinement = item
                result[n] = (index,shot,dict(hit,roi_rejected=True),[],proposal_score,proposal_time,refinement+' + ROI review')
        return result


def prepare(root, projects, run, name, limit, body_context=False, anatomy_model='640m'):
    from analysis_engine import Models, native_frames
    from PIL import Image
    folder = root/'roi'/name
    if folder.exists():
        raise FileExistsError(f'{folder} exists; choose a new name')
    folder.mkdir(parents=True)
    models, rows, pictures = Models(), [], []
    from detectors import NUDITY_CLASSES
    try:
        for project_name in projects:
            project = project_dir(root,project_name)
            for window in read_json(project/'project.json')['windows']:
                if window['split'] != 'train':
                    continue
                prediction = read_json(project/'predictions'/run/f"{window['id']}.json", {})
                previous = -10
                for event in prediction.get('events',[]):
                    t = event.get('sample_time',(event['start']+event['end'])/2)
                    if event['category'] != 'nudity' or not event.get('region') or t-previous < 1:
                        continue
                    frames = native_frames(project/'clips'/f"{window['id']}.mp4",max(0,t-.05),min(window['clip_duration'],t+.08),30)
                    _, pixels = min(frames,key=lambda f:abs(f[0]-t))
                    height,width=pixels.shape[:2]
                    for hit in models.nude('auto',anatomy_model).detect(pixels[:,:,::-1].copy()):
                        if hit['class'] not in NUDITY_CLASSES or hit['score']<.25:
                            continue
                        x,y,w,h=hit['box'];region=[x/width,y/height,w/width,h/height]
                        picture=crop(pixels,region)
                        if min(picture.shape[:2])<16:
                            continue
                        row = dict(id=f'roi-{len(rows):04}',title=window['title'],project=project_name,window=window['id'],time=t,
                                   region=region,detector_class=hit['class'],detector_score=float(hit['score']),
                                   source_identity=file_hash(project/'clips'/f"{window['id']}.mp4"),label=None,labeler=None,split='train')
                        Image.fromarray(picture).save(folder/f"{row['id']}.jpg",quality=95)
                        rows.append(row);pictures.append((t,picture))
                        if body_context:
                            context=body_crop(pixels,region)
                            pictures.append((t,context));Image.fromarray(context).save(folder/f"{row['id']}-body.jpg",quality=95)
                        if len(rows)>=limit:break
                    previous=t
                    if len(rows)>=limit: break
                if len(rows)>=limit: break
            if len(rows)>=limit: break
        if not pictures:
            raise ValueError('No regional proposals found in the training runs')
        features = models.clip_embeddings(pictures,'cpu',fit='squash').reshape(len(rows),-1)
        np.savez_compressed(folder/'features.npz',embeddings=features)
        data = dict(rows=rows,clip_identity=file_hash(root/'clip/onnx/vision_model_quantized.onnx'),
                    feature_views=['local','body'] if body_context else ['local'],anatomy_model=anatomy_model,
                    anatomy_identity=file_hash(root/'scene-verifier/640m.onnx') if anatomy_model=='640m' else None,
                    crop_method='Raw detector box with 25% margin; matches runtime',
                    feature_identity=file_hash(folder/'features.npz'),purpose='Human ROI labels required; exposed=1, covered/look-alike=0, unsure=null')
        write_json(folder/'dataset.json',data)
        cards = ''.join(f'<article><img src="{r["id"]}.jpg">'+(f'<img src="{r["id"]}-body.jpg">' if body_context else '')+f'<p>{html.escape(r["title"])} / {r["time"]:.2f}s</p><select data-id="{r["id"]}"><option value="">Unsure / unreviewed</option><option value="1">Exposed target anatomy</option><option value="0">Covered / look-alike / ordinary skin</option></select></article>' for r in rows)
        payload = json.dumps(data).replace('</','<\\/')
        page = '<!doctype html><meta charset="utf-8"><title>Anatomy crop review</title><style>body{background:#142030;color:#eef;font:16px system-ui;padding:24px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:18px}img{max-width:100%;height:220px;object-fit:contain}article{background:#233246;padding:14px}button,select{padding:10px}</style><h1>Independent anatomy crop review</h1><p>Targets: female nipples, genitals, anus or bare buttocks. Male chests, shoulders, cleavage, clothing and underwear are negatives. Leave uncertain crops unmarked.</p><button id="save">Save reviewed.json</button><main>'+cards+'</main><script>const data='+payload+'; document.getElementById("save").onclick=()=>{for(const r of data.rows){const v=document.querySelector(`[data-id="${r.id}"]`).value;r.label=v===""?null:Number(v);r.labeler=v===""?null:"human";}const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([JSON.stringify(data,null,1)],{type:"application/json"}));a.download="reviewed.json";a.click();URL.revokeObjectURL(a.href);};</script>'
        (folder/'review.html').write_text(page,encoding='utf-8')
        print(f'{len(rows)} unlabeled crops: {folder / "review.html"}')
    finally:
        models.close()


def train(path, output, allow_ai_labels=False, protect_recall=True):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from select_windows import normalize_title
    data = read_json(path)
    if file_hash(path.parent/'features.npz') != data['feature_identity']:
        raise ValueError('ROI feature file changed')
    classes = data.get('model_classes')
    selected = [(i,r) for i,r in enumerate(data['rows']) if r.get('label') in (0,1)
                and (r.get('labeler')=='human' or (allow_ai_labels and r.get('labeler') in ('codex','claude')))
                and r.get('split')=='train' and (classes is None or r.get('detector_class') in classes)]
    if len(selected) < 40:
        raise ValueError('Review at least 40 crops with both classes across at least 5 titles')
    with np.load(path.parent/'features.npz') as cached:
        X = cached['embeddings'][[i for i,_ in selected]].astype(np.float32)
    y = np.array([r['label'] for _,r in selected]); groups = np.array([normalize_title(r['title']) for _,r in selected])
    if len(set(groups)) < 5 or any(len(set(groups[y==v])) < 3 for v in (0,1)):
        raise ValueError('Need 5 titles, each class present in at least 3 titles')
    oof = np.zeros(len(y))
    for fit,held in GroupKFold(5).split(X,y,groups):
        if len(set(y[fit])) < 2: raise ValueError('Training fold contains one class')
        model = make_pipeline(StandardScaler(),LogisticRegression(C=.01,class_weight='balanced',max_iter=3000)).fit(X[fit],y[fit])
        oof[held] = model.decision_function(X[held])
    candidate, decision = thresholds(oof,y)
    # A veto cannot recover missed anatomy. Prefer the high-recall operating
    # point rather than requiring every genuine region to score as a decision.
    operating = candidate if protect_recall else decision
    model = make_pipeline(StandardScaler(),LogisticRegression(C=.01,class_weight='balanced',max_iter=3000)).fit(X,y)
    scaler, head = model.steps[0][1], model.steps[1][1]
    result = dict(mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),coef=head.coef_[0].tolist(),intercept=float(head.intercept_[0]),
                  decision_threshold=operating,clip_identity=data['clip_identity'],dataset_identity=file_hash(path),
                  crops=len(y),titles=sorted(set(groups)),experimental=True,
                  operating_point='protect-recall' if protect_recall else 'precision',
                  labelers={name:sum(r['labeler']==name for _,r in selected) for name in sorted({r['labeler'] for _,r in selected})},
                  ai_labels_allowed=allow_ai_labels,
                  oof_recall=float(((oof>=operating)&(y==1)).sum()/max(1,(y==1).sum())) if operating is not None else 0.,
                  oof_negative_rejection=float(((oof<operating)&(y==0)).sum()/max(1,(y==0).sum())) if operating is not None else 1.)
    result['feature_views'] = data.get('feature_views', ['local'])
    for key in ('anatomy_model','anatomy_identity'):
        if key in data:result[key]=data[key]
    if any('detector_class' in r for _,r in selected):
        support = {}
        for _,row in selected:
            if row['label']==1:
                support.setdefault(row['detector_class'],set()).add(normalize_title(row['title']))
        result['class_positive_titles'] = {key:sorted(value) for key,value in support.items()}
        result['supported_classes'] = sorted(key for key,value in support.items() if len(value)>=3)
    if output.exists(): raise FileExistsError('Choose a new ROI model name')
    write_json(output,result)


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare'); a.add_argument('--projects',nargs='+',required=True); a.add_argument('--run',default='rev9'); a.add_argument('--name',required=True); a.add_argument('--limit',type=int,default=80)
    a.add_argument('--body-context',action='store_true');a.add_argument('--anatomy-model',choices=['320n','640m'],default='640m')
    t=sub.add_parser('train'); t.add_argument('--labels',type=Path,required=True); t.add_argument('--model',required=True)
    t.add_argument('--allow-ai-labels',action='store_true',help='Explicit AI-only experiment; never a human qualification')
    t.add_argument('--precision-operating-point',action='store_true',help='Use the more aggressive precision threshold instead of protecting recall')
    args=p.parse_args(); root=prepare_environment()
    if args.command=='prepare': prepare(root,args.projects,args.run,args.name,args.limit,args.body_context,args.anatomy_model)
    else: train(args.labels,root/'roi-models'/f'{args.model}.json',args.allow_ai_labels,not args.precision_operating_point)


if __name__=='__main__': main()
