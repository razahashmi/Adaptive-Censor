"""Train a cheap anatomy veto from the detector's existing class-score vector.

Training rows must have independently reviewed region labels and be title-held
out for threshold selection. This calibration does not qualify viewing accuracy.
"""
import argparse
from pathlib import Path
import numpy as np
from anatomy_runtime import LABELS
from labeling import read_json,write_json
from model_setup import prepare_environment
from provenance import file_hash
from select_windows import normalize_title


def prepare(path,output,root):
    """Reproduce reviewed raw training proposals; keep their independent labels."""
    from analysis_engine import Models,native_frames,box_iou
    from labeling import project_dir
    data=read_json(path)
    if (data.get('anatomy_model')!='640m'
            or data.get('anatomy_identity')!=file_hash(root/'scene-verifier/640m.onnx')
            or not data.get('crop_method','').lower().startswith('raw detector box')):
        raise ValueError('Use reviewed raw training crops from the current detailed detector')
    if output.exists():raise FileExistsError('Choose a new feature output')
    models=Models();records=[]
    try:
        model=models.nude('auto','640m')
        for row in data['rows']:
            if row.get('split')!='train' or row.get('label') not in (0,1):continue
            source=project_dir(root,row['project'])/'clips'/f"{row['window']}.mp4"
            if not row.get('source_identity') or file_hash(source)!=row['source_identity']:
                raise ValueError('Reviewed training footage changed')
            frames=native_frames(source,max(0,row['time']-.05),row['time']+.08,30)
            t,pixels=min(frames,key=lambda f:abs(f[0]-row['time']))
            height,width=pixels.shape[:2]
            box=[v*s for v,s in zip(row['region'],[width,height,width,height])]
            matches=[h for h in model.detect(pixels[:,:,::-1].copy(),include_class_scores=True)
                     if h['class']==row['detector_class'] and box_iou(h['box'],box)>=.3]
            if not matches:raise ValueError(f"Cannot reproduce reviewed region {row['id']}")
            hit=max(matches,key=lambda h:box_iou(h['box'],box))
            records.append(dict(row,scores=hit['class_scores'],matched_iou=float(box_iou(hit['box'],box)),frame_time=float(t)))
        write_json(output,dict(labels=list(LABELS),anatomy_identity=data['anatomy_identity'],
                               review_identity=file_hash(path),records=records))
    finally:models.close()
    return len(records)


def calibrate(path,output,root,allow_ai_labels=False):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneGroupOut
    data=read_json(path)
    anatomy=root/'scene-verifier/640m.onnx'
    if data.get('labels')!=list(LABELS) or data.get('anatomy_identity')!=file_hash(anatomy):
        raise ValueError('Detector classes or weights changed')
    rows=[r for r in data['records'] if r.get('split')=='train' and r.get('label') in (0,1)
          and r.get('detector_class')=='FEMALE_BREAST_EXPOSED'
          and (r.get('labeler')=='human' or (allow_ai_labels and r.get('labeler') in ('codex','claude')))]
    X=np.asarray([r['scores'] for r in rows],dtype=np.float64)
    y=np.asarray([r['label'] for r in rows]);groups=np.asarray([normalize_title(r['title']) for r in rows])
    if (len(rows)<40 or X.shape!=(len(rows),len(LABELS)) or not np.isfinite(X).all()
            or np.any((X<0)|(X>1)) or len(set(groups))<5
            or any(len(set(groups[y==v]))<3 for v in (0,1))):
        raise ValueError('Need finite scores, 40 reviewed regions, 5 titles and 3 titles per class')
    if np.any(X.argmax(axis=1)!=LABELS.index('FEMALE_BREAST_EXPOSED')):
        raise ValueError('Score vector does not match the reviewed detector class')
    weights=np.asarray([1/sum(groups==title) for title in groups])*len(groups)/len(set(groups))
    def fit(indices):
        return make_pipeline(StandardScaler(),LogisticRegression(C=.01,class_weight='balanced',max_iter=2000)).fit(
            X[indices],y[indices],logisticregression__sample_weight=weights[indices])
    oof=np.zeros(len(y))
    for train,held in LeaveOneGroupOut().split(X,y,groups):
        oof[held]=fit(train).decision_function(X[held])
    # All training positives must survive independently held-title decisions.
    threshold=float(oof[y==1].min())-1e-6
    model=fit(np.arange(len(y)));scaler,head=model.steps[0][1],model.steps[1][1]
    result=dict(feature_kind='detector-scores',class_labels=list(LABELS),
        mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),coef=head.coef_[0].tolist(),intercept=float(head.intercept_[0]),
        decision_threshold=threshold,negative_confirmations=3,supported_classes=['FEMALE_BREAST_EXPOSED'],
        anatomy_model='640m',anatomy_identity=data['anatomy_identity'],dataset_identity=file_hash(path),
        crops=len(rows),titles=sorted(set(groups)),class_positive_titles={'FEMALE_BREAST_EXPOSED':sorted(set(groups[y==1]))},
        experimental=True,ai_labels_allowed=allow_ai_labels,labelers={k:sum(r['labeler']==k for r in rows) for k in sorted({r['labeler'] for r in rows})},
        operating_point='zero training-positive loss in leave-one-title-out scores',
        oof_recall=float(np.mean(oof[y==1]>=threshold)),oof_negative_rejection=float(np.mean(oof[y==0]<threshold)),
        selection_warning='Training-only threshold; independent interval and mask evaluation required')
    if output.exists():raise FileExistsError('Choose a new detector head name')
    write_json(output,result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--labels',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('train');p.add_argument('--labels',type=Path,required=True);p.add_argument('--model',required=True)
    p.add_argument('--allow-ai-labels',action='store_true')
    args=parser.parse_args()
    root=prepare_environment()
    if args.command=='prepare':
        print(f'{prepare(args.labels,args.output,root)} training regions reproduced')
        raise SystemExit(0)
    if not args.model or len(args.model)>64 or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in args.model):
        parser.error('Use a lowercase model name containing letters, digits and hyphens')
    result=calibrate(args.labels,root/'roi-models'/f'{args.model}.json',root,args.allow_ai_labels)
    print({k:result[k] for k in ('crops','oof_recall','oof_negative_rejection','experimental')})
