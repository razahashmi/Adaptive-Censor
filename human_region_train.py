"""Fit an anatomy veto with genuine crop labels and sealed training-title calibration."""
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np

from labeling import read_json,write_json
from model_setup import prepare_environment
from provenance import file_hash


def balanced_calibration(rows):
    """Reserve whole training titles using label support, never model scores."""
    import hashlib
    import itertools
    import math
    from select_windows import normalize_title
    if any(r.get('split')!='train' for r in rows):
        raise ValueError('Balanced calibration accepts training titles only')
    titles=sorted({normalize_title(r['title']) for r in rows},
                  key=lambda t:hashlib.sha256(('human-region-balanced-v1'+t).encode()).hexdigest())
    if len(titles)<10:raise ValueError('Need ten training titles for balanced calibration')
    by_value={v:{normalize_title(r['title']) for r in rows if r['label']==v} for v in (0,1)}
    by_class=defaultdict(set)
    for r in rows:
        if r['label']==1:by_class[r['detector_class']].add(normalize_title(r['title']))
    eligible={c:ts for c,ts in by_class.items() if len(ts)>=6}
    if not eligible:raise ValueError('No anatomy class has positive labels across six training titles')
    size=max(3,min(len(titles)-3,round(len(titles)*.3)))
    if math.comb(len(titles),size)<=50000:
        candidates=itertools.combinations(titles,size)
    else:
        rng=np.random.default_rng(1441)
        candidates=(tuple(rng.choice(titles,size,replace=False)) for _ in range(20000))
    best=None
    for candidate in candidates:
        held=set(candidate)
        if any(len(ts&held)<3 or len(ts-held)<3 for ts in by_value.values()):continue
        supported=sum(len(ts&held)>=3 and len(ts-held)>=3 for ts in eligible.values())
        if not supported:continue
        count=sum(normalize_title(r['title']) in held for r in rows)
        positive=sum(r['label']==1 and normalize_title(r['title']) in held for r in rows)
        score=(supported,-abs(count/len(rows)-.3),-abs(positive/count-sum(r['label'] for r in rows)/len(rows)))
        if best is None or score>best[0]:best=(score,held)
    if best is None:raise ValueError('No title partition meets unchanged three-title class-support requirements')
    return best[1]


def train(labels,output,class_balanced=False):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from learned import title_class_weights,thresholds
    from human_calibration import partitions,calibrated_head
    from roi_gate import validate_head
    from select_windows import normalize_title
    data=read_json(labels)
    if data.get('review_api')!=1 or data.get('dataset_sha256')!=file_hash(labels.parent/'dataset.json'):
        raise ValueError('Use server-saved human crop labels for this immutable dataset')
    if file_hash(labels.parent/'features.npz')!=data.get('feature_identity'):
        raise ValueError('Crop feature file changed')
    if any(not r.get('reviewed') or r.get('labeler')!='human' for r in data['rows']):
        raise ValueError('Complete every crop review first; Unsure is allowed')
    selected=[(i,r) for i,r in enumerate(data['rows']) if type(r.get('label')) is int and r['label'] in (0,1)]
    if len(selected)<40 or any(r.get('split')!='train' for _,r in selected):
        raise ValueError('Need at least 40 definite human labels, from training titles only')
    groups=np.array([normalize_title(r['title']) for _,r in selected]);y=np.array([r['label'] for _,r in selected])
    split=partitions([(None,dict(title=t),None) for t in groups],seed='human-region-calibration-v1')
    # C is fixed, so there is no separate hyperparameter tuning step. Reserve
    # both small held-out groups for calibration to give rare anatomy classes
    # more title coverage; neither group enters coefficient fitting.
    calibration_titles=(balanced_calibration([r for _,r in selected]) if class_balanced
                        else split['calibration']|split['tune'])
    held=np.isin(groups,list(calibration_titles));fit=~held
    if any(len(set(groups[mask&(y==v)]))<3 for mask in (fit,held) for v in (0,1)):
        raise ValueError('Both fitting and calibration need each class across three titles; collect more crop labels')
    with np.load(labels.parent/'features.npz') as cached:
        x=cached['embeddings'][[i for i,_ in selected]].astype(np.float32)
    model=make_pipeline(StandardScaler(),LogisticRegression(C=.01,max_iter=3000))
    model.fit(x[fit],y[fit],logisticregression__sample_weight=title_class_weights(y[fit],groups[fit]))
    scaler,linear=model.steps[0][1],model.steps[1][1]
    raw=dict(mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),coef=linear.coef_[0].tolist(),intercept=float(linear.intercept_[0]))
    events=np.array([r['project']+'/'+r['window'] for _,r in selected])
    calibrated=calibrated_head(raw,model.decision_function(x[held]),y[held],groups[held],events[held])
    if 'coef' not in calibrated or calibrated.get('calibration_warning') or calibrated.get('candidate_threshold') is None:
        raise ValueError('The held-out crop ranking cannot support a useful veto; no model saved')
    # Preserve recall: use the high-recall candidate floor for the regional veto,
    # and never refit the classifier after establishing its operating point.
    score=(x[held]-np.asarray(calibrated['mean']))/np.asarray(calibrated['scale'])
    score=score@np.asarray(calibrated['coef'])+calibrated['intercept']
    # Affine folding changes floating-point addition order. Establish the
    # inclusive boundary on the final runtime scores, including tied groups.
    candidate,_=thresholds(score,y[held])
    calibrated.update(candidate_threshold=candidate,decision_threshold=candidate,
                      calibration_threshold_source='folded runtime head scores')
    positives=defaultdict(set);calibration_positive=defaultdict(set)
    for index,(_,r) in enumerate(selected):
        if r['label']==1:
            (positives if fit[index] else calibration_positive)[r['detector_class']].add(groups[index])
    supported=sorted(c for c,titles in positives.items() if len(titles)>=3 and len(calibration_positive[c])>=3)
    if not supported:raise ValueError('No anatomy class has positive support across three fitting and three calibration titles')
    result=dict(calibrated,clip_identity=data['clip_identity'],anatomy_identity=data['anatomy_identity'],
                anatomy_model=data['anatomy_model'],feature_views=data['feature_views'],supported_classes=supported,
                class_positive_titles={c:sorted(t) for c,t in positives.items()},negative_confirmations=3,
                crops=len(selected),fit_titles=sorted(set(groups[fit])),calibration_titles=sorted(set(groups[held])),
                titles=sorted(set(groups)),dataset_identity=file_hash(labels),experimental=True,
                calibration_refit=False,ai_labels_allowed=False,labelers={'human':len(selected)},
                partition_method='class-balanced-training-titles-v1' if class_balanced else 'fixed-title-hash-v1',
                partition_uses_model_scores=False,minimum_positive_titles_per_supported_class_per_partition=3,
                operating_point='sealed-training-calibration-protect-recall',
                calibration_recall=float(np.mean(score[y[held]==1]>=calibrated['decision_threshold'])),
                calibration_negative_rejection=float(np.mean(score[y[held]==0]<calibrated['decision_threshold'])),
                qualification_note='Training-title calibration only; actual-player and mask evaluation required')
    validate_head(result)
    if output.exists():raise FileExistsError('Never overwrite a region classifier')
    write_json(output,result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--labels',type=Path,required=True);p.add_argument('--model',required=True)
    p.add_argument('--class-balanced',action='store_true',help='Balance class support across whole training titles; never use validation/test labels')
    a=p.parse_args();result=train(a.labels,prepare_environment()/'roi-models'/f'{a.model}.json',a.class_balanced)
    print('Saved experimental human region classifier; not enabled in playback:',a.model)
