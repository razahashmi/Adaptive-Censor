"""Sealed training-title calibration; never fit on validation or test labels."""
import hashlib
import numpy as np

from labeling import LABEL_CATEGORIES, is_complete, read_json
from select_windows import normalize_title


def partitions(rows,seed='human-calibration-v1'):
    titles=sorted({normalize_title(w['title']) for _,w,_ in rows},
                  key=lambda t:hashlib.sha256((seed+t).encode()).hexdigest())
    if len(titles)<10:raise ValueError('Need at least ten training titles for separate tuning and calibration')
    count=max(2,round(len(titles)*.2));tuning=max(2,round(len(titles)*.1))
    calibration=set(titles[:count]); tune=set(titles[count:count+tuning])
    fit=set(titles[count+tuning:])
    return dict(fit=fit,tune=tune,calibration=calibration)


def balanced_partitions(rows, minimum_titles=3, minimum_events=5, trials=50000):
    """Reserve whole training titles with positive/negative support per category.

    The old hash-only split can put nearly every rare positive in fitting even
    when enough titles exist. This choice uses annotations and a fixed seed,
    never features, classifier scores, validation labels or test labels. Group
    cross-validation still selects regularization inside the fitting titles.
    """
    from itertools import combinations
    from math import comb
    from learned import grid_times, grid_labels
    titles=sorted({normalize_title(w['title']) for _,w,_ in rows},
                  key=lambda t:hashlib.sha256(('human-scene-balanced-v1'+t).encode()).hexdigest())
    if len(titles)<10:
        raise ValueError('Need at least ten training titles for separate fitting and calibration')
    index={t:i for i,t in enumerate(titles)}
    # Per title: positive-title presence, negative-title presence and distinct
    # target events. A long or densely sampled scene cannot become many events.
    support=np.zeros((len(titles),len(LABEL_CATEGORIES),3),dtype=np.int64)
    for _,w,label in rows:
        times=grid_times(w);i=index[normalize_title(w['title'])]
        for j,c in enumerate(LABEL_CATEGORIES):
            y=grid_labels(w,label,c,times)
            support[i,j,0]|=int(np.any(y==1))
            support[i,j,1]|=int(np.any(y==0))
            support[i,j,2]+=len(set(event_ids(w,label,c,times)[y==1])-{''})
    totals=support.sum(axis=0)
    supported=(totals[:,0]>=2*minimum_titles)&(totals[:,1]>=2*minimum_titles)
    needed=np.zeros_like(totals)
    needed[supported,:2]=minimum_titles
    needed[supported&(totals[:,2]>=2*minimum_events),2]=minimum_events
    count=max(minimum_titles,round(len(titles)*.2))
    count=min(count,len(titles)-minimum_titles)
    rng=np.random.default_rng(251015)
    choices=(combinations(range(len(titles)),count) if comb(len(titles),count)<=trials else
             (tuple(sorted(rng.choice(len(titles),count,replace=False))) for _ in range(trials)))
    best=None;selected=None
    for choice in choices:
        held=support[list(choice)].sum(axis=0);fit=totals-held
        if np.any(held<needed) or np.any(fit<needed):
            continue
        distance=float(np.square(held/np.maximum(totals,1)-count/len(titles)).sum())
        key=(distance,choice)
        if best is None or key<best:
            best=key;selected=set(choice)
    if selected is None:
        raise ValueError('No supported whole-title calibration split found; collect more training titles')
    return dict(fit={t for i,t in enumerate(titles) if i not in selected},tune=set(),
                calibration={t for i,t in enumerate(titles) if i in selected})


def collect(folders):
    rows=[];seen=set();definitions=None
    for folder in folders:
        for w in read_json(folder/'project.json')['windows']:
            label=read_json(folder/'labels'/f"{w['id']}.json",{})
            if w['split']!='train' or not is_complete(label):continue
            if label.get('labeler')!='human' or label.get('drafts_shown'):
                raise ValueError('Human training requires completed labels without exposed drafts')
            key=(normalize_title(w['title']),w['episode'],round(w['start'],3),round(w['length'],3))
            if key in seen:raise ValueError('Duplicate human training target')
            seen.add(key)
            if definitions is None:definitions=label['definitions_revision']
            if definitions!=label['definitions_revision']:raise ValueError('Mixed label definitions')
            rows.append((folder,w,label))
    if not rows:raise ValueError('No completed human training windows')
    return rows


def event_ids(w,label,category,times):
    ids=np.full(len(times),'',dtype=object)
    for i,(a,b) in enumerate(label['categories'][category]['intervals']):
        ids[(times>=a)&(times<b)]=f"{w['id']}:{category}:{i}"
    return ids


def operating_points(scores,y,groups,events,minimum_events=5,minimum_titles=3):
    from learned import thresholds
    scores=np.asarray(scores);y=np.asarray(y);groups=np.asarray(groups);events=np.asarray(events)
    if len(np.unique(y))<2:
        return dict(candidate_threshold=None,decision_threshold=None,
                    skipped='Calibration subset needs both classes')
    candidate,decision=thresholds(scores,y)
    if decision is not None:
        keep=(scores>=decision)&(y==1)
        if len(set(events[keep])-{''})<minimum_events or len(set(groups[keep]))<minimum_titles:
            decision=None
    return dict(candidate_threshold=candidate,decision_threshold=decision,
                calibration_positive_events=len(set(events[y==1])-{''}),
                calibration_positive_titles=len(set(groups[y==1])),
                decision_minimum_events=minimum_events,decision_minimum_titles=minimum_titles)


def calibrated_head(raw_head,scores,y,groups,events):
    """Keep the fitted classifier fixed; calibrate only on reserved train titles."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    if len(np.unique(y))<2:
        return dict(skipped='Calibration subset needs both classes')
    calibration=LogisticRegression(C=1.,max_iter=2000).fit(np.asarray(scores)[:,None],y)
    slope=float(calibration.coef_[0,0]);intercept=float(calibration.intercept_[0])
    if slope<=0:
        return dict(raw_head,candidate_threshold=float(np.min(scores)),decision_threshold=None,
                    calibration_warning='Held-out scores have reversed ranking; proposals only')
    operating=operating_points(np.asarray(scores)*slope+intercept,y,groups,events)
    # Folding affine logit calibration into the head has negligible runtime cost.
    return dict(raw_head,coef=(np.asarray(raw_head['coef'])*slope).tolist(),
                intercept=float(raw_head['intercept']*slope+intercept),**operating,
                calibration_average_precision=float(average_precision_score(y,scores)),
                score_calibration=dict(method='held-out-training-title Platt scaling',slope=slope,intercept=intercept,
                                       probability='sigmoid of calibrated head score'))


def train_linear(folders,kinds=('clip','vjepa','ast'),offsets=(-.5,.5),balanced=False):
    from learned import train,window_matrix,grid_labels
    from learned_runtime import head_scores
    from provenance import FEATURE_SCHEMA
    rows=collect(folders);split=balanced_partitions(rows) if balanced else partitions(rows)
    fit_titles=split['fit']|split['tune']
    # A temporary project view is unnecessary: train accepts a title restriction.
    model=train(folders,kinds,temporal_offsets=offsets,balance_titles=True,input_layout='player-chunks-v1',
                include_titles=fit_titles)
    heads={}
    for category in LABEL_CATEGORIES:
        raw=model['heads'][category]
        if 'coef' not in raw:heads[category]=raw;continue
        scores=[];targets=[];groups=[];events=[]
        for folder,w,label in rows:
            title=normalize_title(w['title'])
            if title not in split['calibration']:continue
            times,x=window_matrix(folder,w,kinds,temporal_offsets=offsets,input_layout='player-chunks-v1')
            y=grid_labels(w,label,category,times);keep=y>=0
            scores.extend(head_scores(raw,x)[keep]);targets.extend(y[keep])
            groups.extend([title]*int(keep.sum()));events.extend(event_ids(w,label,category,times)[keep])
        heads[category]=calibrated_head(raw,np.asarray(scores),np.asarray(targets),np.asarray(groups),np.asarray(events))
    return dict(model,heads=heads,experimental=True,training_pool_windows=len(rows),
                calibration_titles=sorted(split['calibration']),fit_titles=sorted(fit_titles),
                partition_method='class-balanced-training-titles-v1' if balanced else 'hash-training-titles-v1',
                validation_labels_used=False,calibration_refit=False,
                qualification_note='Training-only calibration; independent full-player evaluation still required')
