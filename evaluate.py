"""Measure filtering against a separately annotated gold timeline; never equate throughput with accuracy."""
import argparse
import json
import math
from pathlib import Path
from core import CATEGORIES, compile_plan, default_policy, validate_events


def union(intervals):
    merged=[]
    for start,end in sorted(intervals):
        if start>=end: continue
        if merged and start<=merged[-1][1]: merged[-1][1]=max(end,merged[-1][1])
        else: merged.append([start,end])
    return merged


def length(intervals):
    return sum(b-a for a,b in intervals)


def overlap(a,b):
    i=j=0; total=0
    while i<len(a) and j<len(b):
        total+=max(0,min(a[i][1],b[j][1])-max(a[i][0],b[j][0]))
        if a[i][1]<b[j][1]: i+=1
        else: j+=1
    return total


def intersect(a, b):
    return union((max(x,u),min(y,v)) for x,y in a for u,v in b if x<v and u<y)


def review_windows(reviewed, category, duration):
    if reviewed is None:
        return [[0,duration]]
    if not isinstance(reviewed,dict) or any(k not in CATEGORIES for k in reviewed):
        raise ValueError('Reviewed windows must map known categories to time intervals')
    windows=reviewed.get(category,[])
    if not isinstance(windows,list):
        raise ValueError('Reviewed windows must be a list')
    clean=[]
    for interval in windows:
        if not isinstance(interval,(list,tuple)) or len(interval)!=2:
            raise ValueError('Each reviewed window needs a start and end')
        a,b=map(float,interval)
        if not math.isfinite(a) or not math.isfinite(b) or not 0<=a<b<=duration:
            raise ValueError('Reviewed windows must be finite intervals inside the media')
        clean.append([a,b])
    return union(clean)


def evaluate(predicted, gold, duration, policy, reviewed=None):
    predicted=validate_events(predicted,duration);gold=validate_events(gold,duration)
    report={}
    for category in CATEGORIES:
        targets=[(e['start'],e['end']) for e in gold if e['category']==category and e['enabled']]
        windows=review_windows(reviewed,category,duration)
        plan=compile_plan([e for e in predicted if e['category']==category],policy,duration)
        redactions=intersect(union((e['start'],e['end']) for e in plan),windows)
        truth=intersect(union(targets),windows); true_seconds=length(truth); matched=overlap(redactions,truth)
        complete_targets=[(a,b) for a,b in targets if overlap(windows,[[a,b]])>=(b-a)-1e-8]
        filtered=length(redactions)
        precision=matched/filtered if filtered else None
        recall=matched/true_seconds if true_seconds else None
        report[category]={
            'reviewed_seconds':length(windows),'unreviewed_seconds':duration-length(windows),
            'gold_events':len(complete_targets), 'gold_seconds':true_seconds,
            'partially_reviewed_gold_events':sum(0<overlap(windows,[[a,b]])<(b-a)-1e-8 for a,b in targets),
            'filter_seconds':filtered,'matched_seconds':matched,
            'missed_seconds':max(0,true_seconds-matched),
            'unnecessary_filter_seconds':max(0,filtered-matched),
            'time_precision':precision, 'time_recall':recall,
            'time_f1':2*matched/(filtered+true_seconds) if true_seconds else None,
            'events_at_least_95_percent_covered':sum(overlap(redactions,[[a,b]])>=(b-a)*.95 for a,b in complete_targets),
        }
    return {'duration':duration,'by_category':report,
            'interpretation':'Temporal coverage inside explicitly reviewed windows only; omitted reviewed categories are unknown. Without reviewed windows the entire media is assumed annotated for every category. Padding outside gold intervals counts as unnecessary filtering. These metrics do not measure region-mask accuracy or establish generalization to other media.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timeline',type=Path,required=True)
    parser.add_argument('--gold',type=Path,required=True)
    parser.add_argument('--profile',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    timeline=json.loads(args.timeline.read_text(encoding='utf-8'))
    gold=json.loads(args.gold.read_text(encoding='utf-8'))
    policy=json.loads(args.profile.read_text(encoding='utf-8'))['policy'] if args.profile else default_policy()
    result=evaluate(timeline['events'],gold['events'],gold['duration'],policy,gold.get('reviewed'))
    result['gold_description']=gold.get('description','User annotations')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
