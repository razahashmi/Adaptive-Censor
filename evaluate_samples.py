"""Score frozen sparse labels without turning clean-frame accuracy into a 95% claim."""
import argparse
import json
import math
from pathlib import Path
from core import CATEGORIES, compile_plan, default_policy


def rates(counts, target=.95):
    tp, fp, fn, tn = (counts[k] for k in ('tp','fp','fn','tn'))
    precision = tp/(tp+fp) if tp+fp else None
    recall = tp/(tp+fn) if tp+fn else None
    values = [precision, recall]
    status = ('below_target' if any(v is not None and v < target for v in values)
              else 'insufficient_evidence' if None in values else 'meets_sample_target')
    return dict(counts, precision=precision, recall=recall,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                overall_accuracy=(tp+tn)/(tp+fp+fn+tn) if tp+fp+fn+tn else None,
                sample_target_status=status)


def evaluate_samples(cases, gold, prediction, target=.95):
    if not math.isfinite(target) or not 0<target<=1:
        raise ValueError('Target must be in (0, 1]')
    categories=list(gold['definitions'])
    if not categories or any(c not in CATEGORIES for c in categories):
        raise ValueError('Gold definitions must use known categories')
    ids=[c['id'] for c in cases]; rows=prediction['results']
    result_ids=[r['id'] for r in rows]
    if not ids or len(set(ids))!=len(ids) or len(set(result_ids))!=len(result_ids):
        raise ValueError('Cases and predictions need unique case ids')
    if set(ids)!=set(result_ids) or set(ids)!=set(gold['cases']):
        raise ValueError('Cases, labels and predictions must cover exactly the same windows')
    by_id={r['id']:r for r in rows}; occupied={}
    for case in cases:
        a,b=case['start'],case['end']; times=case['times']
        if not all(math.isfinite(t) for t in [a,b,*times]) or not 0<=a<b or not times:
            raise ValueError('Cases need finite bounds and nonempty timestamps')
        if any(not a<=t<b for t in times) or any(x>=y for x,y in zip(times,times[1:])):
            raise ValueError('Sample times must be distinct, increasing and inside their window')
        previous=occupied.setdefault(case['title'],[])
        if any(a<y and x<b for x,y in previous):
            raise ValueError('Overlapping windows would double-count evidence')
        previous.append((a,b))
        labels=gold['cases'][case['id']]
        if set(labels)!=set(categories):
            raise ValueError('Every reviewed case needs an explicit label for every category')
        for label in labels.values():
            indices=label['positive']+label['unknown']
            if (any(type(i) is not int or not 0<=i<len(times) for i in indices)
                    or len(indices)!=len(set(indices))):
                raise ValueError('Positive/unknown indices must be distinct and within the case')
        if any(e.get('confirmed') or e.get('source')=='manual' for e in by_id[case['id']]['events']):
            raise ValueError('Manual decisions cannot count as automatic model predictions')
    def empty():
        return {c:dict(tp=0,fp=0,fn=0,tn=0,unknown=0) for c in categories}
    modes={}
    for padding in [False,True]:
        total=empty(); by_title={title:empty() for title in occupied}
        for case in cases:
            events=by_id[case['id']]['events']
            for category in categories:
                policy=default_policy()
                # Measure ability to filter every category, even one allowed by the default profile.
                policy['categories'][category]['visual']='hide'
                if not padding:policy.update(pre_padding=0,post_padding=0)
                plan=compile_plan([e for e in events if e['category']==category],policy,case['end']+3)
                label=gold['cases'][case['id']][category]
                for index,t in enumerate(case['times']):
                    positive=index in label['positive']
                    filtered=any(e['start']<=t<e['end'] for e in plan)
                    key=('unknown' if index in label['unknown'] else
                         'tp' if positive and filtered else 'fn' if positive else 'fp' if filtered else 'tn')
                    total[category][key]+=1;by_title[case['title']][category][key]+=1
        modes['with_padding' if padding else 'without_padding']={
            'by_category':{c:rates(v,target) for c,v in total.items()},
            'by_title':{t:{c:rates(v,target) for c,v in cats.items()} for t,cats in by_title.items()}}
    return dict(target_precision=target,target_recall=target,
                windows=len(cases),sampled_timestamps=sum(len(c['times']) for c in cases),
                target_seconds=sum(c['end']-c['start'] for c in cases),
                analysis_revision=prediction.get('analysis_revision'),
                source_hashes=prediction.get('source_hashes'),
                processing_seconds=sum(r.get('seconds',0) for r in rows),
                modes=modes,representative_accuracy_claim_supported=False,
                interpretation='Sparse timestamp challenge test, not independent events, continuous-time coverage or pixel-mask accuracy. Unknowns are excluded and counted. Unconfirmed predictions miss known positives. Each category is enabled for scoring; default policy thresholds and optional timing padding still apply. A sample target pass is not a real-world 95% guarantee. Independent human-reviewed, title-separated representative evaluation is still required.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('cases','gold','predictions','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--target',type=float,default=.95)
    args=parser.parse_args()
    result=evaluate_samples(*(json.loads(p.read_text(encoding='utf-8')) for p in
                            (args.cases,args.gold,args.predictions)),args.target)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result['modes']['without_padding']['by_category'],indent=2))
