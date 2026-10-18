"""Train-only scene calibration selection and fixed development ranking replay.

No verifier inference or model promotion. The previous classifier and new
classifier see the same exact-player input matrices and human review intervals.
"""
import argparse
from pathlib import Path
import time

import numpy as np

from labeling import LABEL_CATEGORIES, read_json, write_json
from provenance import digest, file_hash, head_identity

NAME='human-balanced-20261015-2'
BASE='human-aligned-20261014-1'
PROJECTS=['training-actions-human-1','train-drive-expansion-1',
          'training-content-human-2','training-batch-more-20261012-2']


def train(root, output):
    from human_calibration import train_linear
    from review_training_experiment import verify
    from threadpoolctl import threadpool_limits
    snapshot=verify()
    target=root/'heads'/(NAME+'.json')
    if target.exists() or (output/'choice.json').exists():
        raise FileExistsError('Never overwrite a trained experiment')
    contract=dict(frozen_training_inputs_identity=digest(snapshot), projects=PROJECTS,
        training_code={f:file_hash(Path(__file__).with_name(f)) for f in
            ('human_calibration.py','learned.py','balanced_scene_experiment.py')},
        baseline_head_identity=head_identity(read_json(root/'heads'/(BASE+'.json'))),
        selection='Fixed linear model, label-only balanced whole-training-title calibration; no validation/test selection',
        scope='Repeatedly inspected development clips; classifier ranking only, not full-player accuracy or qualification')
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/'experiment.json',contract)
    started=time.perf_counter()
    with threadpool_limits(limits=4):
        model=train_linear([root/'labeling'/n for n in PROJECTS],balanced=True)
    verify()
    model.update(training_dataset_identity=digest(snapshot),training_code=contract['training_code'],
                 projects=PROJECTS)
    write_json(target,model)
    choice=dict(name=NAME, head_identity=head_identity(model), experiment_identity=digest(contract),
                selected_before_regression=True,training_seconds=time.perf_counter()-started,
                calibration={c:{k:v for k,v in h.items() if k in
                    ('C','candidate_threshold','decision_threshold','calibration_positive_events',
                     'calibration_positive_titles','calibration_average_precision','calibration_warning','skipped')}
                    for c,h in model['heads'].items()})
    write_json(output/'choice.json',choice)
    print('Training complete:',choice['calibration'],flush=True)


def report(root, output):
    from sklearn.metrics import average_precision_score
    from learned import window_matrix,grid_labels
    from human_calibration import event_ids
    from learned_runtime import head_scores,head_outputs
    from evaluate_labels import evaluate
    from review_training_experiment import verify
    verify()
    contract=read_json(output/'experiment.json');choice=read_json(output/'choice.json')
    heads={n:read_json(root/'heads'/(n+'.json')) for n in (BASE,NAME)}
    if (choice['experiment_identity']!=digest(contract)
            or head_identity(heads[NAME])!=choice['head_identity']
            or head_identity(heads[BASE])!=contract['baseline_head_identity']):
        raise ValueError('Preselected models changed')
    for name,sha in contract['training_code'].items():
        if file_hash(Path(__file__).with_name(name))!=sha:
            raise ValueError('Experiment code changed')
    frozen=read_json(root/'benchmarks/human-validation-20261009-1/dataset.json')
    project,labels=frozen['contract']['project'],frozen['contract']['labels']
    folder=root/'labeling/val-2';results={};data={}
    for w in project['windows']:
        t,x=window_matrix(folder,w,heads[BASE]['kinds'],temporal_offsets=heads[BASE]['temporal_offsets'],
                          input_layout=heads[BASE]['input_layout'])
        data[w['id']]=(t,x)
    for name,model in heads.items():
        records={};ranking={};timings=[]
        for w in project['windows']:
            times,x=data[w['id']];trace=[];events=[]
            started=time.perf_counter()
            for start in np.arange(0,w['clip_duration'],6.):
                keep=(times>=start)&(times<min(start+6,w['clip_duration']))
                tr,ev=head_outputs(model,times[keep],x[keep]);trace+=tr;events+=ev
            timings.append(time.perf_counter()-started)
            records[w['id']]=dict(window=w['id'],events=events,trace=trace)
        for c in LABEL_CATEGORIES:
            scores=[];gold=[];event_count=set();positive_titles=set()
            head=model['heads'][c]
            if 'coef' not in head:
                ranking[c]=dict(skipped=head.get('skipped','No fitted classifier'));continue
            for w in project['windows']:
                times,x=data[w['id']];y=grid_labels(w,labels[w['id']],c,times);keep=y>=0
                scores.extend(head_scores(head,x)[keep]);gold.extend(y[keep])
                event_count.update(set(event_ids(w,labels[w['id']],c,times)[y==1])-{''})
                if np.any(y==1):positive_titles.add(w['title'])
            ranking[c]=dict(average_precision=float(average_precision_score(gold,scores)) if any(gold) else None,
                positive_events=len(event_count),positive_titles=len(positive_titles),positive_grid_points=int(sum(gold)),
                calibration_decision_threshold=head.get('decision_threshold'))
        results[name]=dict(ranking=ranking,
            standalone_classifier=evaluate(project,labels,records,split='val',padding=False),
            head_scoring_seconds=sum(timings), clips=len(records),
            speed_scope='Cached feature matrices; excludes decoding, feature extraction, verification and playback')
    verify()
    summary=dict(choice=choice,results=results,inputs={w['id']:file_hash(folder/'clips'/(w['id']+'.mp4')) for w in project['windows']},
        qualification_passed=False,model_promoted=False,player_defaults_changed=False,
        evaluation_scope=contract['scope'],
        interpretation='Average precision measures ranking across thresholds; it is not the precision of playback. '
                       'Standalone classifier decisions exclude scene verification, region checks, policy padding '
                       'and release gates, so these metrics do not describe deployed censoring.')
    write_json(output/'report.json',summary)
    print({n:r['ranking'] for n,r in results.items()},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['train','report'])
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    from model_setup import prepare_environment
    {'train':train,'report':report}[a.stage](prepare_environment(),a.output)
