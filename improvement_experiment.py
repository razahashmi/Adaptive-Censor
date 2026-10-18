"""Frozen human training, calibration, real-player comparison and queue reporting."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import time
import statistics
from collections import Counter

from labeling import read_json,write_json
from model_setup import prepare_environment
from provenance import digest,file_hash,head_identity

ROOT=prepare_environment();REPO=Path(__file__).parent
OUT=ROOT/'benchmarks/human-improvements-20261011-1'
PROJECTS=['training-actions-human-1','train-drive-expansion-1','training-content-human-2']
NAMES=dict(linear='human-aligned-20261011-1',attention='human-attention-20261011-1')
REGIONS='anatomy-human-20261011-2'


def verify():
    training=read_json(ROOT/'benchmarks/human-supervised-20261009-1/dataset.json')
    validation=read_json(ROOT/'benchmarks/human-validation-20261009-1/dataset.json')
    for row in training['contract']['rows']:
        folder=ROOT/'labeling'/row['project'];w=row['window']
        if file_hash(folder/'labels'/f"{w['id']}.json")!=row['label_sha256'] or file_hash(folder/'clips'/f"{w['id']}.mp4")!=row['clip_sha256']:
            raise ValueError('Frozen human training data changed')
    for wid,label in validation['contract']['labels'].items():
        folder=ROOT/'labeling/val-2'
        if read_json(folder/'labels'/f'{wid}.json')!=label or file_hash(folder/'clips'/f'{wid}.mp4')!=validation['contract']['clips'][wid]:
            raise ValueError('Human validation data changed')
    return training,validation


def freeze():
    training,validation=verify()
    value=dict(date='2026-10-11',training_identity=training['identity'],validation_identity=validation['identity'],
               projects=PROJECTS,names=NAMES,selection='Highest mean sealed training-calibration AP; no validation fitting or threshold adjustment',
               planned_comparisons=['baseline','linear','attention','confirmed'],confirmation_mode='confirm',workers=2)
    path=OUT/'experiment.json'
    if path.exists() and read_json(path)!=value:raise ValueError('Experiment contract changed')
    write_json(path,value)
    print('Frozen 113 training and 42 human validation clips; validation is development evidence, not a fresh final test',flush=True)


def train():
    from human_calibration import train_linear
    from attention_training import train_probe
    from threadpoolctl import threadpool_limits
    training,_=verify();folders=[ROOT/'labeling'/n for n in PROJECTS]
    with threadpool_limits(limits=4):
        if not (ROOT/'heads'/f"{NAMES['linear']}.json").exists():
            linear=train_linear(folders)
            write_json(ROOT/'heads'/f"{NAMES['linear']}.json",linear)
            print('Saved exact-player linear candidate',head_identity(linear),flush=True)
        if not (ROOT/'heads'/f"{NAMES['attention']}.json").exists():train_probe(ROOT,folders,NAMES['attention'],epochs=8)
    models={k:read_json(ROOT/'heads'/f'{n}.json') for k,n in NAMES.items()}
    # Threshold calibration also uses training labels: promotion must exclude
    # all 40 titles, not only the subset used to fit coefficients.
    from human_calibration import collect
    from select_windows import normalize_title
    titles=sorted({normalize_title(w['title']) for _,w,_ in collect(folders)})
    for kind,model in models.items():
        updated=dict(model,titles=titles,projects=PROJECTS,training_dataset_identity=training['identity'])
        if head_identity(updated)!=head_identity(model):raise ValueError('Training-scope annotation changed model function')
        write_json(ROOT/'heads'/f'{NAMES[kind]}.json',updated);models[kind]=updated
    scores={k:sum(h.get('calibration_average_precision',0) for h in m['heads'].values())/5 for k,m in models.items()}
    selected=max(scores,key=scores.get)
    value=dict(selected=selected,selected_head=NAMES[selected],calibration_scores=scores,
               heads={k:head_identity(m) for k,m in models.items()},selected_before_validation_predictions=True,
               training_code={p:file_hash(REPO/p) for p in ('learned.py','human_calibration.py','attention_probe.py','attention_training.py','player_inputs.py')})
    previous=read_json(OUT/'choice.json',{})
    if previous and previous!=value:raise ValueError('Frozen classifier selection changed; choose a new experiment')
    if not previous:write_json(OUT/'choice.json',value)
    verify()
    print('Preselected confirmation candidate:',selected,scores,flush=True)


def benchmark(configuration=None):
    if configuration is None:
        # A separate process releases CUDA arenas, not just Python references.
        # Several resident backbone variants otherwise exceed this GPU's VRAM.
        import subprocess,sys
        for name in ('baseline','linear','attention','confirmed'):
            subprocess.run([sys.executable,str(Path(__file__)), 'benchmark','--configuration',name],check=True)
        return
    from analysis_engine import Models,ScanConfig
    from predict_project import run_contract,predict_window
    from evaluate_labels import evaluate,table
    _,validation=verify();project=validation['contract']['project'];labels=validation['contract']['labels']
    folder=ROOT/'labeling/val-2';choice=read_json(OUT/'choice.json')
    if any(head_identity(read_json(ROOT/'heads'/f'{name}.json'))!=choice['heads'][kind] for kind,name in NAMES.items()):
        raise ValueError('Preselected classifier changed before validation')
    base=dict(validation['contract']['baseline_config'])
    configs={n:ScanConfig(**dict(base,learned_heads=head,learned_verification=mode))
             for n,head,mode in [('baseline','','off'),('linear',NAMES['linear'],'off'),
                                  ('attention',NAMES['attention'],'off'),('confirmed',choice['selected_head'],'confirm')]}
    configs={configuration:configs[configuration]}
    contracts={n:run_contract(ROOT,folder,project,project['windows'],c,2,True) for n,c in configs.items()}
    for n,c in contracts.items():
        path=OUT/'predictions'/n/'run.json';previous=read_json(path,{})
        if previous and previous['identity']!=c['identity']:raise ValueError('Player run changed; choose a new experiment')
        write_json(path,dict(c,complete=False))
    models=Models(workers=2,evaluate_unpromoted=True);started=time.perf_counter()
    try:
        def pair(w):
            timings={}
            for n,c in configs.items():
                path=OUT/'predictions'/n/f"{w['id']}.json"
                if path.exists():
                    result=read_json(path)
                    if result['run_identity']!=contracts[n]['identity']:raise ValueError('Stale improvement prediction')
                else:
                    result=predict_window(models,folder/'clips'/f"{w['id']}.mp4",w,c)
                    if run_contract(ROOT,folder,project,project['windows'],c,2,True)['identity']!=contracts[n]['identity']:
                        raise ValueError('Player code/assets changed during benchmark')
                    write_json(path,dict(result,run_identity=contracts[n]['identity'],execution_mode='isolated-configuration'))
                timings[n]=result['seconds']
            return timings
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs={pool.submit(pair,w):w for w in project['windows']}
            for i,future in enumerate(as_completed(jobs),1):
                print(f"{i}/42 {jobs[future]['id']}: {future.result()}",flush=True)
        verify()
        for n in configs:
            rows={w['id']:read_json(OUT/'predictions'/n/f"{w['id']}.json") for w in project['windows']}
            reports={p:evaluate(project,labels,rows,split='val',padding=p=='balanced') for p in ('balanced','none')}
            write_json(OUT/f'{n}.json',reports)
            write_json(OUT/'predictions'/n/'run.json',dict(contracts[n],complete=True,windows=len(rows),
                       comparison_wall_seconds=time.perf_counter()-started,
                       timing_scope='Initial partial paired passes had multiple backbones resident; remaining clips resumed with one configuration per process. Mixed warm/cold and contention timings; not an isolated speed comparison or sustained-live benchmark.'))
            print(n+'\n'+table(reports['balanced']),flush=True)
    finally:models.close()


def report():
    from learned import promotion_check,category_choice
    from evaluate_labels import evaluate,filtered,candidates,difference
    from evaluate import intersect,union,length
    from labeling import LABEL_CATEGORIES
    _,validation=verify();project=validation['contract']['project'];labels=validation['contract']['labels']
    base=read_json(OUT/'baseline.json')['balanced'];results={};prediction_count=0
    base_rows={w['id']:read_json(OUT/'predictions/baseline'/f"{w['id']}.json") for w in project['windows']}
    learned_sources={'Learned head','Confirmed learned interval'}
    def proposal_report(predictions):
        result={}
        for category in LABEL_CATEGORIES:
            stages={}
            for name,stage in [('all',None),('classifier','classifier')]:
                proposed=tp=gold_seconds=0.
                for w in project['windows']:
                    rule=labels[w['id']]['categories'][category]
                    if rule['status']!='reviewed':continue
                    scored=difference([list(w['target'])],union(map(tuple,rule['ambiguous'])))
                    gold=intersect(union(map(tuple,rule['intervals'])),scored)
                    spans=intersect(candidates(predictions[w['id']]['trace'],category,stage),scored)
                    proposed+=length(spans);tp+=length(intersect(spans,gold));gold_seconds+=length(gold)
                stages[name]=dict(proposed_seconds=proposed,true_proposed_seconds=tp,
                                  precision=tp/proposed if proposed else None,recall=tp/gold_seconds if gold_seconds else None)
            result[category]=stages
        return result
    for n in ('baseline','linear','attention','confirmed'):
        manifest=read_json(OUT/'predictions'/n/'run.json')
        if not manifest['complete']:raise ValueError('Incomplete full-player comparison')
        reports=read_json(OUT/f'{n}.json')
        predictions={};identities={}
        for w in project['windows']:
            path=OUT/'predictions'/n/f"{w['id']}.json";row=read_json(path)
            if row.get('run_identity')!=manifest['identity'] or row.get('window')!=w['id']:
                raise ValueError('Prediction provenance mismatch')
            predictions[w['id']]=row;identities[w['id']]=file_hash(path)
        if manifest['expected_windows']!=[w['id'] for w in project['windows']]:
            raise ValueError('Incomplete validation window selection')
        recomputed={p:evaluate(project,labels,predictions,split='val',padding=p=='balanced') for p in ('balanced','none')}
        if recomputed!=reports:raise ValueError('Saved metrics differ from complete raw predictions')
        prediction_count+=len(predictions)
        replay={};variations={c:[] for c in LABEL_CATEGORIES};matches=True
        for w in project['windows']:
            baseline=base_rows[w['id']];actual=predictions[w['id']]
            additions=[e for e in actual['events'] if e['source'] in learned_sources]
            original=[e for e in actual['events'] if e['source'] not in learned_sources]
            replay[w['id']]=dict(baseline,events=baseline['events']+additions,
                trace=baseline['trace']+[e for e in actual['trace'] if e['stage'] in ('classifier','learned_confirmation')])
            for c in LABEL_CATEGORIES:
                if filtered(original,c,w['clip_duration'],True)!=filtered(baseline['events'],c,w['clip_duration'],True):
                    variations[c].append(w['id'])
                if filtered(actual['events'],c,w['clip_duration'],True)!=filtered(replay[w['id']]['events'],c,w['clip_duration'],True):matches=False
        replay_report=evaluate(project,labels,replay,split='val',padding=True)
        proposals=proposal_report(predictions)
        for c in LABEL_CATEGORIES:
            if abs(proposals[c]['all']['recall']-(reports['balanced']['categories'][c]['all']['candidate_recall'] or 0))>1e-9:
                raise ValueError('Proposal coverage disagrees with independent interval scorer')
        eligible,selection=category_choice(base,reports['balanced'])
        qualified,checks=promotion_check(base,reports['balanced'])
        durations=[r['seconds'] for r in predictions.values()]
        confirmation=Counter(t['state'] for r in predictions.values() for t in r['trace']
                             if t['stage']=='learned_confirmation')
        results[n]=dict(reports=reports,run_identity=manifest['identity'],prediction_identity=digest(identities),config=manifest['config'],
                        eligible_categories=eligible,qualification_passed=qualified,qualification_rows=checks,
                        fixed_baseline_head_replay=replay_report,head_replay_matches_full_automatic=matches,
                        proposals=proposals,
                        non_head_variation_windows={c:len(ids) for c,ids in variations.items()},
                        confirmation_checks=dict(confirmation),
                        timings=dict(recorded_clip_seconds_sum=sum(durations),recorded_clip_seconds_median=statistics.median(durations),
                                     recorded_clip_seconds_min=min(durations),recorded_clip_seconds_max=max(durations),
                                     resume_pass_wall_seconds=manifest['comparison_wall_seconds'],
                                     scope='Clip timings include context, cache reuse and concurrent work. Sum is service time, not elapsed time; resume wall excludes earlier partial passes. Neither establishes sustained playback speed.'),
                        execution_counts={mode:sum(r.get('execution_mode','initial-paired')==mode for r in predictions.values())
                                          for mode in ('initial-paired','isolated-configuration')})
    crop=read_json(ROOT/'roi'/REGIONS/'dataset.json',{})
    reviewed=read_json(ROOT/'roi'/REGIONS/'reviewed.json',{})
    queue=read_json(ROOT/'benchmarks/training-targeted-20261011-1/summary.json',{})
    summary=dict(date='2026-10-11',training_clips=113,validation_clips=42,validation_titles=14,
                 validation_scope='Repeatedly inspected positive-enriched human validation; not a fresh final test or 95% certificate',
                 choice=read_json(OUT/'choice.json'),comparisons=results,
                 region_review=dict(project=REGIONS,crops=len(crop.get('rows',[])),
                  reviewed=sum(r.get('labeler')=='human' and r.get('reviewed',False) for r in reviewed.get('rows',[])),
                  region_classifier_trained=False,reason='Human region labels required; scene intervals cannot assign crop labels'),
                 targeted_training=queue,player_defaults_changed=False,models_deployed=False,
                 timing_scope='Initial partial paired passes, then separate processes per configuration to avoid VRAM saturation; mixed timing scope, no isolated speed gain or sustained live claim',
                 checks=dict(human_labels_unchanged=True,validation_not_used_for_fitting=True,
                             expected_predictions=168,actual_predictions=prediction_count,all_comparisons_complete=prediction_count==168),
                 input_parity=read_json(OUT/'input-parity.json'),
                 attention_export={k:read_json(ROOT/'heads'/f"{NAMES['attention']}.json")[k] for k in (
                     'onnx_maximum_logit_difference','onnx_verified_batch_sizes','training_gpu_cpu_maximum_difference','calibration_inference')})
    write_json(OUT/'summary.json',summary);write_json(REPO/'benchmarks/accuracy-improvements-2026-10-11.json',summary)
    print('Recorded all improvements and full-player measurements; no model deployed',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['freeze','train','benchmark','report'])
    p.add_argument('--configuration',choices=['baseline','linear','attention','confirmed'])
    a=p.parse_args()
    if a.configuration and a.stage!='benchmark':p.error('--configuration applies only to benchmark')
    if a.stage=='benchmark':benchmark(a.configuration)
    else:{'freeze':freeze,'train':train,'report':report}[a.stage]()
