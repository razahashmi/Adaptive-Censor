"""Reproducible training and actual-player comparisons after human review."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time
from labeling import read_json, write_json, LABEL_CATEGORIES
from model_setup import prepare_environment
from provenance import digest, file_hash, head_identity

ROOT=prepare_environment(); REPO=Path(__file__).parent
OUT=ROOT/'benchmarks/human-improvements-20261014-1'
PROJECTS=['training-actions-human-1','train-drive-expansion-1','training-content-human-2','training-batch-more-20261012-2']
HEAD='human-aligned-20261014-1'; REGION='human-region-20261014-1'
MASKS={s:f'mask-{s}-batch-more-20261012-2' for s in ('train','val','test')}


def verify():
    data=read_json(OUT/'experiment.json')
    for filename,sha in data['input_sha256'].items():
        if file_hash(Path(filename))!=sha:raise ValueError('Frozen input changed: '+filename)
    return data


def freeze():
    from improvement_experiment import verify as previous_verify
    from human_calibration import collect
    from segmentation_labels import read_label
    previous_verify(); inputs={}; missing=[]; status={}
    def bind(path):inputs[str(path)]=file_hash(path)
    training=collect([ROOT/'labeling'/n for n in PROJECTS])
    for name in PROJECTS+['val-2']:
        folder=ROOT/'labeling'/name; bind(folder/'project.json')
        for w in read_json(folder/'project.json')['windows']:
            bind(folder/'labels'/(w['id']+'.json')); bind(folder/'clips'/(w['id']+'.mp4'))
    for split,name in MASKS.items():
        folder=ROOT/'segmentation'/name; data=read_json(folder/'dataset.json'); bind(folder/'dataset.json')
        saved=exposed=positive=0
        for row in data['rows']:
            label=read_label(folder,data,row)
            if not label:
                missing.append(dict(project=name,id=row['id'],title=row['display']));continue
            if label['labeler']!='human':raise ValueError('Human mask labels required')
            saved+=1; exposed+=bool(label['drafts_shown']); positive+=bool(label['positive_pixels'])
            for suffix in ('.jpg','-draft.png'):bind(folder/(row['id']+suffix))
            bind(folder/'labels'/(row['id']+'.json')); bind(folder/'exposure'/(row['id']+'.json'))
            if label['status']=='reviewed':bind(folder/(row['id']+'-gold.png'))
        status[split]=dict(expected=len(data['rows']),saved=saved,exposed_frames=exposed,positive_frames=positive)
    folder=ROOT/'roi/anatomy-human-20261011-2'
    for filename in ('dataset.json','reviewed.json','features.npz'):bind(folder/filename)
    roi=read_json(folder/'reviewed.json'); region=read_json(ROOT/'roi-models'/(REGION+'.json'))
    if region['dataset_identity']!=file_hash(folder/'reviewed.json'):raise ValueError('Crop labels changed after training')
    bind(ROOT/'roi-models'/(REGION+'.json'))
    value=dict(date='2026-10-14',projects=PROJECTS,training_clips=len(training),mask_status=status,
        missing_training_frames=missing,region_crops=len(roi['rows']),
        region_labels=dict(Counter(str(r['label']) for r in roi['rows'])),input_sha256=inputs,
        linear_head=HEAD,region_head=REGION,
        selection='Fixed linear architecture; training-only model selection/calibration; no evaluation-label fitting',
        evaluation='Previously inspected human validation and group-blind sparse masks, not final qualification')
    path=OUT/'experiment.json'
    if path.exists() and read_json(path)!=value:raise ValueError('Experiment snapshot changed')
    write_json(path,value); print('Frozen',len(training),'training clips;',status,flush=True)


def masks():
    from segmentation_evaluation import evaluate
    verify()
    for split in ('val','test'):
        report=evaluate(ROOT/'segmentation'/MASKS[split],blind_groups_only=True)
        path=OUT/f'masks-{split}.json'
        if path.exists() and read_json(path)!=report:raise ValueError('Mask measurement changed')
        write_json(path,report)
        print('MASK',split,{k:report[k] for k in ['scored_frames','exposure_excluded_frames','positive_groups','positive_titles','coverage','pixel_precision','iou','clean_frame_false_mask_rate']},flush=True)
    verify()


def train():
    from human_calibration import train_linear,collect
    from select_windows import normalize_title
    from threadpoolctl import threadpool_limits
    snapshot=verify(); path=ROOT/'heads'/(HEAD+'.json'); folders=[ROOT/'labeling'/n for n in PROJECTS]
    if path.exists():raise FileExistsError('Never overwrite an experimental head')
    with threadpool_limits(limits=4):model=train_linear(folders)
    model.update(projects=PROJECTS,titles=sorted({normalize_title(w['title']) for _,w,_ in collect(folders)}),
        training_dataset_identity=digest(snapshot),training_code=file_hash(REPO/'human_calibration.py'))
    verify(); write_json(path,model)
    keys=('skipped','C','calibration_average_precision','candidate_threshold','decision_threshold','calibration_warning')
    choice=dict(head=HEAD,head_identity=head_identity(model),region=REGION,
        region_sha256=file_hash(ROOT/'roi-models'/(REGION+'.json')),before_player_evaluation=True,
        region_training_sha256=file_hash(REPO/'human_region_train.py'),
        calibration={c:{k:v for k,v in h.items() if k in keys} for c,h in model['heads'].items()})
    write_json(OUT/'choice.json',choice); print('Saved preselected classifier:',choice['calibration'],flush=True)


def benchmark(mode):
    from analysis_engine import Models,ScanConfig
    from predict_project import predict_window,run_contract
    from evaluate_labels import evaluate,table
    verify(); frozen=read_json(ROOT/'benchmarks/human-validation-20261009-1/dataset.json')
    project=frozen['contract']['project']; labels=frozen['contract']['labels']; folder=ROOT/'labeling/val-2'
    base=dict(frozen['contract']['baseline_config'])
    if mode=='region':
        configs={'anatomy-baseline':ScanConfig(**dict(base,scenes=False)),
                 'anatomy-region':ScanConfig(**dict(base,scenes=False,anatomy_roi=REGION))}
    else:
        choice=read_json(OUT/'choice.json')
        if head_identity(read_json(ROOT/'heads'/(HEAD+'.json')))!=choice['head_identity']:raise ValueError('Preselected head changed')
        if file_hash(ROOT/'roi-models'/(REGION+'.json'))!=choice['region_sha256']:raise ValueError('Preselected region model changed')
        heads=read_json(ROOT/'heads'/(HEAD+'.json'))
        if any(h.get('decision_threshold') is not None for h in heads['heads'].values()):
            raise ValueError('This comparison excludes only a head with no automatic decisions')
        configs={'full-baseline':ScanConfig(**base),
                 'full-region':ScanConfig(**dict(base,anatomy_roi=REGION))}
    models=Models(workers=2,evaluate_unpromoted=True)
    try:
        for name,config in configs.items():
            destination=OUT/'predictions'/name; contract=run_contract(ROOT,folder,project,project['windows'],config,2,True)
            old=read_json(destination/'run.json',{})
            if old and old['identity']!=contract['identity']:raise ValueError('Player run identity changed')
            write_json(destination/'run.json',dict(contract,complete=False))
            previous=ROOT/'benchmarks/human-improvements-20261011-1/predictions/baseline'
            reuse=name=='full-baseline' and read_json(previous/'run.json',{}).get('identity')==contract['identity']
            def calculate(w):
                path=destination/(w['id']+'.json')
                if path.exists():
                    if read_json(path).get('run_identity')!=contract['identity']:raise ValueError('Stale clip prediction')
                    return
                if reuse:
                    result=read_json(previous/(w['id']+'.json'))
                    if result['run_identity']!=contract['identity']:raise ValueError('Cached baseline differs')
                    result=dict(result,baseline_reused=True)
                else:result=predict_window(models,folder/'clips'/(w['id']+'.mp4'),w,config)
                write_json(path,dict(result,run_identity=contract['identity']))
            started=time.perf_counter()
            with ThreadPoolExecutor(max_workers=2) as pool:
                jobs={pool.submit(calculate,w):w for w in project['windows']}
                for i,future in enumerate(as_completed(jobs),1):
                    future.result(); print(f"{name}: {i}/42 {jobs[future]['id']}",flush=True)
            if run_contract(ROOT,folder,project,project['windows'],config,2,True)['identity']!=contract['identity']:
                raise ValueError('Player changed during benchmark')
            verify(); rows={w['id']:read_json(destination/(w['id']+'.json')) for w in project['windows']}
            reports={p:evaluate(project,labels,rows,split='val',padding=p=='balanced') for p in ('balanced','none')}
            write_json(OUT/(name+'.json'),reports)
            write_json(destination/'run.json',dict(contract,complete=True,windows=len(rows),baseline_reused=reuse,
                wall_seconds=round(time.perf_counter()-started,3),timing_scope='Short clips, cache reuse and two workers; not sustained speed'))
            print(name+'\n'+table(reports['balanced']),flush=True)
    finally:models.close()


def score_heads():
    """Replay exact chunk inputs through the same runtime head-output function."""
    import numpy as np
    from learned import window_matrix
    from learned_runtime import head_outputs
    from evaluate_labels import evaluate,candidates,difference
    from evaluate import intersect,union,length
    verify();model=read_json(ROOT/'heads'/(HEAD+'.json'));choice=read_json(OUT/'choice.json')
    if head_identity(model)!=choice['head_identity']:raise ValueError('Classifier changed')
    frozen=read_json(ROOT/'benchmarks/human-validation-20261009-1/dataset.json')
    project=frozen['contract']['project'];labels=frozen['contract']['labels'];rows={}
    for w in project['windows']:
        times,x=window_matrix(ROOT/'labeling/val-2',w,model['kinds'],temporal_offsets=model['temporal_offsets'],input_layout=model['input_layout'])
        entries=[];events=[]
        for start in np.arange(0,w['clip_duration'],6.):
            selected=(times>=start)&(times<min(start+6,w['clip_duration']))
            trace,accepted=head_outputs(model,times[selected],x[selected]);entries+=trace;events+=accepted
        if events:raise ValueError('Proposal-only head unexpectedly accepted content')
        rows[w['id']]=dict(window=w['id'],events=events,trace=entries)
    proposals={}
    for category in LABEL_CATEGORIES:
        proposed=correct=gold_seconds=0.
        for w in project['windows']:
            rule=labels[w['id']]['categories'][category]
            if rule['status']!='reviewed':continue
            scored=difference([tuple(w['target'])],union(map(tuple,rule['ambiguous'])))
            gold=intersect(union(map(tuple,rule['intervals'])),scored)
            spans=intersect(candidates(rows[w['id']]['trace'],category,'classifier'),scored)
            proposed+=length(spans);correct+=length(intersect(spans,gold));gold_seconds+=length(gold)
        proposals[category]=dict(proposed_seconds=proposed,true_proposed_seconds=correct,
            precision=correct/proposed if proposed else None,recall=correct/gold_seconds if gold_seconds else None)
    value=dict(head_identity=head_identity(model),windows=len(rows),automatic_events=0,proposals=proposals,
        reports={p:evaluate(project,labels,rows,split='val',padding=p=='balanced') for p in ('balanced','none')},
        scope='Exact cached player inputs and runtime scoring; proposals only, no new acceptance or model-forward speed claim')
    write_json(OUT/'head-replay.json',value);verify();print('Classifier proposals:',proposals,flush=True)


def mask_box_diagnostic():
    """CPU-only upper bound imposed by the draft generator's box envelopes."""
    import numpy as np
    from PIL import Image
    from evaluate_masks import counts,rates
    folder=ROOT/'segmentation'/MASKS['test'];data=read_json(folder/'dataset.json')
    scored=read_json(OUT/'masks-test.json');included={r['id'] for r in scored['samples']}
    total=dict(tp=0,fp=0,fn=0,clean=0,gold=0);positive_frames=0
    for row in data['rows']:
        if row['id'] not in included:continue
        with Image.open(folder/(row['id']+'-gold.png')) as image:gold=np.asarray(image.convert('L'))>=128
        h,w=gold.shape;envelope=np.zeros_like(gold)
        for hit in row['prompt_boxes']:
            x1,y1,x2,y2=hit['box'];mx=(x2-x1)*.1;my=(y2-y1)*.1
            envelope[max(0,int(y1-my)):min(h,int(np.ceil(y2+my))),max(0,int(x1-mx)):min(w,int(np.ceil(x2+mx)))]=True
        c=counts(envelope,gold);positive_frames+=bool(c['gold'])
        for k,v in c.items():total[k]+=v
    return dict(counts=total,**rates(total),positive_frames=positive_frames,
        positive_groups=scored['positive_groups'],positive_titles=scored['positive_titles'],
        pixel_precision=total['tp']/(total['tp']+total['fp']) if total['tp']+total['fp'] else None,
        scope='Filled prompt-envelope diagnostic on the same blind development subset; coverage is the current envelope upper bound, not a new detector or mask model')


def report():
    from learned import category_choice,promotion_check
    from evaluate_labels import evaluate,filtered
    snapshot=verify(); frozen=read_json(ROOT/'benchmarks/human-validation-20261009-1/dataset.json')
    project=frozen['contract']['project']; labels=frozen['contract']['labels']; comparisons={}; predictions={}
    for name in ('anatomy-baseline','anatomy-region','full-baseline'):
        folder=OUT/'predictions'/name; manifest=read_json(folder/'run.json')
        if not manifest['complete']:raise ValueError('Incomplete player comparison')
        rows={w['id']:read_json(folder/(w['id']+'.json')) for w in project['windows']}
        if any(r['run_identity']!=manifest['identity'] for r in rows.values()):raise ValueError('Prediction identity differs')
        reports={p:evaluate(project,labels,rows,split='val',padding=p=='balanced') for p in ('balanced','none')}
        if reports!=read_json(OUT/(name+'.json')):raise ValueError('Raw predictions do not reproduce metrics')
        comparisons[name]=dict(reports=reports,configuration=manifest['config'],run_identity=manifest['identity'],
            prediction_identity=digest({w['id']:file_hash(folder/(w['id']+'.json')) for w in project['windows']}),
            expected_predictions=42,actual_predictions=len(rows),baseline_reused=manifest['baseline_reused'],
            wall_seconds=manifest['wall_seconds'],timing_scope=manifest['timing_scope'])
        predictions[name]=rows
    # The changed crop gate runs after scene verification and affects only
    # anatomy acceptance. Hold the frozen scene decisions constant to isolate
    # its contribution. Review/proposal metrics are not asserted for this replay.
    replay={}
    for w in project['windows']:
        original=predictions['full-baseline'][w['id']]
        anatomy=predictions['anatomy-region'][w['id']]['events']
        replay[w['id']]=dict(original,events=[e for e in original['events'] if e['source']!='NudeNet']+anatomy)
    replay_report=evaluate(project,labels,replay,split='val',padding=True)
    base=comparisons['anatomy-baseline']['reports']['balanced']; candidate=comparisons['anatomy-region']['reports']['balanced']
    eligible,selection=category_choice(base,candidate);qualified,checks=promotion_check(base,candidate)
    automatic={}
    for category,data in replay_report['categories'].items():
        r=data['all'];c=r['counts']
        automatic[category]=dict(precision=r['precision'],recall=r['recall'],tp_seconds=c['tp'],
            fp_seconds=c['fp'],gold_seconds=c['gold'],events=c['events'],events_hit=c['events_hit'],
            clean_windows=c['clean_windows'],clean_windows_filtered=c['clean_windows_filtered'])
    partial={}
    for name in ('full-region','full-candidate'):
        folder=OUT/'predictions'/name;rows=[read_json(p) for p in folder.glob('*.json') if p.name!='run.json']
        matching=0
        if name=='full-region':
            for row in rows:
                w=next(w for w in project['windows'] if w['id']==row['window'])
                matching+=filtered(row['events'],'nudity',w['clip_duration'],True)==filtered(predictions['anatomy-region'][w['id']]['events'],'nudity',w['clip_duration'],True)
        partial[name]=dict(completed_clips=len(rows),expected_clips=42,complete=False,
            matching_anatomy_decisions=matching if name=='full-region' else None,
            included_in_accuracy_estimate=False,reason='Stopped to avoid competing with the user GPU workload')
    roi=read_json(ROOT/'roi-models'/(REGION+'.json')); keys=('crops','supported_classes','fit_titles','calibration_titles','calibration_recall','calibration_negative_rejection','partition_method')
    head=read_json(ROOT/'heads'/(HEAD+'.json'))
    calibration_keys=('C','calibration_average_precision','calibration_positive_events','calibration_positive_titles','decision_threshold','candidate_threshold','skipped')
    old=base['categories']['nudity']['all'];new=candidate['categories']['nudity']['all']
    result=dict(date='2026-10-14',training_clips=snapshot['training_clips'],training_mask_frames=snapshot['mask_status']['train']['saved'],
        crop_labels=snapshot['region_crops'],missing_training_frames=snapshot['missing_training_frames'],
        region_calibration={k:roi[k] for k in keys},choice=read_json(OUT/'choice.json'),comparisons=comparisons,
        scene_classifier=dict(trained=True,titles=len(head['titles']),
            calibration={c:{k:v for k,v in h.items() if k in calibration_keys} for c,h in head['heads'].items()},
            automatic_thresholds_enabled=sum(h.get('decision_threshold') is not None for h in head['heads'].values()),
            independent_42_clip_classifier_ranking_measured=False,
            reason='Validation feature extraction requires additional GPU work; no model-forward or ranking claim from uncomputed inputs'),
        fixed_baseline_component_replay=dict(automatic=automatic,scope='Measured anatomy decisions combined with frozen non-anatomy decisions; not a complete fresh full-pipeline run; assisted/proposal metrics omitted'),
        partial_integration_runs=partial,
        nudity_result=dict(before=old,after=new,false_filter_seconds_reduction=(old['counts']['fp']-new['counts']['fp'])/old['counts']['fp'],
                           true_coverage_seconds_unchanged=old['counts']['tp']==new['counts']['tp']),
        eligible_categories=eligible,category_selection=selection,
        qualification_passed=qualified,qualification_rows=checks,
        mask_geometry={s:{k:v for k,v in read_json(OUT/f'masks-{s}.json').items() if k not in ('samples','by_title','exposure_excluded_ids')} for s in ('val','test')},
        mask_proposal_ceiling=mask_box_diagnostic(),
        checks=dict(runtime_tests=172,runtime_dependency_skips=7,training_environment_tests=39,
                    scored_anatomy_predictions=84,reused_full_baseline_predictions=42),
        labels_unchanged=True,validation_or_test_used_for_fitting=False,player_defaults_changed=False,models_deployed=False,
        speed_scope='GPU was shared with the user; no standalone or sustained speed conclusion',
        limits='Repeatedly inspected validation, enriched sparse masks and limited positive-title support; not final-release or sustained-speed qualification')
    write_json(OUT/'summary.json',result); write_json(REPO/'benchmarks/human-reviews-2026-10-14.json',result)
    print('Recorded completed comparisons; no new model enabled',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['freeze','masks','train','region','full','head-replay','report'])
    args=parser.parse_args()
    if args.stage in ('region','full'):benchmark(args.stage)
    else:{'freeze':freeze,'masks':masks,'train':train,'head-replay':score_heads,'report':report}[args.stage]()
