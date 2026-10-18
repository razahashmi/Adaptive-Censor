"""Small real-clip checks of shared decoding, V-JEPA batches and player throughput.

This is a smoke/performance benchmark, not a whole-episode accuracy claim.
Reference source is optional, used only for exact decoder parity.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
from pathlib import Path
import time

import numpy as np
from analysis_engine import Models, ScanConfig, native_frames, sample_frames, sample_burst
from core import compile_plan, default_policy
from evaluate_labels import evaluate
from labeling import project_dir, read_json, write_json
from learned_runtime import squash_frames, vjepa_clips
from model_setup import prepare_environment
from predict_project import predict_window, run_contract


def decoding(clip, reference):
    spec=importlib.util.spec_from_file_location('reference_analysis',reference)
    old=importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name]=old
    spec.loader.exec_module(old)
    cases=[('native_frames',(0,8,12)),('sample_frames',(0,8,2)),('sample_burst',(3.0,))]
    results={}
    for name,args in cases:
        started=time.perf_counter(); expected=getattr(old,name)(clip,*args); before=time.perf_counter()-started
        started=time.perf_counter(); actual=globals()[name](clip,*args); after=time.perf_counter()-started
        if name=='sample_burst':
            assert expected[1]==actual[1]
            expected,actual=expected[0],actual[0]
        assert len(expected)==len(actual)
        assert all(t==u and np.array_equal(p,q) for (t,p),(u,q) in zip(expected,actual))
        results[name]=dict(frames=len(actual),reference_seconds=before,shared_seconds=after,exact_pixels=True)
    return results


def batches(models, clip):
    features=models.learned_features()
    times,pixels=squash_frames(native_frames(clip,0,6,8))
    clips=vjepa_clips(times,pixels,[2,2.5,3,3.5])
    session=features.session('vjepa')
    results={}
    reference=None
    for batch in (1,2):
        inputs=[np.stack(clips[i:i+batch]).astype(np.float16) for i in range(0,len(clips),batch)]
        # Warm this shape before timed runs; report cold setup separately.
        cold=time.perf_counter()
        output=np.concatenate([session.run(None,{'frames':v})[0] for v in inputs])
        cold=time.perf_counter()-cold
        timings=[]
        for _ in range(3):
            started=time.perf_counter()
            output=np.concatenate([session.run(None,{'frames':v})[0] for v in inputs])
            timings.append(time.perf_counter()-started)
        if reference is None: reference=output.copy()
        cosine=np.sum(output*reference,1)/(np.linalg.norm(output,axis=1)*np.linalg.norm(reference,axis=1)+1e-9)
        results[str(batch)]=dict(first_seconds=cold,median_seconds=float(np.median(timings)),
                                 clips=len(clips),minimum_cosine=float(cosine.min()),max_difference=float(np.abs(reference-output).max()))
    return results


def concurrency_parity(windows, first, second):
    """Compare decisions and geometry, allowing only tiny diagnostic-score jitter."""
    if not len(windows)==len(first)==len(second):
        raise ValueError('Worker comparison is missing predictions')
    rows=[]
    for window,a,b in zip(windows,first,second):
        trace_equal=len(a['trace'])==len(b['trace'])
        largest_difference=0.
        for left,right in zip(a['trace'],b['trace']):
            left,right=dict(left),dict(right)
            if 'score' in left and 'score' in right:
                before,after=left.pop('score'),right.pop('score')
                difference=abs(before-after)
                largest_difference=max(largest_difference,difference)
                trace_equal=trace_equal and bool(np.isfinite(difference)) and difference<=1e-7
            trace_equal=trace_equal and left==right
        policy=default_policy()
        rows.append(dict(window=window['id'],events_identical=a['events']==b['events'],
                         plan_identical=compile_plan(a['events'],policy,window['clip_duration'])==
                                        compile_plan(b['events'],policy,window['clip_duration']),
                         trace_identical=a['trace']==b['trace'],trace_within_tolerance=trace_equal,
                         maximum_trace_score_difference=largest_difference))
    return dict(passed=all(row['events_identical'] and row['plan_identical'] and row['trace_within_tolerance'] for row in rows),
                trace_score_absolute_tolerance=1e-7,windows=rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name',default='val-1');p.add_argument('--limit',type=int,default=6)
    p.add_argument('--reference',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();root=prepare_environment();folder=project_dir(root,args.name)
    project=read_json(folder/'project.json')
    selected, titles = [], set()
    for category in ('sexual_activity','nudity','violence','gore','kissing'):
        for window in project['windows']:
            label=read_json(folder/'labels'/f"{window['id']}.json",{})
            if window['title'] not in titles and label.get('categories',{}).get(category,{}).get('intervals'):
                selected.append(window); titles.add(window['title']); break
    for window in project['windows']:
        if window['title'] not in titles:
            selected.append(window); titles.add(window['title'])
    windows=selected[:args.limit]
    config=ScanConfig(speech=False,anatomy_model='640m',scene_mode='learned',learned_heads='heads-3',
                      learned_verification='off',scene_filtering='automatic')
    result=dict(scope='Positive-enriched clips from distinct validation titles, selected using AI labels; experimental, not blind or unbiased; no automatic qualification',
                config=config.model_dump(),windows=[w['id'] for w in windows])
    initial_contract=run_contract(root,folder,project,windows,config,2,True)
    if args.reference:
        result['decoding']=decoding(folder/'clips'/f"{windows[0]['id']}.mp4",args.reference)
        print('Shared decoder parity passed',flush=True)
    models=Models(workers=2,evaluate_unpromoted=True)
    try:
        started=time.perf_counter();models.warm_up(('640m',));models.learned_features().warm_up(['vjepa','ast'])
        result['session_setup_seconds']=time.perf_counter()-started
        result['batches']=batches(models,folder/'clips'/f"{windows[0]['id']}.mp4")
        print('V-JEPA batch parity/timing measured',flush=True)
        first_outputs=None
        for workers in (1,2):
            # Explicit repeat-cache measurement; never label it first-view sustained speed.
            started=time.perf_counter()
            with ThreadPoolExecutor(workers) as pool:
                outputs=list(pool.map(lambda w:predict_window(models,folder/'clips'/f"{w['id']}.mp4",w,config),windows))
            wall=time.perf_counter()-started
            if first_outputs is None:first_outputs=outputs
            else:result['concurrency_parity']=concurrency_parity(windows,first_outputs,outputs)
            media=sum(w['clip_duration'] for w in windows)
            labels={w['id']:read_json(folder/'labels'/f"{w['id']}.json") for w in windows}
            report=evaluate(dict(windows=windows),labels,{w['id']:r for w,r in zip(windows,outputs)},padding=True)
            result[f'workers_{workers}']=dict(wall_seconds=wall,processed_media_seconds=media,speed=media/wall,
                                             cache_state='partially warm' if workers==1 else 'repeat-cache',report=report)
            contract=run_contract(root,folder,project,windows,config,workers,True)
            directory=args.output.parent/args.output.stem/f'workers-{workers}'
            write_json(directory/'run.json',dict(contract,complete=True,benchmark_only=True))
            for window,prediction in zip(windows,outputs):
                write_json(directory/f"{window['id']}.json",dict(prediction,run_identity=contract['identity']))
            print(f'{workers} workers: {media:.1f}s in {wall:.1f}s wall',flush=True)
        final_contract=run_contract(root,folder,project,windows,config,2,True)
        if final_contract['identity'] != initial_contract['identity']:
            raise RuntimeError('Code or assets changed during the benchmark; rerun before using these results')
        result['contract']=final_contract
        write_json(args.output,result)
        if not result['concurrency_parity']['passed']:
            raise RuntimeError('Worker-count parity failed; inspect the saved decisions and regions before using the timing result')
        print(f'Wrote {args.output}')
    finally:models.close()


if __name__=='__main__':main()
