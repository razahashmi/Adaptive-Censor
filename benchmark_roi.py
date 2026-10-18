"""Same-window anatomy comparison with an optional pre-tracking region veto."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

from analysis_engine import Models,ScanConfig
from evaluate_labels import evaluate,load_projects
from labeling import project_dir,read_json,write_json
from model_setup import prepare_environment
from predict_project import predict_window,run_contract
from provenance import digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',nargs='+',required=True)
    parser.add_argument('--roi',required=True)
    parser.add_argument('--baseline-roi',default='',help='Compare with this existing region check instead of no check')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--baseline-from',type=Path,help='Reuse a complete, content-matching baseline from an earlier paired run')
    parser.add_argument('--resume',action='store_true',help='Reuse per-window checkpoints with an identical run contract')
    args=parser.parse_args();root=prepare_environment()
    project,labels,_=load_projects(root,args.name,[]);windows=project['windows']
    def source(window):
        name,identifier=window['id'].split('/',1)
        return project_dir(root,name)/'clips'/f'{identifier}.mp4'
    def contract(config):
        projects={}
        for name in args.name:
            folder=project_dir(root,name);original=read_json(folder/'project.json')
            projects[name]=run_contract(root,folder,original,original['windows'],config,args.workers,False)
        return dict(project_contracts=projects,identity=digest(projects),config=config.model_dump())
    configs={name:ScanConfig(speech=False,scenes=False,anatomy_model='640m',anatomy_roi=roi) for name,roi in [('baseline',args.baseline_roi),('roi',args.roi)]}
    contracts={name:contract(c) for name,c in configs.items()}
    models=Models(workers=args.workers)
    result=dict(scope='Anatomy-only paired benchmark; experimental ROI, not a qualification certificate',windows=len(windows),runs={})
    if args.baseline_from:
        previous=read_json(args.baseline_from)
        directory=args.baseline_from.parent/args.baseline_from.stem/'baseline'
        manifest=read_json(directory/'run.json')
        if not manifest.get('complete') or manifest.get('identity')!=contracts['baseline']['identity']:
            raise ValueError('Baseline is incomplete or code/assets/settings/sources changed')
        predictions={w['id']:read_json(directory/f"{w['id']}.json") for w in windows}
        if any(p.get('run_identity')!=manifest['identity'] for p in predictions.values()):
            raise ValueError('Baseline prediction identity mismatch')
        report=evaluate(project,labels,predictions,padding=True)
        if digest(report)!=digest(previous['runs']['baseline']['report']):
            raise ValueError('Baseline labels/predictions/evaluation changed')
        result['runs']['baseline']=dict(previous['runs']['baseline'],reference=str(args.baseline_from),recomputed=False)
    try:
        models.warm_up(('640m',))
        for name,config in configs.items():
            if name=='baseline' and args.baseline_from:
                continue
            directory=args.output.parent/args.output.stem/name
            existing_manifest=read_json(directory/'run.json',{})
            if existing_manifest and (not args.resume or existing_manifest.get('identity')!=contracts[name]['identity']):
                raise ValueError('Output exists or its identity changed; resume a matching run or choose another output')
            write_json(directory/'run.json',dict(contracts[name],complete=False,benchmark_only=True))
            started=time.perf_counter()
            outputs=[];reused=[];new_seconds=[]
            def predict(window):
                path=directory/f"{window['id']}.json"
                if args.resume and path.is_file():
                    try:
                        saved=read_json(path)
                    except (OSError,ValueError):
                        saved=None
                    if saved:
                        if saved.get('run_identity')!=contracts[name]['identity'] or saved.get('window')!=window['id']:
                            raise ValueError('Checkpoint identity mismatch')
                        reused.append(window['id'])
                        return saved
                prediction=dict(predict_window(models,source(window),window,config),run_identity=contracts[name]['identity'])
                temporary=path.with_suffix('.partial')
                write_json(temporary,prediction);temporary.replace(path)
                new_seconds.append(window['clip_duration'])
                return prediction
            with ThreadPoolExecutor(args.workers) as pool:
                for i,prediction in enumerate(pool.map(predict,windows),1):
                    outputs.append(prediction)
                    if i%6==0:print(f'{name}: {i}/{len(windows)}',flush=True)
            wall=time.perf_counter()-started
            report=evaluate(project,labels,{w['id']:p for w,p in zip(windows,outputs)},padding=True)
            if contract(config)['identity']!=contracts[name]['identity']:
                raise RuntimeError('Code/assets changed during ROI comparison')
            write_json(directory/'run.json',dict(contracts[name],complete=True,benchmark_only=True))
            result['runs'][name]=dict(wall_seconds=wall,processed_media_seconds=sum(w['clip_duration'] for w in windows),
                reused_windows=len(reused),new_media_seconds=sum(new_seconds),timing_scope='Resumed subset only' if reused else 'Full pass',
                cache_state='warm sessions, first anatomy pass' if name=='baseline' else
                            'warm anatomy sessions, first decoding and ROI pass' if args.baseline_from else 'repeat decoding, first ROI pass',report=report)
            print(name,report['categories']['nudity']['all'],flush=True)
        write_json(args.output,result)
    finally:models.close()


if __name__=='__main__':main()
