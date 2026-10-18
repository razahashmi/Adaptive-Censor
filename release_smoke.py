"""Exercise extracted code with generated test fixtures and an empty offline runtime."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from uuid import uuid4
import zipfile

import httpx
from build_source_release import verify


def smoke(archive):
    identity = verify(archive)
    checks = []
    with tempfile.TemporaryDirectory(prefix='adaptive-release-') as temporary:
        work = Path(temporary)
        # verify() has rejected traversal, links, unreviewed media and unlisted entries.
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(work/'source')
        source = work/'source'/f"adaptive-censor-{identity['version']}"
        runtime = work/'runtime'
        environment = os.environ.copy()
        environment.pop('PYTHONPATH', None)
        environment.update(ADAPTIVE_CENSOR_WORKDIR=str(runtime), HF_HOME=str(runtime/'hf'),
                           HF_HUB_CACHE=str(runtime/'hf'/'hub'), HUGGINGFACE_HUB_CACHE=str(runtime/'hf'/'hub'),
                           TRANSFORMERS_CACHE=str(runtime/'hf'/'transformers'), CUDA_CACHE_PATH=str(runtime/'cuda'),
                           HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
        with socket.socket() as listener:
            listener.bind(('127.0.0.1',0));port=listener.getsockname()[1]
        base=f'http://127.0.0.1:{port}'
        process=None
        with (work/'server.log').open('wb') as log:
            def start():
                nonlocal process
                process=subprocess.Popen([sys.executable,str(source/'server.py'),'--port',str(port)],
                    cwd=source,env=environment,stdin=subprocess.DEVNULL,stdout=log,stderr=log,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                deadline=time.monotonic()+45
                while time.monotonic()<deadline:
                    if process.poll() is not None:
                        raise RuntimeError('Extracted server exited before readiness; inspect the artifact locally')
                    try:
                        if httpx.get(base+'/api/bootstrap',timeout=1).status_code==200:return
                    except httpx.HTTPError:pass
                    time.sleep(.2)
                raise RuntimeError('Extracted server did not become ready')

            def stop():
                if process and process.poll() is None:
                    process.terminate()
                    try:process.wait(timeout=10)
                    except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)

            try:
                start();checks.append('extracted_source_empty_offline_runtime')
                with httpx.Client(base_url=base,timeout=30) as client:
                    assert client.get('/api/feedback').status_code==403
                    token=client.get('/api/bootstrap').json()['token']
                    headers={'x-censor-token':token}
                    assert client.post('/api/demo').status_code==403
                    assert client.post('/api/demo',headers={**headers,'origin':'https://untrusted.invalid'}).status_code==403
                    checks.append('cookie_token_and_origin_checks')
                    for asset in ('/','/feedback.html','/feedback-page.mjs','/batch.json','/readiness.html','/readiness.mjs'):
                        assert client.get(asset).status_code==200,asset
                    assert client.get('/batch.json').json()['tasks']==[]
                    assert client.get('/api/labeling').json()==[]
                    assert client.get('/api/roi-projects').json()==[]
                    assert client.get('/api/segmentation').json()==[]
                    checks.append('assets_and_empty_public_data_queues')
                    response=client.post('/api/demo',headers=headers);assert response.status_code==200,response.text
                    job=response.json();assert 11.9<job['duration']<12.1
                    assert not job['config']['speech'] and not job['config']['nudity'] and not job['config']['scenes']
                    assert client.get(f"/api/media/{job['id']}/source").status_code==200
                    checks.append('generated_demo_without_model_downloads')
                    setup=client.get('/api/readiness')
                    assert setup.status_code==200 and setup.json()['demo_ready']
                    assert not setup.json()['inference_performed'] and not setup.json()['downloads_performed']
                    checks.append('setup_diagnostics_without_inference')
                    categories=('nudity','sexual_activity','violence','gore','kissing','language','custom')
                    policy=dict(pre_padding=.3,post_padding=.35,categories={category:dict(visual='blur',audio='mute',
                        blur=50,min_confidence=0,min_severity=1) for category in categories})
                    body=dict(id=str(uuid4()),session_id=job['id'],kind='wrongly_censored',position=3,
                        generation=job['generation'],snapshot_updated=job['updated'],policy=policy,
                        viewer_action=dict(visual='blur',audio='mute',blur=50,regional=False))
                    first=client.post('/api/feedback',json=body,headers=headers)
                    assert first.status_code==200,first.text
                    row=first.json()
                    assert client.post('/api/feedback',json=body,headers=headers).json()['id']==row['id']
                    assert client.get('/api/feedback').json()['total']==1
                    assert row['evidence']['status']=='none'
                    assert client.post('/api/feedback/'+row['id']+'/evidence',headers=headers).status_code==200
                    deadline=time.monotonic()+20
                    while time.monotonic()<deadline:
                        retained=client.get('/api/feedback/'+row['id']).json()
                        if retained['evidence']['status'] not in ('queued','saving'):break
                        time.sleep(.1)
                    assert retained['evidence']['status']=='ready',retained['evidence']
                    assert client.get('/api/feedback/'+row['id']+'/evidence').status_code==200
                    response=client.patch('/api/feedback/'+row['id'],json=dict(revision=1,category='violence',
                        note='Generated release smoke test',status='confirmed'),headers=headers)
                    assert response.status_code==200,response.text
                    assert not response.json()['verified_label']
                    response=client.post('/api/feedback/'+row['id']+'/labeling',json=dict(title='Generated controls demo',start=.2,end=4),headers=headers)
                    assert response.status_code==200,response.text
                    task=response.json()['evidence']['task'];assert not task['complete']
                    assert client.post('/api/feedback/'+row['id']+'/training',headers=headers).status_code==409
                    # Synthetic control data exists only inside this temporary smoke runtime.
                    label_categories=('nudity','sexual_activity','violence','gore','kissing')
                    fake_labels=dict(labeler='human',categories={category:dict(status='reviewed',intervals=[],ambiguous=[]) for category in label_categories})
                    response=client.put(f"/api/labeling/{task['project']}/labels/{task['window']}",json=fake_labels,headers=headers)
                    assert response.status_code==200 and response.json()['drafts_shown']
                    assert client.post('/api/feedback/'+row['id']+'/training',headers=headers).status_code==409
                    (runtime/'labeling/library.json').write_text(json.dumps(dict(splits={'generated controls demo':'train'})),encoding='utf-8')
                    response=client.post('/api/feedback/'+row['id']+'/training',headers=headers)
                    assert response.status_code==200 and not response.json()['trained'] and not response.json()['model_changed']
                    checks.append('retained_clip_human_label_ingress_and_training_guard')
                    assert len(client.get('/api/feedback/export').text.splitlines())==1
                    response=client.patch('/api/feedback/'+row['id'],json=dict(revision=2,status='withdrawn'),headers=headers)
                    assert response.status_code==200,response.text
                    assert client.get('/api/feedback/export').text==''
                    checks.append('feedback_save_idempotency_review_undo_and_export')
                    response=client.post(f"/api/media/{job['id']}/export",json=policy,headers=headers)
                    assert response.status_code==200,response.text
                    export_id=response.json()['id'];deadline=time.monotonic()+45
                    while time.monotonic()<deadline:
                        state=client.get('/api/exports/'+export_id).json()
                        if state['status']!='rendering':break
                        time.sleep(.2)
                    assert state['status']=='complete',state
                    result=client.get('/api/exports/'+export_id+'/download')
                    assert result.status_code==200 and len(result.content)>1000
                    checks.append('filtered_mp4_export')
                stop();start()
                with httpx.Client(base_url=base,timeout=15) as client:
                    data=client.get('/api/bootstrap').json()
                    assert any(session['id']==job['id'] for session in data['sessions'])
                    row=client.get('/api/feedback/'+row['id']).json()
                    assert row['status']=='withdrawn' and row['note']=='Generated release smoke test'
                    assert row['media_available']
                    assert not data['models_ready']
                    checks.append('saved_media_and_feedback_survive_restart')
                    headers={'x-censor-token':data['token']}
                    assert client.delete('/api/media/'+job['id'],headers=headers).status_code==200
                    retained=client.get('/api/feedback/'+row['id']).json()
                    assert not retained['media_available'] and retained['evidence']['status']=='ready'
                    assert client.get('/api/feedback/'+row['id']+'/evidence').status_code==200
                    assert client.delete('/api/feedback/'+row['id']+'/evidence',headers=headers).status_code==200
                    assert client.get(f"/api/labeling/{task['project']}/clip/{task['window']}").status_code==200
                    checks.append('retained_clip_survives_source_removal_and_label_copy_survives_clip_removal')
            finally:
                stop()
    return dict(artifact=Path(archive).name, **identity, python=sys.version.split()[0],
                dependency_environment='caller-provided; fresh runtime, not necessarily a fresh dependency install',
                checks=checks, passed=True, model_accuracy_tested=False, gpu_compatibility_tested=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive',type=Path)
    parser.add_argument('--report',type=Path)
    args=parser.parse_args();result=smoke(args.archive)
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
