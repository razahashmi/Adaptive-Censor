import {categories, labels} from './policy.mjs';

export function installQueue({document, request}) {
  const $=id=>document.getElementById(id), video=$('report-video');
  let offset=0, filter='pending', selected=null, rows=[], total=0;
  let loading=false, saving=false, polling=false, dirty=false, taskDirty=false, clipPlayback=false;
  const time=t=>{const ticks=Math.round(t*100);return `${Math.floor(ticks/6000)}:${((ticks%6000)/100).toFixed(2).padStart(5,'0')}`;};
  const sceneCategories=categories.filter(c=>!['language','custom'].includes(c));
  for(const category of categories) {
    const option=document.createElement('option');option.value=category;option.textContent=labels[category];
    $('review-category').append(option);
  }
  function clearVideo() {
    video.onloadedmetadata=null;video.onerror=null;video.pause();video.removeAttribute('src');video.load();video.hidden=true;clipPlayback=false;
  }
  function edited() {
    $('review-message').textContent='Save or discard your edits before moving to another report.';
    return true;
  }
  function controls() {
    const busy=loading||saving, evidence=selected?.evidence||{status:'none'}, ready=evidence.status==='ready';
    $('report-refresh').disabled=$('report-filter').disabled=busy;
    $('report-prev').disabled=busy||offset===0;$('report-next').disabled=busy||offset+rows.length>=total;
    for(const name of ['review-category','review-status','review-note','review-save','review-discard'])$(name).disabled=busy;
    $('review-discard').hidden=!dirty&&!taskDirty;
    $('report-watch').disabled=busy||!(ready||selected?.media_available);
    $('report-keep').disabled=busy||ready||['saving','queued'].includes(evidence.status)||!selected?.media_available||selected?.context.live||selected?.status==='withdrawn';
    $('report-remove-clip').hidden=['none','missing','removed'].includes(evidence.status);
    $('report-remove-clip').disabled=busy||['queued','saving'].includes(evidence.status);
    $('label-task-panel').hidden=!ready&&!evidence.task;
    for(const id of ['label-title','label-start','label-end'])$(id).disabled=busy||!!evidence.task||!ready;
    $('create-label-task').disabled=busy||!!evidence.task||!ready||selected?.status!=='confirmed'||!sceneCategories.includes(selected?.category);
    $('label-mark-start').disabled=$('label-mark-end').disabled=busy||!ready||!!evidence.task||!clipPlayback;
    $('open-label-task').hidden=!evidence.task;
    if(evidence.task)$('open-label-task').href=evidence.task.url;
    $('promote-labels').hidden=!evidence.task?.complete||evidence.task.promoted;
    $('promote-labels').disabled=busy||selected?.status!=='confirmed';
  }
  function evidenceView() {
    const evidence=selected?.evidence||{status:'none'};
    $('report-watch').textContent=evidence.status==='ready'?'Review saved clip (uncensored)':'Review original (uncensored)';
    const messages={none:'No clip retained. Keep one to preserve the moment for precise labeling.',
      queued:'Local clip is queued. Playback and reports can continue.',saving:'Saving the short clip on CPU…',
      ready:`Clip retained locally · ${Number(evidence.duration||0).toFixed(2)}s · ${((evidence.bytes||0)/1048576).toFixed(1)} MiB. No upload.`,
      interrupted:'Clip saving was interrupted. Retry while the source is available.',removed:'Retained clip removed. Any copied labeling task remains separate.',
      missing:'The retained clip is missing. Keep it again before labeling.',changed:'The retained clip changed. Keep it again before labeling.'};
    $('evidence-status').textContent=evidence.message||messages[evidence.status]||'Clip is unavailable.';
    controls();
  }
  function choose(report, force=false) {
    if(saving||loading)return;
    if(!force&&(dirty||taskDirty))return edited();
    selected=report;dirty=false;taskDirty=false;clearVideo();$('report-detail').hidden=false;
    $('report-title').textContent=report.context.title;
    $('report-moment').textContent=`${report.kind==='wrongly_censored'?'Wrongly censored':'Missed something'} at ${time(report.position)} · Review ${time(report.window[0])}–${time(report.window[1])}`;
    const context=report.context;
    $('report-context').textContent=`Reported ${new Date(report.created_at).toLocaleString()}. ${context.context_changed_since_playback?'Analysis changed before the report arrived; saved detections are from receipt. ':''}${!context.analyzed_at_receipt?'This moment had not finished analysis when the report arrived. ':''}The report window is approximate.`;
    $('report-availability').textContent=report.media_available?'The original session is available.':
      report.evidence?.status==='ready'?'The original session is unavailable. The retained clip can still be reviewed.':
      context.live?'This was a live stream. Retaining live buffers is not supported yet.':'Original video is unavailable or has changed. The report is still saved.';
    $('review-category').value=report.category;$('review-note').value=report.note;$('review-status').value=report.status;
    const task=report.evidence?.task;
    $('label-title').value=task?.title||context.title;
    $('label-start').value=String(task?.target[0]??0);
    $('label-end').value=String(task?.target[1]??report.evidence?.duration??(report.window[1]-report.window[0]));
    $('review-message').textContent='';$('label-task-status').textContent=task?.promoted?'Human labels were added to training. Models were not changed.':'';
    evidenceView();drawList();
  }
  function drawList() {
    $('report-list').replaceChildren();
    if(!rows.length){const p=document.createElement('p');p.textContent='No reports here yet.';$('report-list').append(p);}
    for(const row of rows) {
      const button=document.createElement('button');button.className='report-choice';button.disabled=saving||loading;
      button.setAttribute('aria-pressed',String(row.id===selected?.id));button.textContent=row.context.title;
      const small=document.createElement('small');small.textContent=`${time(row.position)} · ${row.kind==='wrongly_censored'?'Wrongly censored':'Missed something'} · ${row.status}`;
      button.append(small);button.onclick=()=>choose(row);$('report-list').append(button);
    }
  }
  async function reload(nextOffset=offset) {
    if(loading||saving)return;
    if(dirty||taskDirty){$('report-filter').value=filter;return edited();}
    loading=true;controls();drawList();
    try {
      const nextFilter=$('report-filter').value;
      const data=await request(`?status=${nextFilter}&limit=50&offset=${nextOffset}`,undefined,'GET');
      rows=data.reports;total=data.total;offset=nextOffset;filter=nextFilter;
      $('queue-status').textContent=total?`${offset+1}–${offset+rows.length} of ${total} reports`:'No reports in this view.';
      loading=false;
      if(selected){const updated=rows.find(row=>row.id===selected.id);if(updated)choose(updated);else{selected=null;clearVideo();$('report-detail').hidden=true;}}
    } catch(error) {$('report-filter').value=filter;$('queue-status').textContent=`Could not load reports. ${error.message}`;}
    finally {loading=false;controls();drawList();}
  }
  $('report-refresh').onclick=()=>reload();
  $('report-filter').onchange=()=>reload(0);
  $('report-prev').onclick=()=>reload(Math.max(0,offset-50));
  $('report-next').onclick=()=>reload(offset+50);
  for(const id of ['review-category','review-status','review-note']) {
    const changed=()=>{dirty=true;controls();};
    $(id).oninput=changed;$(id).onchange=changed;
  }
  for(const id of ['label-title','label-start','label-end']) {
    const changed=()=>{taskDirty=true;controls();};
    $(id).oninput=changed;$(id).onchange=changed;
  }
  $('review-discard').onclick=()=>{if(selected&&!saving&&!loading)choose(selected,true);};
  $('report-watch').onclick=()=>{
    if(!selected||saving||loading)return;
    const report=selected, ready=report.evidence?.status==='ready';
    if(!ready&&!report.media_available)return;
    clearVideo();clipPlayback=ready;video.src=ready?`/api/feedback/${report.id}/evidence`:`/api/media/${report.session_id}/source`;video.hidden=false;
    video.onloadedmetadata=()=>{if(selected?.id===report.id)video.currentTime=ready?0:report.window[0];};
    video.onerror=()=>{$('review-message').textContent='Video is no longer available. The report is still saved.';};
    video.load();controls();
  };
  video.ontimeupdate=()=>{
    const end=clipPlayback?Number($('label-end').value):selected?.window[1];
    if(selected&&video.currentTime>=end&&!video.paused)video.pause();
  };
  for(const [id,field] of [['label-mark-start','label-start'],['label-mark-end','label-end']])$(id).onclick=()=>{
    if(!clipPlayback||saving||selected?.evidence?.task)return;
    $(field).value=video.currentTime.toFixed(3);taskDirty=true;controls();
  };
  $('review-save').onclick=async()=>{
    if(!selected||saving||loading)return;
    saving=true;controls();drawList();
    try {
      selected=await request('/'+selected.id,{revision:selected.revision,category:$('review-category').value,
        note:$('review-note').value,status:$('review-status').value},'PATCH');
      rows=rows.map(row=>row.id===selected.id?selected:row);dirty=false;
      $('review-message').textContent='Review saved. This is still separate from training labels.';
    } catch(error) {$('review-message').textContent=`Not saved. ${error.message}`;}
    finally {saving=false;evidenceView();drawList();}
  };
  async function operation(path, body, method='POST') {
    if(!selected||saving||loading)return;
    if(dirty)return edited();
    saving=true;controls();drawList();$('label-task-status').textContent='';
    try {
      const result=await request('/'+selected.id+path,body,method);
      if(path==='/training') {
        selected.evidence.task.promoted=true;
        $('label-task-status').textContent='Added human labels to training. No model was trained or enabled.';
      } else {
        const prior=selected.evidence?.status;selected=result;rows=rows.map(row=>row.id===selected.id?selected:row);
        if(path==='/labeling') {
          taskDirty=false;$('label-task-status').textContent='Labeling task created with empty labels. Open the editor to review every category.';
        }
        if(prior!=='ready'&&selected.evidence?.status==='ready'&&!taskDirty)$('label-end').value=String(selected.evidence.duration);
        if(method==='DELETE')clearVideo();
      }
    } catch(error) {$('label-task-status').textContent=`Not saved. ${error.message}`;}
    finally {saving=false;evidenceView();drawList();}
  }
  $('report-keep').onclick=()=>operation('/evidence');
  $('report-remove-clip').onclick=()=>operation('/evidence',undefined,'DELETE');
  $('create-label-task').onclick=()=>operation('/labeling',{title:$('label-title').value,start:Number($('label-start').value),end:Number($('label-end').value)});
  $('promote-labels').onclick=()=>{if(taskDirty)return edited();return operation('/training');};
  async function poll() {
    if(!selected||polling||loading||saving||document.hidden)return;
    const state=selected.evidence||{status:'none'};
    if(!['queued','saving'].includes(state.status)&&!(state.task&&!state.task.promoted))return;
    const id=selected.id;polling=true;
    try {
      const latest=await request('/'+id,undefined,'GET');
      if(selected?.id!==id||saving||loading)return;
      const wasReady=selected.evidence?.status==='ready';
      selected={...selected,evidence:latest.evidence,media_available:latest.media_available};
      rows=rows.map(row=>row.id===id?{...row,evidence:latest.evidence}:row);
      if(!wasReady&&latest.evidence?.status==='ready'&&!taskDirty)$('label-end').value=String(latest.evidence.duration);
      evidenceView();
    } catch(error) {$('evidence-status').textContent=`Could not refresh clip status. ${error.message}`;}
    finally {polling=false;}
  }
  return {load:()=>reload(),poll};
}
