import {categories, labels} from './policy.mjs';

// A server restart changes the local token. Retrying a report uses the same ID.
export function feedbackApi({getToken, setToken, fetcher=fetch}) {
  return async function request(path='', body, method='POST') {
    for(let attempt=0;attempt<2;attempt++) {
      const response=await fetcher('/api/feedback'+path, {method,
        headers:{'x-censor-token':getToken(),'Content-Type':'application/json'},
        body:body===undefined?undefined:JSON.stringify(body)});
      if(response.status===403 && attempt===0) {
        const bootstrap=await fetcher('/api/bootstrap');
        if(!bootstrap.ok)throw Error('Cannot reconnect to the local service.');
        setToken((await bootstrap.json()).token);continue;
      }
      const data=await response.json();
      if(!response.ok)throw Error(typeof data.detail==='string'?data.detail:'The report could not be saved.');
      return data;
    }
  };
}

export function installReporter({document, request, capture, storage, makeId=()=>crypto.randomUUID()}) {
  const $=id=>document.getElementById(id), key='adaptive-feedback-pending-v1';
  let pending=null, report=null, busy=false;
  const time=t=>`${Math.floor(t/60)}:${Math.floor(t%60).toString().padStart(2,'0')}`;
  try {pending=JSON.parse(storage?.getItem(key)||'null');} catch {}
  for(const category of categories) {
    const option=document.createElement('option');option.value=category;option.textContent=labels[category];
    $('feedback-category').append(option);
  }
  function remember() {try {if(pending)storage?.setItem(key,JSON.stringify(pending));else storage?.removeItem(key);} catch {}}
  function refresh() {
    const disabled=!capture() || busy || !!pending;
    $('report-wrong').disabled=$('report-missed').disabled=disabled;
    $('feedback-retry').hidden=!pending;$('feedback-retry').disabled=busy;
    $('feedback-undo').hidden=!report || report.status==='withdrawn';$('feedback-undo').disabled=busy || !!pending;
    $('feedback-details').hidden=!report || report.status==='withdrawn' || !!pending;
    $('feedback-save-details').disabled=busy;
  }
  function showReport() {
    $('feedback-category').value=report.category;$('feedback-note').value=report.note;
    $('feedback-details').open=false;
    $('feedback-status').textContent=report.status==='withdrawn'?'Report undone.':
      `Saved: ${report.kind==='wrongly_censored'?'wrongly censored':'missed something'} at ${time(report.position)}. Thank you.`;
    if(report.status!=='withdrawn' && ['queued','saving'].includes(report.evidence?.status))$('feedback-status').textContent+=' Local clip is saving in the background.';
    if(report.status!=='withdrawn' && report.evidence?.status==='error')$('feedback-status').textContent+=' The clip was not saved; retry from Saved reports.';
  }
  async function save() {
    if(!pending || busy)return;
    busy=true;refresh();$('feedback-status').textContent=`Saving report at ${time(pending.position)}…`;
    try {
      report=await request('',pending);pending=null;remember();showReport();
    } catch(error) {
      $('feedback-status').textContent=`Not saved at ${time(pending.position)}. Retry when connected. ${error.message}`;
    } finally {busy=false;refresh();}
  }
  async function begin(kind) {
    if(busy || pending)return;
    const context=capture();if(!context)return;
    pending=structuredClone({...context,id:makeId(),kind,retain_clip:$('feedback-keep-clip').checked===true});remember();await save();
  }
  $('report-wrong').onclick=()=>begin('wrongly_censored');
  $('report-missed').onclick=()=>begin('missed_content');
  $('feedback-retry').onclick=save;
  async function edit(changes) {
    if(!report || busy || pending)return;
    busy=true;refresh();
    try {report=await request('/'+report.id,{revision:report.revision,...changes},'PATCH');showReport();}
    catch(error) {$('feedback-status').textContent=`Change not saved. ${error.message}`;}
    finally {busy=false;refresh();}
  }
  $('feedback-undo').onclick=()=>edit({status:'withdrawn'});
  $('feedback-save-details').onclick=()=>edit({category:$('feedback-category').value,note:$('feedback-note').value});
  if(pending)$('feedback-status').textContent=`An earlier report at ${time(pending.position)} still needs saving. Press Retry.`;
  refresh();return {refresh};
}
