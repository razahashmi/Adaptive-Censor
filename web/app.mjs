import {categories, labels, defaultPolicy, validPolicy, compilePlan, actionAt, coverageEnd, audioSchedule, selectedEvents} from './policy.mjs';
import {feedbackApi, installReporter} from './feedback.mjs';

const $ = id => document.getElementById(id);
const video=$('source'), canvas=$('picture'), ctx=canvas.getContext('2d',{alpha:false});
const originalCanvas=$('original-picture'), originalCtx=originalCanvas.getContext('2d',{alpha:false});
let token='', job=null, plan=[], wanted=false, fullScan=false, busy=false, configuring=false;
let audio=null, sourceGain=null, beepGain=null, masterGain=null, lastFrameTime=0, pollBusy=false, playPending=false;
let mediaSource=null, sourceBuffer=null, liveFetchBusy=false, liveAppended=new Set(), mediaURL=null;
let drawingRegion=false, regionStart=null, markedRegion=null;
const regionCanvas=document.createElement('canvas'),regionCtx=regionCanvas.getContext('2d');
let policy=defaultPolicy();
try {const saved=localStorage.getItem('adaptive-policy-v2');if(saved)policy=validPolicy(JSON.parse(saved));} catch {}
const format = seconds => `${Math.floor(Math.max(0,seconds)/60)}:${Math.floor(Math.max(0,seconds)%60).toString().padStart(2,'0')}`;
const formatPrecise = seconds => {const ticks=Math.round(Math.max(0,seconds)*100);return `${Math.floor(ticks/6000)}:${((ticks%6000)/100).toFixed(2).padStart(5,'0')}`;};

function notice(message) {$('notice').textContent=message;$('notice').hidden=!message;}
async function api(path,body,method='POST') {
  const headers={'x-censor-token':token};
  if(body!==undefined && !(body instanceof FormData)) headers['Content-Type']='application/json';
  const response=await fetch('/api/'+path,{method,headers,body:body===undefined?undefined:body instanceof FormData?body:JSON.stringify(body)});
  const data=await response.json();
  if(!response.ok) throw Error(typeof data.detail==='string'?data.detail:JSON.stringify(data.detail||data));
  return data;
}
function safely(fn){return async(...args)=>{try{await fn(...args);}catch(error){notice(error.message);}};}
const reporter=installReporter({document, storage:localStorage,
  request:feedbackApi({getToken:()=>token,setToken:value=>{token=value;}}),
  capture:()=>{
    if(!job || !Number.isFinite(video.currentTime))return null;
    const action=actionAt(plan,video.currentTime);
    return {session_id:job.id,position:video.currentTime,generation:job.generation,snapshot_updated:job.updated,
      policy,viewer_action:{visual:action.visual,audio:action.audio,blur:action.blur,regional:!!action.regions?.length}};
  }});
function message(title,subtitle='') {
  $('player-message').replaceChildren();
  const strong=document.createElement('strong');strong.textContent=title;
  const p=document.createElement('p');p.textContent=subtitle;
  $('player-message').append(strong,p);$('player-message').hidden=false;
}

function renderRules() {
  $('rules').replaceChildren();
  for(const category of categories) {
    const rule=policy.categories[category], block=document.createElement('section');block.className='rule';
    const heading=document.createElement('div');heading.className='rule-heading';
    const name=document.createElement('strong');name.textContent=labels[category];
    const mode=document.createElement('small');mode.textContent=category==='language'?'WORD MATCH':category==='nudity'?'ANATOMY':'EXPERIMENTAL';
    heading.append(name,mode);block.append(heading);
    const actions=document.createElement('div');actions.className='rule-actions';
    for(const [field,options] of [['visual',[['allow','Keep picture'],['blur','Blur'],['hide','Conceal completely']]],['audio',[['allow','Keep sound'],['beep','Beep'],['mute','Mute']]]]) {
      const select=document.createElement('select');select.setAttribute('aria-label',`${labels[category]} ${field==='visual'?'picture':'sound'}`);
      for(const [value,label] of options){const option=document.createElement('option');option.value=value;option.textContent=label;select.append(option);}
      select.value=rule[field];select.onchange=()=>{rule[field]=select.value;policyChanged();};actions.append(select);
    }
    block.append(actions);
    if(category==='nudity') {
      const area=document.createElement('select');area.setAttribute('aria-label','Nudity filter area');
      for(const [value,text] of [['region','Detected body regions'],['frame','Whole picture']]){const option=document.createElement('option');option.value=value;option.textContent=text;area.append(option);}
      area.value=rule.scope||'frame';area.onchange=()=>{rule.scope=area.value;policyChanged();};block.append(area);
      const hint=document.createElement('p');hint.className='hint';hint.textContent='Sampled masks can lag movement. Whole-picture anatomy claims need review when no matching body region is found.';block.append(hint);
    }
    for(const [field,label,min,max,step] of [['blur','Blur strength',1,100,1],['min_confidence','Min. score',0,1,.01]]) {
      const row=document.createElement('label');row.className='range-row';row.textContent=label;
      const input=document.createElement('input');Object.assign(input,{type:'range',min,max,step,value:rule[field]});input.setAttribute('aria-label',`${labels[category]} ${label.toLowerCase()}`);
      const output=document.createElement('output');output.textContent=field==='blur'?rule[field]:rule[field].toFixed(2);
      input.oninput=()=>{rule[field]=+input.value;output.textContent=field==='blur'?rule[field]:rule[field].toFixed(2);policyChanged();};
      row.append(input,output);block.append(row);
    }
    $('rules').append(block);
  }
  $('pre-padding').value=policy.pre_padding;$('post-padding').value=policy.post_padding;
}

function policyChanged() {
  silence();
  try {localStorage.setItem('adaptive-policy-v2',JSON.stringify(policy));} catch {}
  plan=job?compilePlan(job.events,policy,job.duration):[];
  document.querySelectorAll('[data-preset]').forEach(b=>b.classList.remove('selected'));
  draw(lastFrameTime);scheduleAudio();renderTimeline();renderEvents();
}

function config() {return {speech:$('speech').checked,nudity:$('nudity').checked,scenes:$('scenes').checked,
  speech_model:$('speech-model').value,
  scene_mode:$('scene-mode').value,anatomy_model:$('anatomy-model').value,
  scene_filtering:$('scene-filtering').value,
  scene_model:$('scene-model').value,learned_heads:$('learned-heads').value,
  learned_verification:$('learned-verification').value,
  anatomy_fps:job?.config?.anatomy_fps??12,anatomy_tracking:job?.config?.anatomy_tracking??true,
  anatomy_roi:$('anatomy-roi').value,
  phrases:$('phrases').value.split(/\r?\n/).filter(Boolean),allow_phrases:$('allow-phrases').value.split(/\r?\n/).filter(Boolean),
  custom_visual:$('custom-visual').value,fps:+$('fps').value,lookahead:+$('lookahead').value,device:$('device').value};}

function syncConfig(c) {
  for(const name of ['speech','nudity','scenes']) $(name).checked=c[name];
  $('phrases').value=c.phrases.join('\n');$('allow-phrases').value=c.allow_phrases.join('\n');
  $('custom-visual').value=c.custom_visual;
  for(const name of ['fps','lookahead','device']) $(name).value=c[name];
  $('speech-model').value=c.speech_model || 'tiny.en';
  $('scene-mode').value=c.scene_mode || 'verified';
  $('anatomy-model').value=c.anatomy_model || '320n';
  $('scene-filtering').value=c.scene_filtering || 'review';
  $('scene-model').value=c.scene_model || '4B';
  $('learned-verification').value=c.learned_verification || 'off';
  $('anatomy-roi').value=[...$('anatomy-roi').options].some(o=>o.value===c.anatomy_roi)?c.anatomy_roi:'';
  $('learned-heads').value=[...$('learned-heads').options].some(o=>o.value===c.learned_heads)?c.learned_heads:'';
}

function silence() {
  if(!audio)return;
  for(const gain of [sourceGain,beepGain]) {gain.gain.cancelScheduledValues(audio.currentTime);gain.gain.setValueAtTime(0,audio.currentTime);}
}

async function initAudio() {
  if(!audio) {
    audio=new AudioContext();
    const source=audio.createMediaElementSource(video);
    sourceGain=audio.createGain();beepGain=audio.createGain();masterGain=audio.createGain();
    sourceGain.gain.value=0;beepGain.gain.value=0;masterGain.gain.value=+$('volume').value;
    source.connect(sourceGain).connect(masterGain).connect(audio.destination);
    const tone=audio.createOscillator();tone.frequency.value=880;tone.connect(beepGain).connect(masterGain);tone.start();
    video.muted=false;
  }
  await audio.resume();
}

function scheduleAudio() {
  if(!audio)return;
  if(!job || configuring || video.paused || video.seeking || video.readyState<3 || document.hidden || !wanted || job.error) {silence();return;}
  const now=audio.currentTime, t=video.currentTime;
  const localPlan=plan.filter(e=>e.end+.1>t && e.start-.1<t+1);
  const decisions=audioSchedule(localPlan,job.coverage,t,job.duration);
  for(const gain of [sourceGain,beepGain]) gain.gain.cancelScheduledValues(now);
  for(const decision of decisions) {
    const at=now+Math.max(0,decision.time-t);
    sourceGain.gain.setValueAtTime(decision.source,at);beepGain.gain.setValueAtTime(decision.beep,at);
  }
}

function draw(t) {
  const analyzed=job && !configuring && !job.error && coverageEnd(job.coverage,t)>t;
  const action=actionAt(plan,t);
  ctx.filter='none';ctx.fillStyle='#070b11';ctx.fillRect(0,0,canvas.width,canvas.height);
  // Both canvases read the same decoded video frame. No second player or clock.
  originalCtx.fillStyle='#070b11';originalCtx.fillRect(0,0,originalCanvas.width,originalCanvas.height);
  if(!analyzed || video.readyState<2 || video.seeking) {$('effect-badge').hidden=true;return;}
  if($('compare-original').checked) originalCtx.drawImage(video,0,0,originalCanvas.width,originalCanvas.height);
  if(action.visual!=='hide') {
    ctx.save();
    if(action.visual==='blur') ctx.filter=`blur(${(2+action.blur*.78)*canvas.width/(video.videoWidth||canvas.width)}px)`;
    ctx.drawImage(video,0,0,canvas.width,canvas.height);ctx.restore();
    for(const region of [...(action.regions||[])].sort((a,b)=>(a.visual==='hide')-(b.visual==='hide'))) {
      const [nx,ny,nw,nh]=region.box,x=nx*canvas.width,y=ny*canvas.height,w=nw*canvas.width,h=nh*canvas.height;
      if(region.visual==='hide'){ctx.fillStyle='#000';ctx.fillRect(x,y,w,h);}
      else if(region.visual==='blur') {
        regionCanvas.width=Math.max(2,Math.ceil(w));regionCanvas.height=Math.max(2,Math.ceil(h));
        regionCtx.fillStyle='#000';regionCtx.fillRect(0,0,w,h);
        regionCtx.filter=`blur(${(2+Math.max(region.blur,action.visual==='blur'?action.blur:0)*.78)*canvas.width/(video.videoWidth||canvas.width)}px)`;
        regionCtx.drawImage(video,nx*video.videoWidth,ny*video.videoHeight,nw*video.videoWidth,nh*video.videoHeight,0,0,w,h);
        ctx.drawImage(regionCanvas,x,y,w,h);
      }
    }
  }
  if(markedRegion && drawingRegion){const [x,y,w,h]=markedRegion;ctx.strokeStyle='#bce899';ctx.lineWidth=3;ctx.strokeRect(x*canvas.width,y*canvas.height,w*canvas.width,h*canvas.height);}
  $('effect-badge').hidden=action.visual==='allow' && action.audio==='allow' && !action.regions?.length;
  $('effect-badge').textContent=[action.visual==='hide'?'Picture concealed':action.visual==='blur'?`Blur ${action.blur}`:action.regions?.length?'Region filtered':'',action.audio!=='allow'?action.audio:''].filter(Boolean).join(' · ');
  if(!wanted || !video.paused) $('player-message').hidden=true;
}

function frame(_now,metadata) {
  lastFrameTime=metadata.mediaTime;
  draw(lastFrameTime);
  video.requestVideoFrameCallback(frame);
}
if('requestVideoFrameCallback' in video) video.requestVideoFrameCallback(frame);
else {notice('This browser needs requestVideoFrameCallback support. Use a recent Chrome or Edge release.');$('play').disabled=true;}

function readyToStart() {
  if(!job || configuring || job.error || document.hidden || video.seeking || video.readyState<2)return false;
  const t=video.currentTime, end=coverageEnd(job.coverage,t);
  // Resume with a useful reserve; full lookahead remains the analysis target.
  const reserve=job.live && job.capture_running?job.config.lookahead:Math.min(6,job.config.lookahead);
  return end>=Math.min(job.live && job.capture_running?Infinity:job.duration-.025,t+reserve);
}

async function maintainPlayback() {
  if(!job)return;
  const t=video.currentTime,end=coverageEnd(job.coverage,t),lead=Math.max(0,end-t);
  $('lead').textContent=`${lead.toFixed(1)}s analyzed ahead`;
  $('clock').textContent=`${format(t)} / ${job.live?'LIVE':format(job.duration)}`;
  if(document.activeElement!==$('seek'))$('seek').value=t;
  if(!video.paused && (job.error || configuring || ((job.live && job.capture_running || end<job.duration-.01) && lead<.2))) {
    silence();video.pause();draw(t);message('Waiting for analysis','Playback resumes when the next interval is ready.');
  }
  if(wanted && video.paused && !playPending && readyToStart() && !video.ended) {
    playPending=true;
    try {await initAudio();await video.play();scheduleAudio();$('player-message').hidden=true;}
    catch(error){wanted=false;notice(`Playback could not start: ${error.message}`);}
    finally{playPending=false;}
  }
  if(wanted && video.paused && !job.error) message('Analyzing ahead',`Ready: ${lead.toFixed(1)} seconds. The picture and sound stay closed while waiting.`);
  $('play').textContent=wanted?'Pause':'Play';
  $('playback-status').textContent=job.error?'Analysis stopped':!video.paused?'Playing with your rules':wanted?'Waiting for analysis':job.status==='analyzing'?'Analyzing ahead':'Paused · filters ready';
}

function renderTimeline() {
  const c=$('timeline'), context=c.getContext('2d');context.fillStyle='#28303c';context.fillRect(0,0,c.width,c.height);
  if(!job)return;
  const scale=c.width/Math.max(1,job.duration);
  for(const [a,b] of job.coverage){context.fillStyle='#618458';context.fillRect(a*scale,0,(b-a)*scale,32);}
  for(const e of plan){context.fillStyle='#d8a967';context.fillRect(e.start*scale,20,Math.max(1,(e.end-e.start)*scale),12);}
  context.fillStyle='#f7fbf2';context.fillRect(video.currentTime*scale,0,2,32);
}

function renderEvents() {
  const list=$('event-list');list.replaceChildren();if(!job)return;
  const t=video.currentTime, selected=job.events.filter(e=>e.confidence>=policy.categories[e.category]?.min_confidence);
  const nearby=selected.filter(e=>e.end>=Math.max(0,t-5) && e.start<t+20).slice(0,16);
  if(!nearby.length){const p=document.createElement('p');p.className='empty';p.textContent='No candidates above your thresholds near this position.';list.append(p);return;}
  for(const event of nearby) {
    const row=document.createElement('div');row.className='event-row'+(event.enabled===false?' disabled':'');
    const time=document.createElement('button');time.className='event-time';time.textContent=`${formatPrecise(event.start)}–${formatPrecise(event.end)}`;time.onclick=()=>seekTo(event.start);
    const text=document.createElement('div');text.className='event-text';text.textContent=labels[event.category];
    const needsReview=event.review_required || event.source?.startsWith('CLIP');
    const small=document.createElement('small');small.textContent=`${event.source} · ${needsReview?(event.confirmed?'Confirmed':'Needs review'):event.score_type==='decision'?'Model decision':`score ${event.confidence.toFixed(2)}`}`;text.append(small);
    if(event.note){const evidence=document.createElement('small');evidence.textContent=event.note;text.append(evidence);}
    if(needsReview){
      const confirm=document.createElement('button');confirm.textContent=event.confirmed?'Undo confirmation':'Confirm filter';
      confirm.onclick=safely(async()=>{silence();applySnapshot(await api(`media/${job.id}/review`,{id:event.id,enabled:true,confirmed:!event.confirmed}));});
      text.append(confirm);
    }
    const toggle=document.createElement('button');toggle.textContent=event.enabled===false?'Restore':'Exclude';
    toggle.onclick=safely(async()=>{silence();applySnapshot(await api(`media/${job.id}/review`,{id:event.id,enabled:event.enabled===false}));});
    row.append(time,text,toggle);list.append(row);
  }
}

function applySnapshot(data) {
  job=data;plan=compilePlan(job.events,policy,job.duration);
  reporter.refresh();
  $('speed').textContent=job.speed?`${job.speed.toFixed(1)}× real time`:'—';
  $('coverage').textContent=job.live?`${job.analyzed_seconds.toFixed(0)}s retained`:`${Math.round(job.analyzed_seconds/Math.max(.01,job.duration)*100)}%`;
  $('matches').textContent=plan.length;
  const backends=Object.values(job.backends);$('device-status').textContent=!backends.length?'Waiting':backends.every(s=>/CUDA/.test(s))?'GPU':backends.some(s=>/CUDA/.test(s))?'GPU + CPU':'CPU';
  $('device-status').title=JSON.stringify(job.backends);
  $('scan-detail').textContent=job.p95_chunk_seconds?`6s chunks · p95 ${job.p95_chunk_seconds.toFixed(2)}s · ${job.status}`:job.status;
  $('export').disabled=job.live || job.analyzed_seconds<job.duration-.01;
  $('seek').max=Math.max(1,job.duration);$('seek').min=job.retained_start||0;
  $('export-status').textContent=$('export').disabled?'Analyze the full file to export.':'Ready to render your current policy.';
  if(job.error){silence();video.pause();wanted=false;message('Analysis stopped','Adjust processing settings and retry.');notice(job.error);}
  renderEvents();renderTimeline();draw(lastFrameTime);scheduleAudio();
}

async function loadJob(data,analyze=true) {
  if(job?.live && job.capture_running && job.id!==data.id)await api(`media/${job.id}/stop-live`);
  if(job)await api(`media/${job.id}/cursor`,{position:video.currentTime,active:false,full_scan:false});
  wanted=false;fullScan=false;silence();video.pause();lastFrameTime=0;
  markedRegion=null;regionStart=null;drawingRegion=false;canvas.style.cursor='';canvas.style.touchAction='';$('region-status').textContent='Whole picture';
  if(mediaURL)URL.revokeObjectURL(mediaURL);mediaSource=null;sourceBuffer=null;mediaURL=null;liveAppended=new Set();
  job=data;
  if(job.live)await initLive();else {video.src=`/api/media/${job.id}/source`;video.load();}
  $('media-title').textContent=job.title;$('seek').max=job.duration;$('seek').value=0;
  for(const name of ['play','back','seek','scan','full-scan','stop-scan','delete'])$(name).disabled=false;
  $('full-scan').disabled=job.live;$('stop-live').hidden=!job.live;
  $('download-timeline').href=`/api/media/${job.id}/timeline`;$('download-timeline').hidden=false;
  message('Preparing your media','Analysis will stay ahead of playback.');
  applySnapshot(data);
  if(analyze)await applyConfig();else syncConfig(data.config);
}

async function initLive() {
  if(!window.MediaSource)throw Error('This browser does not support segmented live playback.');
  mediaSource=new MediaSource();mediaURL=URL.createObjectURL(mediaSource);video.src=mediaURL;video.load();
  const current=mediaSource;
  await new Promise((resolve,reject)=>{current.addEventListener('sourceopen',resolve,{once:true});current.addEventListener('error',reject,{once:true});});
  const mime='video/mp4; codecs="avc1.4d401f, mp4a.40.2"';
  if(!MediaSource.isTypeSupported(mime))throw Error('This browser cannot decode the live H.264/AAC stream.');
  sourceBuffer=current.addSourceBuffer(mime);
}

function appendFragment(buffer,data) {
  return new Promise((resolve,reject)=>{
    const done=()=>{cleanup();resolve();},fail=()=>{cleanup();reject(Error('The browser could not decode a live fragment.'));};
    const cleanup=()=>{buffer.removeEventListener('updateend',done);buffer.removeEventListener('error',fail);};
    buffer.addEventListener('updateend',done,{once:true});buffer.addEventListener('error',fail,{once:true});
    try{buffer.appendBuffer(data);}catch(error){cleanup();reject(error);}
  });
}

async function updateLive() {
  if(!job?.live || liveFetchBusy || !sourceBuffer || sourceBuffer.updating || mediaSource.readyState!=='open')return;
  liveFetchBusy=true;const id=job.id,buffer=sourceBuffer;
  try{
    const manifest=await api(`media/${id}/manifest`,undefined,'GET');
    if(job?.id!==id)return;
    for(const index of liveAppended)if(typeof index==='number' && index*6<manifest.retained_start)liveAppended.delete(index);
    if(!liveAppended.size && manifest.segments.length){
      const response=await fetch(`/api/media/${id}/fragment/init`);if(!response.ok)throw Error('Live initialization fragment unavailable.');
      await appendFragment(buffer,await response.arrayBuffer());liveAppended.add('init');
    }
    for(const segment of manifest.segments){
      if(liveAppended.has(segment.index) || segment.start>video.currentTime+job.config.lookahead+12)continue;
      const response=await fetch(`/api/media/${id}/fragment/${segment.index}`);if(!response.ok)throw Error('Live fragment expired before playback.');
      await appendFragment(buffer,await response.arrayBuffer());liveAppended.add(segment.index);
      if(job?.id!==id)return;
    }
    if(video.currentTime>36 && !buffer.updating){
      await new Promise((resolve,reject)=>{buffer.addEventListener('updateend',resolve,{once:true});try{buffer.remove(0,video.currentTime-30);}catch(e){reject(e);}});
    }
    if(!manifest.running && manifest.segments.length && manifest.segments.every(s=>liveAppended.has(s.index)) && !buffer.updating)mediaSource.endOfStream();
  }finally{liveFetchBusy=false;}
}

async function applyConfig() {
  if(!job)return;silence();video.pause();configuring=true;draw(video.currentTime);notice('');
  try{const data=await api(`media/${job.id}/configure`,config());applySnapshot(data);}
  finally{configuring=false;draw(video.currentTime);}
}

async function seekTo(time) {
  if(!job)return;silence();video.pause();video.currentTime=Math.max(job.retained_start||0,Math.min(time,job.duration-.05));
  lastFrameTime=video.currentTime;draw(lastFrameTime);renderTimeline();
  await api(`media/${job.id}/cursor`,{position:video.currentTime,active:true,full_scan:fullScan});
}

$('demo').onclick=safely(async()=>{notice('');await loadJob(await api('demo'),false);});
$('open-file').onclick=()=>$('upload').click();
$('recent-media').onchange=safely(async()=>{const id=$('recent-media').value;if(id)await loadJob(await api(`media/${id}`,undefined,'GET'),false);});
async function openLive(url){
  if(job?.live && job.capture_running)await api(`media/${job.id}/stop-live`);
  notice('');await loadJob(await api('live',{url,config:config()}),false);
  message('Filling the live buffer',`Waiting for ${job.config.lookahead} seconds of analyzed media. Press Play to begin when ready.`);
}
$('start-live').onclick=safely(()=>openLive($('live-url').value));
$('live-demo').onclick=safely(()=>openLive('demo'));
$('stop-live').onclick=safely(async()=>{if(job?.live)applySnapshot(await api(`media/${job.id}/stop-live`));$('stop-live').hidden=true;});
$('upload').onchange=safely(async()=>{
  const file=$('upload').files[0];if(!file || busy)return;busy=true;$('upload-status').textContent='Copying and preparing the file…';notice('');
  try{const form=new FormData();form.append('file',file);await loadJob(await api('upload',form));$('upload-status').textContent='Ready. This copy stays on your computer.';}
  finally{busy=false;$('upload').value='';}
});
$('scan').onclick=safely(applyConfig);
$('full-scan').onclick=safely(async()=>{fullScan=true;await api(`media/${job.id}/cursor`,{position:video.currentTime,active:true,full_scan:true});notice('Full-file analysis is running. You can watch while it continues.');});
$('stop-scan').onclick=safely(async()=>{wanted=false;fullScan=false;silence();video.pause();await api(`media/${job.id}/cursor`,{position:video.currentTime,active:false,full_scan:false});});
$('play').onclick=safely(async()=>{if(!job)return;notice('');wanted=!wanted;if(!wanted){silence();video.pause();}else{await initAudio();if(video.ended)await seekTo(0);await api(`media/${job.id}/cursor`,{position:video.currentTime,active:true,full_scan:fullScan});}await maintainPlayback();});
$('seek').oninput=safely(()=>seekTo(+$('seek').value));
$('back').onclick=safely(()=>seekTo(video.currentTime-10));
$('volume').oninput=()=>{if(masterGain)masterGain.gain.value=+$('volume').value;};
$('fullscreen').onclick=safely(()=>$('viewing-area').requestFullscreen());
$('compare-original').onchange=()=>{
  const enabled=$('compare-original').checked;
  $('player').classList.toggle('comparing',enabled);$('original-pane').hidden=!enabled;$('comparison-hint').hidden=!enabled;
  draw(lastFrameTime);
};
video.addEventListener('loadedmetadata',()=>{const scale=Math.min(1,1280/(video.videoWidth||960));canvas.width=originalCanvas.width=Math.round((video.videoWidth||960)*scale);canvas.height=originalCanvas.height=Math.round((video.videoHeight||540)*scale);});
video.addEventListener('loadeddata',()=>{lastFrameTime=video.currentTime;draw(lastFrameTime);});
video.addEventListener('seeking',()=>{silence();draw(video.currentTime);});
video.addEventListener('seeked',()=>{lastFrameTime=video.currentTime;draw(lastFrameTime);renderEvents();});
video.addEventListener('waiting',silence);
video.addEventListener('pause',silence);
video.addEventListener('ended',()=>{wanted=false;silence();$('play').textContent='Play';});
video.addEventListener('error',()=>{wanted=false;silence();notice('The browser could not play this media. Try opening it again.');});
document.addEventListener('visibilitychange',()=>{if(document.hidden){wanted=false;silence();video.pause();}});
window.addEventListener('pagehide',()=>{silence();video.pause();});

document.querySelectorAll('[data-preset]').forEach(button=>button.onclick=()=>{
  policy=defaultPolicy();
  for(const category of categories){const r=policy.categories[category];
    if(button.dataset.preset==='gentle'){if(category!=='language')r.visual=category==='kissing'?'allow':'blur';r.blur=20;r.min_confidence=category==='language'?.25:category==='nudity'?.65:.80;}
    if(button.dataset.preset==='strict'){if(category!=='language')r.visual='hide';r.min_confidence=category==='language'?0:.35;r.audio=category==='language'?'beep':'mute';}
  }
  renderRules();policyChanged();button.classList.add('selected');
});
for(const id of ['pre-padding','post-padding'])$(id).onchange=safely(()=>{const n=+$(id).value;if(!Number.isFinite(n)||n<0||n>1)throw Error('Padding must be 0–1 seconds.');policy[id.replace('-','_')]=n;policyChanged();});

function saveJSON(data,filename){const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=filename;a.click();setTimeout(()=>URL.revokeObjectURL(url),2000);}
$('save-profile').onclick=()=>saveJSON({version:1,policy,analysis:config()},'adaptive-censor-profile.json');
$('load-profile').onchange=safely(async()=>{const file=$('load-profile').files[0];if(!file)return;if(file.size>100000)throw Error('Profile is too large.');const data=JSON.parse(await file.text());policy=validPolicy(data.policy);if(data.analysis)syncConfig(data.analysis);renderRules();policyChanged();notice('Policy loaded. Apply & analyze ahead to apply any changed detector settings.');});
for(const c of categories){const option=document.createElement('option');option.value=c;option.textContent=labels[c];$('manual-category').append(option);}
$('draw-region').onclick=safely(()=>{if(!job)throw Error('Open a file first.');wanted=false;silence();video.pause();drawingRegion=true;markedRegion=null;canvas.style.cursor='crosshair';canvas.style.touchAction='none';$('manual-start').value=video.currentTime.toFixed(2);$('manual-end').value=Math.min(job.duration,video.currentTime+2).toFixed(2);$('region-status').textContent='Drag a rectangle on the paused picture, then add the interval. The rectangle stays fixed during this interval.';canvas.scrollIntoView({block:'center'});});
$('clear-region').onclick=()=>{markedRegion=null;regionStart=null;drawingRegion=false;canvas.style.cursor='';canvas.style.touchAction='';$('region-status').textContent='Whole picture';draw(lastFrameTime);};
function picturePoint(event){
  const box=canvas.getBoundingClientRect(),scale=Math.min(box.width/canvas.width,box.height/canvas.height);
  const width=canvas.width*scale,height=canvas.height*scale;
  return [Math.max(0,Math.min(1,(event.clientX-box.left-(box.width-width)/2)/width)),Math.max(0,Math.min(1,(event.clientY-box.top-(box.height-height)/2)/height))];
}
canvas.addEventListener('pointerdown',event=>{if(!drawingRegion)return;event.preventDefault();regionStart=picturePoint(event);canvas.setPointerCapture(event.pointerId);});
canvas.addEventListener('pointermove',event=>{if(!drawingRegion || !regionStart)return;const [x,y]=picturePoint(event),[sx,sy]=regionStart;markedRegion=[Math.min(x,sx),Math.min(y,sy),Math.abs(x-sx),Math.abs(y-sy)];draw(lastFrameTime);});
canvas.addEventListener('pointerup',event=>{if(!regionStart)return;regionStart=null;if(canvas.hasPointerCapture(event.pointerId))canvas.releasePointerCapture(event.pointerId);if(!markedRegion || markedRegion[2]<.01 || markedRegion[3]<.01){markedRegion=null;$('region-status').textContent='Draw a larger rectangle.';}else $('region-status').textContent='Region selected. Set the end time and category, then Add interval. This fixed rectangle does not track movement.';draw(lastFrameTime);});
canvas.addEventListener('pointercancel',()=>{regionStart=null;});
$('manual-form').onsubmit=safely(async event=>{event.preventDefault();if(!job)throw Error('Open a file first.');if(drawingRegion && !markedRegion)throw Error('Draw a region or choose Whole picture.');silence();applySnapshot(await api(`media/${job.id}/events`,{start:+$('manual-start').value,end:+$('manual-end').value,category:$('manual-category').value,region:markedRegion}));drawingRegion=false;canvas.style.cursor='';canvas.style.touchAction='';markedRegion=null;$('region-status').textContent='Interval added';draw(lastFrameTime);});
$('export').onclick=safely(async()=>{
  const exported=await api(`media/${job.id}/export`,policy);$('export').disabled=true;$('export-status').textContent='Rendering filtered MP4…';
  const timer=setInterval(safely(async()=>{const state=await api(`exports/${exported.id}`,undefined,'GET');if(state.status==='rendering')return;clearInterval(timer);$('export').disabled=false;
    if(state.status==='error')throw Error(state.error);$('export-status').textContent='Export complete.';$('export-download').href=`/api/exports/${exported.id}/download`;$('export-download').hidden=false;
  }),1500);
});
$('delete').onclick=safely(async()=>{
  if(!job)return;wanted=false;fullScan=false;silence();video.pause();const id=job.id;
  await api(`media/${id}/cursor`,{position:video.currentTime,active:false,full_scan:false});
  video.removeAttribute('src');video.load();
  await api(`media/${id}`,undefined,'DELETE');job=null;plan=[];draw(0);
  reporter.refresh();
  message('Local copy removed','Open another video or podcast.');$('media-title').textContent='Nothing loaded';
  for(const name of ['play','back','seek','scan','full-scan','stop-scan','delete','export'])$(name).disabled=true;
  notice('The app’s copy was removed. Your original file is untouched.');
});

setInterval(()=>{maintainPlayback().catch(e=>notice(e.message));scheduleAudio();},100);
setInterval(async()=>{
  if(!job || pollBusy)return;pollBusy=true;const id=job.id;
  try {
    const result=await api(`media/${id}?since=${job.updated}`,undefined,'GET');
    if(!job || job.id!==id)return;
    if(!result.unchanged)applySnapshot(result);else {job.status=result.status;job.error=result.error;}
    if(job.live)await updateLive();
    if(wanted || fullScan)await api(`media/${id}/cursor`,{position:video.currentTime,active:true,full_scan:fullScan});
    renderTimeline();
  }catch(error){if(job?.id===id){silence();video.pause();wanted=false;notice(`Connection to the local engine stopped: ${error.message}`);}}
  finally{pollBusy=false;}
},750);

renderRules();
try {const data=await api('bootstrap',undefined,'GET');token=data.token;
  for(const session of data.sessions){const option=document.createElement('option');option.value=session.id;option.textContent=session.title;$('recent-media').append(option);}
  $('recent-row').hidden=!data.sessions.length;
  if(!data.anatomy_640_ready)$('anatomy-model').querySelector('[value="640m"]').disabled=true;
  for(const name of data.anatomy_roi_models||[]){const option=document.createElement('option');option.value=name;option.textContent=`Experimental · ${name}`;option.dataset.anatomyModel=data.anatomy_roi_requirements?.[name]||'';$('anatomy-roi').append(option);}
  $('anatomy-roi').disabled=!(data.anatomy_roi_models||[]).length;
  $('anatomy-roi').addEventListener('change',()=>{const required=$('anatomy-roi').selectedOptions[0]?.dataset.anatomyModel;if(required)$('anatomy-model').value=required;});
  $('anatomy-model').addEventListener('change',()=>{const required=$('anatomy-roi').selectedOptions[0]?.dataset.anatomyModel;if(required&&required!==$('anatomy-model').value)$('anatomy-roi').value='';});
  for(const name of data.learned_heads||[]){const option=document.createElement('option');option.value=name;option.textContent=`Review · ${name}`;$('learned-heads').append(option);}
  $('learned-heads').disabled=!(data.learned_heads||[]).length;
  $('scene-mode').querySelector('[value="learned"]').disabled=!(data.learned_heads||[]).length;
  if(!data.verifier_8b_ready)$('scene-model').querySelector('[value="8B"]').disabled=true;
  if(!data.verifier_4b_ready){$('scene-model').querySelector('[value="4B"]').disabled=true;if(data.verifier_8b_ready)$('scene-model').value='8B';}
  if(!data.verifier_ready){$('scenes').checked=false;notice('The sequence model is not installed. Run Setup.ps1 -SceneVerifier to enable it. CLIP suggestions require confirmation in the review panel.');}}
catch(error){notice(`Cannot reach the local engine: ${error.message}`);}
