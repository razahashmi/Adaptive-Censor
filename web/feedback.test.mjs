import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {feedbackApi, installReporter} from './feedback.mjs';
import {installQueue} from './feedback-queue.mjs';
import {defaultPolicy} from './policy.mjs';

function dom(file) {
  const elements=new Map();
  function element() {return {children:[],hidden:false,disabled:false,value:'',textContent:'',paused:true,
    append(...nodes){this.children.push(...nodes);},replaceChildren(...nodes){this.children=[...nodes];},
    setAttribute(k,v){this[k]=v;},removeAttribute(k){delete this[k];},
    pause(){this.paused=true;this.pauses=(this.pauses||0)+1;},load(){this.loads=(this.loads||0)+1;}};}
  for(const [,id] of readFileSync(new URL(file,import.meta.url),'utf8').matchAll(/\bid="([^"]+)"/g))elements.set(id,element());
  return {getElementById:id=>{assert.ok(elements.has(id),`Missing element ${id}`);return elements.get(id);},createElement:element};
}
function memory() {const map=new Map();return {getItem:k=>map.get(k),setItem:(k,v)=>map.set(k,v),removeItem:k=>map.delete(k)};}
const context=()=>({session_id:'a'.repeat(32),position:34.25,generation:2,snapshot_updated:8,policy:defaultPolicy(),
  viewer_action:{visual:'blur',audio:'allow',blur:35,regional:false}});
const saved=body=>({...structuredClone(body),revision:1,status:'pending',category:'unspecified',note:'',
  context:{title:'Fixture <script>text</script>',live:false,analyzed_at_receipt:true},window:[24.25,36.25],
  media_available:true,created_at:'2026-10-15T00:00:00Z'});
function reporter(request, extra={}) {
  const document=dom('index.html'),storage=memory();
  const controller=installReporter({document,request,capture:context,storage,makeId:()=> 'fixture-id',...extra});
  return {$:id=>document.getElementById(id),storage,controller,document};
}

test('report buttons require media and one click freezes the moment without playback changes',async()=>{
  let active=null,resolve;const calls=[];
  const gate=new Promise(r=>{resolve=r;});
  const {$,controller}=reporter(async(path,body)=>{calls.push(structuredClone(body));await gate;return saved(body);},{capture:()=>active});
  assert.equal($('report-wrong').disabled,true);
  active=context();controller.refresh();assert.equal($('report-wrong').disabled,false);
  const pending=$('report-wrong').onclick();await $('report-wrong').onclick();
  active.position=70;active.policy.categories.nudity.visual='allow';resolve();await pending;
  assert.equal(calls.length,1);assert.equal(calls[0].position,34.25);
  assert.equal(calls[0].kind,'wrongly_censored');assert.equal(calls[0].policy.categories.nudity.visual,'hide');
  assert.equal($('source').pauses,undefined);assert.match($('feedback-status').textContent,/Saved.*0:34/);
  assert.equal($('report-missed').disabled,false);assert.equal($('feedback-details').open,false);
});

test('a failed save survives reload and retries the original media/time/ID',async()=>{
  const storage=memory(),calls=[];
  let active=context();
  const first=reporter(async(path,body)=>{calls.push(structuredClone(body));throw Error('offline');},{storage,capture:()=>active});
  await first.$('report-missed').onclick();assert.match(first.$('feedback-status').textContent,/Not saved/);
  active={...context(),position:80,session_id:'b'.repeat(32)};
  const second=reporter(async(path,body)=>{calls.push(structuredClone(body));return saved(body);},{storage,capture:()=>active});
  assert.equal(second.$('feedback-retry').hidden,false);await second.$('feedback-retry').onclick();
  assert.deepEqual(calls[0],calls[1]);assert.equal(calls[1].kind,'missed_content');
  assert.equal(storage.getItem('adaptive-feedback-pending-v1'),undefined);
});

test('optional details and undo use the saved report revision, even after switching media',async()=>{
  const calls=[];
  const {$}=reporter(async(path,body,method)=>{
    calls.push({path,body:structuredClone(body),method});
    return method==='PATCH'?{...saved(context()),id:'fixture-id',...body,revision:body.revision+1}:saved(body);
  });
  await $('report-wrong').onclick();$('feedback-category').value='violence';$('feedback-note').value='Normal conversation';
  await $('feedback-save-details').onclick();await $('feedback-undo').onclick();
  assert.equal(calls[1].body.note,'Normal conversation');assert.equal(calls[1].body.revision,1);
  assert.deepEqual(calls[2],{path:'/fixture-id',method:'PATCH',body:{revision:2,status:'withdrawn'}});
  assert.equal($('feedback-undo').hidden,true);assert.equal($('feedback-details').hidden,true);
});

test('restarting service renews authentication and retries an identical report only once',async()=>{
  const calls=[];let token='old';
  const request=feedbackApi({getToken:()=>token,setToken:value=>{token=value;},fetcher:async(url,options)=>{
    calls.push({url,options});
    if(calls.length===1)return {status:403,ok:false};
    return {status:200,ok:true,json:async()=>url==='/api/bootstrap'?{token:'new'}:{id:'fixture-id'}};
  }});
  await request('',{id:'fixture-id'});
  assert.equal(calls.length,3);assert.equal(token,'new');assert.equal(calls[0].options.body,calls[2].options.body);
  assert.equal(calls[2].options.headers['x-censor-token'],'new');
  let tries=0;
  const rejected=feedbackApi({getToken:()=>'',setToken:()=>{},fetcher:async()=>{tries++;return {status:403,ok:false,json:async()=>({detail:'denied'})};}});
  await assert.rejects(rejected('',{}),/Cannot reconnect/);assert.equal(tries,2);
});

test('queue keeps original hidden until explicitly requested, then opens the saved review window',async()=>{
  const document=dom('feedback.html'),$=id=>document.getElementById(id);
  $('report-filter').value='pending';const row=saved({...context(),id:'fixture-id',kind:'missed_content'}),calls=[];
  const queue=installQueue({document,request:async(path,body,method)=>{
    calls.push({path,body,method});return method==='PATCH'?{...row,...body,revision:2}:{total:1,reports:[row]};
  }});
  await queue.load();assert.equal($('report-video').src,undefined);
  $('report-list').children[0].onclick();assert.equal($('report-title').textContent,row.context.title);
  assert.equal($('report-video').hidden,true);assert.equal($('report-video').src,undefined);
  $('report-watch').onclick();$('report-video').onloadedmetadata();
  assert.equal($('report-video').currentTime,24.25);assert.equal($('report-video').paused,true);
  assert.equal($('report-video').src,'/api/media/'+row.session_id+'/source');
  $('review-category').value='kissing';$('review-status').value='confirmed';$('review-note').value='Needs exact interval';
  await $('review-save').onclick();assert.equal(calls[1].body.category,'kissing');
  assert.match($('review-message').textContent,/separate from training labels/);
});

test('queue handles unavailable media and preserves edits after a save conflict',async()=>{
  const document=dom('feedback.html'),$=id=>document.getElementById(id);$('report-filter').value='pending';
  const row={...saved({...context(),id:'fixture-id',kind:'wrongly_censored'}),media_available:false};
  const queue=installQueue({document,request:async(path,body,method)=>{
    if(method==='PATCH')throw Error('Report changed in another tab. Reload it before editing.');
    return {total:1,reports:[row]};
  }});
  await queue.load();$('report-list').children[0].onclick();assert.equal($('report-watch').disabled,true);
  $('report-watch').onclick();assert.equal($('report-video').src,undefined);
  $('review-note').value='My correction';await $('review-save').onclick();
  assert.equal($('review-note').value,'My correction');assert.match($('review-message').textContent,/Not saved.*Reload/);
});

test('clip retention is opt-in and is frozen in a retry along with the playback time',async()=>{
  const calls=[];let fail=true;
  const {$}=reporter(async(path,body)=>{calls.push(structuredClone(body));if(fail)throw Error('offline');return saved(body);});
  $('feedback-keep-clip').checked=true;await $('report-missed').onclick();
  $('feedback-keep-clip').checked=false;fail=false;await $('feedback-retry').onclick();
  assert.equal(calls[0].retain_clip,true);assert.deepEqual(calls[0],calls[1]);
});

test('unsaved review changes block navigation and refresh until saved or discarded',async()=>{
  const document=dom('feedback.html'),$=id=>document.getElementById(id);$('report-filter').value='pending';
  const rows=['one','two'].map(id=>saved({...context(),id,kind:'missed_content'}));let reads=0;
  const queue=installQueue({document,request:async()=>{reads++;return {total:2,reports:rows};}});
  await queue.load();$('report-list').children[0].onclick();
  $('review-note').value='Unsaved correction';$('review-note').oninput();
  $('report-list').children[1].onclick();assert.equal($('review-note').value,'Unsaved correction');
  await $('report-refresh').onclick();assert.equal(reads,1);assert.match($('review-message').textContent,/Save or discard/);
  $('review-discard').onclick();$('report-list').children[1].onclick();assert.equal($('review-note').value,'');
});

test('retained clip is used after original removal and a precise task requires an explicit action',async()=>{
  const document=dom('feedback.html'),$=id=>document.getElementById(id);$('report-filter').value='confirmed';
  const row={...saved({...context(),id:'clip-id',kind:'missed_content'}),status:'confirmed',category:'violence',media_available:false,
    evidence:{status:'ready',duration:12,bytes:1000}};
  const calls=[];
  const queue=installQueue({document,request:async(path,body,method)=>{
    calls.push({path,body,method});
    if(method==='GET')return {total:1,reports:[row]};
    return {...row,evidence:{...row.evidence,task:{url:'/label.html#feedback-review-v1/clip-id',title:'Example show',target:[2,5],complete:false}}};
  }});
  await queue.load();$('report-list').children[0].onclick();
  assert.equal(calls.length,1);assert.equal($('report-video').hidden,true);
  $('report-watch').onclick();$('report-video').onloadedmetadata();
  assert.equal($('report-video').src,'/api/feedback/clip-id/evidence');assert.equal($('report-video').currentTime,0);
  $('report-video').currentTime=2;$('label-mark-start').onclick();
  $('report-video').currentTime=5;$('label-mark-end').onclick();$('label-title').value='Example show';
  await $('create-label-task').onclick();
  assert.deepEqual(calls[1],{path:'/clip-id/labeling',body:{title:'Example show',start:2,end:5},method:'POST'});
  assert.equal($('open-label-task').href,'/label.html#feedback-review-v1/clip-id');
  assert.equal($('create-label-task').disabled,true);
});

test('clip polling updates availability without discarding unsaved notes or triggering playback',async()=>{
  const document=dom('feedback.html'),$=id=>document.getElementById(id);$('report-filter').value='pending';
  const row={...saved({...context(),id:'clip-id',kind:'missed_content'}),evidence:{status:'saving'}};
  const queue=installQueue({document,request:async(path)=>path.startsWith('?')?{total:1,reports:[row]}:
    {...row,note:'Older server note',evidence:{status:'ready',duration:11.95,bytes:2000}}});
  await queue.load();$('report-list').children[0].onclick();$('review-note').value='My new note';$('review-note').oninput();
  await queue.poll();assert.equal($('review-note').value,'My new note');
  assert.equal($('label-end').value,'11.95');assert.equal($('report-video').src,undefined);
  assert.match($('evidence-status').textContent,/retained locally/);
});

test('training promotion waits for completed human labels and surfaces a held-out-title rejection',async()=>{
  const document=dom('feedback.html'),$=id=>document.getElementById(id);$('report-filter').value='confirmed';
  const task={project:'feedback-review-v1',window:'clip-id',url:'/label.html#feedback-review-v1/clip-id',title:'Example',target:[0,4],complete:true,promoted:false};
  const row={...saved({...context(),id:'clip-id',kind:'missed_content'}),status:'confirmed',category:'violence',
    evidence:{status:'ready',duration:4,bytes:1000,task}};
  const queue=installQueue({document,request:async(path,body,method)=>{
    if(path.endsWith('/training'))throw Error('This title belongs to test and cannot enter training.');
    return {total:1,reports:[row]};
  }});
  await queue.load();$('report-list').children[0].onclick();assert.equal($('promote-labels').hidden,false);
  await $('promote-labels').onclick();assert.match($('label-task-status').textContent,/test and cannot enter training/);
  assert.equal(task.promoted,false);assert.equal($('open-label-task').href,task.url);
});
