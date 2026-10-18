import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
let instance=0;
const ids=[...readFileSync(new URL('./masks.html',import.meta.url),'utf8').matchAll(/id="([^"]+)"/g)].map(m=>m[1]);
const clone=x=>structuredClone(x);
function deferred(){let resolve;const promise=new Promise(r=>resolve=r);return {resolve,promise};}
async function page(options={}){
 const originals=new Map(),context={drawImage(){},clearRect(){},fillRect(){},beginPath(){},moveTo(){},lineTo(){},stroke(){},arc(){},fill(){},getImageData(){return {data:new Uint8ClampedArray(800)};},putImageData(){}};
 class Element{constructor(){Object.assign(this,{value:'10',textContent:'',disabled:false,width:20,height:10,classList:{toggle(){}},children:[]});}append(e){this.children.push(e);}removeAttribute(n){delete this[n];}setAttribute(n,v){this[n]=v;}getContext(){return context;}setPointerCapture(){}getBoundingClientRect(){return {left:0,top:0,width:20,height:10};}}
 const elements=Object.fromEntries(ids.map(id=>[id,new Element()]));let writes=0,drafts=0;
 const project={name:'fixture',split:options.split||'train',dataset_sha256:'bound',rows:[{id:'mask-00000',display:'First',time:1,width:20,height:10},{id:'mask-00001',display:'Second',time:2,width:20,height:10}]};
 const response=(status,body)=>({status,ok:status>=200&&status<300,json:async()=>clone(body)});
 class Picture{set src(url){this.url=url;if(url.endsWith('/draft'))drafts++;queueMicrotask(()=>this.onload());}get src(){return this.url;}}
 const globals={document:{getElementById:id=>elements[id],createElement:()=>new Element(),querySelectorAll:()=>Object.values(elements)},window:{addEventListener(){}},Image:Picture,location:{hash:options.hash||'#fixture'},fetch:async(url,init={})=>{
  if(url==='/api/bootstrap')return response(200,{token:'t'});if(url==='/api/segmentation')return response(200,options.projects||[{name:'fixture',split:project.split,reviewed:0,total:2}]);
  if(url==='/api/segmentation/fixture')return response(200,project);
  if(init.method==='PUT'){writes++;const payload=JSON.parse(init.body);return options.write?options.write({payload,writes,response}):response(200,{label:{status:payload.status},revision:'new'});}
  if(url.includes('/labels/'))return response(200,{label:null,revision:null});return response(404,{detail:'Missing project'});
 }};
 for(const [key,value] of Object.entries(globals)){originals.set(key,Object.getOwnPropertyDescriptor(globalThis,key));Object.defineProperty(globalThis,key,{value,writable:true,configurable:true});}
 const cleanup=()=>{for(const [key,old] of originals){if(old)Object.defineProperty(globalThis,key,old);else delete globalThis[key];}};
 try{await import(`./masks.mjs?test=${++instance}`);return {elements,cleanup,writes:()=>writes,drafts:()=>drafts};}catch(e){cleanup();throw e;}
}
test('mask save waits for acknowledgement before advancing',async()=>{const d=deferred(),p=await page({write:()=>d.promise});try{const saved=p.elements.save.onclick();assert.match(p.elements.detail.textContent,/First/);assert.equal(p.elements.next.disabled,true);d.resolve({ok:true,status:200,json:async()=>({label:{status:'reviewed'},revision:'new'})});await saved;assert.match(p.elements.detail.textContent,/Second/);}finally{p.cleanup();}});
test('failed mask save retains frame and strokes, and unsaved navigation is blocked',async()=>{const sent=[],p=await page({write:({payload,response})=>{sent.push(payload);return response(409,{detail:'Stale revision'});}});try{p.elements.editor.onpointerdown({clientX:10,clientY:5,pointerId:1});p.elements.editor.onpointerup();await p.elements.next.onclick();assert.match(p.elements.detail.textContent,/First/);await p.elements.save.onclick();assert.equal(sent[0].strokes.length,1);assert.match(p.elements.status.textContent,/Outline retained/);assert.match(p.elements.detail.textContent,/First/);}finally{p.cleanup();}});
test('evaluation drafts require explicit second reveal and clean save stays marked as human correction',async()=>{const sent=[],p=await page({split:'test',write:({payload,response})=>{sent.push(payload);return response(200,{label:{status:'reviewed'},revision:'new'});}});try{assert.equal(p.drafts(),0);await p.elements.draft.onclick();assert.equal(p.drafts(),0);await p.elements.draft.onclick();assert.equal(p.drafts(),1);await p.elements.clean.onclick();assert.equal(sent[0].draft_included,false);assert.deepEqual(sent[0].strokes,[]);}finally{p.cleanup();}});
test('failed mask project load cannot save the previous picture',async()=>{const p=await page();try{p.elements.project.value='missing';await p.elements.project.onchange();await p.elements.save.onclick();assert.equal(p.writes(),0);assert.equal(p.elements.original.src,undefined);}finally{p.cleanup();}});
test('combined queue hides duplicate source queues and resets a source frame bookmark',async()=>{const p=await page({hash:'#old/mask-00001',projects:[{name:'old',split:'train',reviewed:0,total:2},{name:'fixture',split:'train',reviewed:0,total:2,supersedes:['old']}]});try{assert.equal(p.elements.project.children.length,1);assert.match(p.elements.project.children[0].textContent,/combined/);assert.match(p.elements.detail.textContent,/First/);assert.equal(globalThis.location.hash,'fixture/mask-00000');}finally{p.cleanup();}});
