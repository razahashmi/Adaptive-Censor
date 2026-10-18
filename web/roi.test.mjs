// Exercise actual page handlers without a browser or production annotations.
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
let instance=0;
const ids=[...readFileSync(new URL('./roi.html',import.meta.url),'utf8').matchAll(/id="([^"]+)"/g)].map(m=>m[1]);
const clone=x=>structuredClone(x);
function deferred(){let resolve;const promise=new Promise(r=>resolve=r);return {resolve,promise};}
async function page(options={}){
  const originals=new Map();
  class Element{
    constructor(){Object.assign(this,{value:'',textContent:'',disabled:false,dataset:{},classList:{toggle(){}},children:[]});}
    append(e){this.children.push(e);}removeAttribute(name){delete this[name];}
  }
  const elements=Object.fromEntries(ids.map(id=>[id,new Element()]));
  const choices=['1','0','unsure'].map(label=>Object.assign(new Element(),{dataset:{label}}));
  const project={name:'fixture',revision:null,dataset_sha256:'bound',feature_views:['local','body'],rows:[
    {id:'roi-0000',title:'First',time:1,label:null,reviewed:false},
    {id:'roi-0001',title:'Second',time:2,label:null,reviewed:false}]};
  let bootstraps=0,writes=0;
  const response=(status,body)=>({status,ok:status>=200&&status<300,json:async()=>clone(body)});
  const globals={document:{getElementById:id=>elements[id],createElement:()=>new Element(),
    querySelectorAll:selector=>selector==='[data-label]'?choices:[...choices,elements.previous,elements.next,elements.project]},
    window:{addEventListener(){}},location:{hash:'#fixture'},fetch:async(url,init={})=>{
      if(url==='/api/bootstrap'){bootstraps++;return response(200,{token:`t${bootstraps}`});}
      if(url==='/api/roi-projects')return response(200,[{name:'fixture',total:2,reviewed:0}]);
      if(url==='/api/roi/fixture')return response(200,project);
      if(init.method==='PUT'){
        writes++;const payload=JSON.parse(init.body);
        if(options.write)return options.write({payload,writes,init,response,project});
        return response(200,{row:{id:url.split('/').at(-1),label:payload.label,labeler:'human',reviewed:true},revision:'r1'});
      }
      return response(404,{detail:'Missing crop project'});
    }};
  for(const [key,value] of Object.entries(globals)){originals.set(key,Object.getOwnPropertyDescriptor(globalThis,key));Object.defineProperty(globalThis,key,{value,writable:true,configurable:true});}
  const cleanup=()=>{for(const [key,old] of originals){if(old)Object.defineProperty(globalThis,key,old);else delete globalThis[key];}};
  try{await import(`./roi.mjs?test=${++instance}`);return {elements,choices,cleanup,bootstraps:()=>bootstraps,writes:()=>writes};}
  catch(e){cleanup();throw e;}
}
test('crop stays open until its save is acknowledged',async()=>{
  const pending=deferred();const p=await page({write:()=>pending.promise});
  try{const save=p.choices[0].onclick();assert.equal(p.elements.detail.textContent,'First · 1.00s');
    assert.equal(p.elements.next.disabled,true);p.elements.next.onclick();assert.match(p.elements.progress.textContent,/crop 1/);
    pending.resolve({status:200,ok:true,json:async()=>({row:{id:'roi-0000',label:1,reviewed:true},revision:'r1'})});
    await save;assert.equal(p.elements.detail.textContent,'Second · 2.00s');assert.match(p.elements.progress.textContent,/1\/2 reviewed/);
  }finally{p.cleanup();}
});
test('failed or stale save never advances the crop',async()=>{
  for(const code of [500,409]){const p=await page({write:({response})=>response(code,{detail:'Write not confirmed'})});
    try{await p.choices[1].onclick();assert.equal(p.elements.detail.textContent,'First · 1.00s');assert.match(p.elements.status.textContent,/Write not confirmed/);assert.equal(p.elements.next.disabled,false);}
    finally{p.cleanup();}}
});
test('expired token renews once and unsure saves explicitly',async()=>{
  const labels=[];const p=await page({write:({payload,writes,response})=>{labels.push(payload.label);return writes===1?response(403,{}):response(200,{row:{id:'roi-0000',label:null,labeler:'human',reviewed:true},revision:'r1'});}});
  try{await p.choices[2].onclick();assert.equal(p.bootstraps(),2);assert.deepEqual(labels,[null,null]);assert.equal(p.elements.detail.textContent,'Second · 2.00s');}
  finally{p.cleanup();}
});
test('failed project load clears the old crop before any further choice',async()=>{
  const p=await page();try{p.elements.project.value='missing';await p.elements.project.onchange();
    await p.choices[0].onclick();assert.equal(p.writes(),0);assert.equal(p.elements.local.src,undefined);assert.match(p.elements.status.textContent,/Missing crop project/);
  }finally{p.cleanup();}
});
