const $=id=>document.getElementById(id);
let project=null,index=0,token='',busy=false;
async function bootstrap(){const r=await fetch('/api/bootstrap');if(!r.ok)throw new Error('Service unavailable');token=(await r.json()).token;}
async function api(url,options={},retry=true){
  const r=await fetch(url,{...options,headers:{'Content-Type':'application/json','x-censor-token':token,...options.headers}});
  if(r.status===403&&retry){await bootstrap();return api(url,options,false);}
  if(!r.ok){const body=await r.json().catch(()=>({}));throw new Error(body.detail||`Save failed (${r.status})`);}return r.json();
}
function status(message,error=false){$('status').textContent=message;$('status').classList.toggle('error',error);}
function lock(value){busy=value;document.querySelectorAll('button,select').forEach(e=>e.disabled=value);}
function render(){
  const row=project?.rows[index];if(!row)return;
  $('local').src=`/api/roi/${project.name}/images/${row.id}.jpg`;
  $('body').hidden=!project.feature_views.includes('body');$('body').src=`/api/roi/${project.name}/images/${row.id}-body.jpg`;
  $('progress').textContent=`${project.rows.filter(r=>r.reviewed).length}/${project.rows.length} reviewed · crop ${index+1}`;
  $('detail').textContent=`${row.title} · ${row.time.toFixed(2)}s`;
  status(row.reviewed?`Saved: ${row.label===1?'exposed':row.label===0?'covered / look-alike':'unsure'}`:'Choose a label.');
}
async function load(name){lock(true);project=null;$('local').removeAttribute('src');$('body').removeAttribute('src');$('detail').textContent='Loading crops…';$('progress').textContent='';try{project=await api(`/api/roi/${name}`);index=Math.max(0,project.rows.findIndex(r=>!r.reviewed));location.hash=name;render();}catch(e){status(e.message,true);}finally{lock(false);}}
async function save(label){
  if(busy||!project)return;lock(true);const row=project.rows[index];status('Saving…');
  try{
    const result=await api(`/api/roi/${project.name}/labels/${row.id}`,{method:'PUT',body:JSON.stringify({label,revision:project.revision,dataset_sha256:project.dataset_sha256})});
    Object.assign(row,result.row);project.revision=result.revision;
    if(index<project.rows.length-1)index++;render();
    if(project.rows.every(r=>r.reviewed))status('All crop reviews are saved.');
  }catch(e){status(`${e.message}. Your choice has not been confirmed; retry or reload.`,true);}finally{lock(false);}
}
document.querySelectorAll('[data-label]').forEach(b=>b.onclick=()=>save(b.dataset.label==='unsure'?null:Number(b.dataset.label)));
$('previous').onclick=()=>{if(!busy&&project){index=Math.max(0,index-1);render();}};
$('next').onclick=()=>{if(!busy&&project){index=Math.min(project.rows.length-1,index+1);render();}};
$('project').onchange=()=>load($('project').value);
window.addEventListener('keydown',e=>{if(e.target.matches('input,select,textarea')||busy)return;if(e.key==='1'||e.key==='0')save(Number(e.key));if(e.key.toLowerCase()==='u')save(null);});
try{await bootstrap();const projects=await api('/api/roi-projects');for(const p of projects){const o=document.createElement('option');o.value=p.name;o.textContent=`${p.name} · ${p.reviewed}/${p.total}`;$('project').append(o);}
const name=projects.some(p=>p.name===location.hash.slice(1))?location.hash.slice(1):projects.at(-1)?.name;if(name){$('project').value=name;await load(name);}else status('No crop review project is prepared yet.');}catch(e){status(e.message,true);}
