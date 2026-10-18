const $=id=>document.getElementById(id);
let busy=false;
async function check() {
  if(busy)return;busy=true;$('setup-refresh').disabled=true;
  $('setup-status').textContent='Checking your setup…';
  try {
    const bootstrap=await fetch('/api/bootstrap');if(!bootstrap.ok)throw Error('Cannot connect to the local service.');
    const response=await fetch('/api/readiness');if(!response.ok)throw Error('Setup check is unavailable. Restart the service and retry.');
    const result=await response.json();$('setup-checks').replaceChildren();
    $('setup-status').textContent=result.status==='blocked'?'Resolve the items below before playback.':
      result.status==='needs_setup'?'Controls demo is ready. Some analysis models still need setup.':'Base setup is ready. Review detections before relying on them.';
    for(const item of result.checks) {
      const section=document.createElement('section');section.className='panel';
      const heading=document.createElement('h2');heading.textContent=`${item.status==='ok'?'✓':item.status==='error'?'Needs attention':'Check'} · ${item.label}`;
      const detail=document.createElement('p');detail.className='hint';detail.textContent=item.detail;
      section.append(heading,detail);
      if(item.action){const action=document.createElement('p');action.textContent=item.action;section.append(action);}
      $('setup-checks').append(section);
    }
  } catch(error) {$('setup-status').textContent=error.message;}
  finally {busy=false;$('setup-refresh').disabled=false;}
}
$('setup-refresh').onclick=check;
await check();
