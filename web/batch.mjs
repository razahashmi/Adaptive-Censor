const status=document.getElementById('status'),tasks=document.getElementById('tasks');
async function json(path,token){const response=await fetch(path,{headers:token?{'x-censor-token':token}:{}});if(!response.ok)throw Error(`Could not load the review batch (${response.status}).`);return response.json();}
try{
 const batch=await json('/batch.json'),bootstrap=await json('/api/bootstrap');
 const [video,masks,crops]=await Promise.all([json('/api/labeling',bootstrap.token),json('/api/segmentation',bootstrap.token),json('/api/roi-projects',bootstrap.token)]);
 const inventories={video,masks,crops};
 for(const task of batch.tasks){
  const project=inventories[task.type]?.find(p=>p.name===task.project);if(!project)throw Error('A prepared review queue is unavailable. Restart the service and refresh.');
  const card=document.createElement('section');card.className='task';
  const title=document.createElement('h2');title.textContent=task.title;card.append(title);
  const count=document.createElement('p');count.className='count';count.textContent=`${project.reviewed??project.complete??0} / ${project.total??project.windows} ${task.type==='video'?'clips':task.type==='crops'?'crop pairs':'frames'} completed`;card.append(count);
  const description=document.createElement('p');description.textContent=task.description;card.append(description);
  const link=document.createElement('a');link.href=task.url;link.textContent='Open review';card.append(link);tasks.append(card);
 }
 status.textContent=batch.description;
}catch(error){status.textContent=error.message;status.classList.add('error');}
