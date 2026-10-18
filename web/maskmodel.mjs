export function point(event,canvas){const b=canvas.getBoundingClientRect();return [Math.min(1,Math.max(0,(event.clientX-b.left)/b.width)),Math.min(1,Math.max(0,(event.clientY-b.top)/b.height))];}
export function payload(project,row,revision,strokes,draftIncluded,status){return {dataset_sha256:project.dataset_sha256,revision,strokes:structuredClone(strokes),draft_included:draftIncluded,status};}
export function nextIndex(rows,index){return Math.min(rows.length-1,index+1);}
