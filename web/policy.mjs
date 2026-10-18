export const categories = ['nudity', 'sexual_activity', 'violence', 'gore', 'kissing', 'language', 'custom'];
export const labels = {nudity:'Nudity',sexual_activity:'Sexual scenes',violence:'Violence',gore:'Gore',kissing:'Kissing',language:'Language',custom:'Custom visual'};
const visuals = ['allow','blur','hide'], audios = ['allow','beep','mute'];

export function defaultPolicy() {
  const rules = Object.fromEntries(categories.map(category => [category, {visual:'hide', audio:'allow', blur:50, min_confidence:.55, min_severity:1}]));
  rules.violence.visual = 'blur'; rules.violence.blur = 35;
  rules.kissing.visual = 'allow'; rules.custom.visual = 'blur';
  rules.sexual_activity.audio = 'mute';
  rules.language = {...rules.language, visual:'allow', audio:'beep', min_confidence:0};
  for(const category of ['gore','kissing','custom'])rules[category].min_confidence=.70;
  rules.sexual_activity.min_confidence=.65;rules.violence.min_confidence=.65;rules.nudity.scope='region';
  return {pre_padding:.3, post_padding:.35, categories:rules};
}

export function validPolicy(input) {
  if (!input || !Number.isFinite(input.pre_padding) || !Number.isFinite(input.post_padding) ||
      input.pre_padding < 0 || input.pre_padding > 1 || input.post_padding < 0 || input.post_padding > 1) throw Error('Padding must be 0–1 seconds.');
  for (const category of categories) {
    const r = input.categories?.[category];
    if (!r || !visuals.includes(r.visual) || !audios.includes(r.audio) || !Number.isFinite(r.blur) || r.blur<1 || r.blur>100 ||
        !Number.isFinite(r.min_confidence) || r.min_confidence<0 || r.min_confidence>1 ||
        (r.scope!==undefined && !['frame','region'].includes(r.scope))) throw Error(`Invalid rule for ${category}.`);
  }
  return structuredClone(input);
}

export function selectedEvents(events, policy) {
  return events.filter(e => e.enabled !== false && policy.categories[e.category] &&
    (!(e.review_required || e.source?.startsWith('CLIP')) || e.confirmed === true) &&
    (e.severity ?? 2) >= (policy.categories[e.category].min_severity ?? 1) &&
    e.confidence >= policy.categories[e.category].min_confidence &&
    (policy.categories[e.category].visual !== 'allow' || policy.categories[e.category].audio !== 'allow'));
}

export function compilePlan(events, policy, duration) {
  const changes = new Map();
  for (const event of selectedEvents(events,policy)) {
    const rule = policy.categories[event.category];
    const [low,high]=event.time_bounds||[0,duration];
    const start = Math.max(0,low,event.start-policy.pre_padding), end = Math.min(duration,high,event.end+policy.post_padding);
    if (start >= end) continue;
    const region=(rule.scope==='region' || event.source==='manual') && event.region && rule.visual!=='allow'?event.region:null;
    const key = JSON.stringify([rule.visual,rule.visual==='blur'?rule.blur:0,rule.audio,region]);
    for (const [time,delta] of [[start,1],[end,-1]]) {
      if (!changes.has(time)) changes.set(time,[]);
      changes.get(time).push([key,delta]);
    }
  }
  const times = [...changes.keys()].sort((a,b)=>a-b), counts = new Map(), plan = [];
  for (let i=0;i<times.length-1;i++) {
    const start=times[i], end=times[i+1];
    for (const [key,delta] of changes.get(start)) {
      const count=(counts.get(key)||0)+delta;
      if (count) counts.set(key,count); else counts.delete(key);
    }
    if (!counts.size) continue;
    let visual='allow', audio='allow', blur=0;const regions=new Map();
    for (const key of counts.keys()) {
      const [v,b,a,box]=JSON.parse(key);
      if(box)regions.set(JSON.stringify([box,v,b]),{box,visual:v,blur:b});
      else if (visuals.indexOf(v)>visuals.indexOf(visual)) {visual=v;blur=+b;}
      else if(v===visual) blur=Math.max(blur,+b);
      if(audios.indexOf(a)>audios.indexOf(audio)) audio=a;
    }
    const item={start,end,visual,audio,blur};
    if(regions.size && visual!=='hide')item.regions=[...regions.entries()].sort(([a],[b])=>a.localeCompare(b)).map(([,r])=>r);
    const last=plan.at(-1);
    if(last && last.end===start && last.visual===visual && last.audio===audio && last.blur===blur && JSON.stringify(last.regions||[])===JSON.stringify(item.regions||[])) last.end=end;
    else plan.push(item);
  }
  return plan;
}

export function actionAt(plan,time) {
  // Binary search matters for long recordings with thousands of decisions.
  let left=0,right=plan.length;
  while(left<right) {const middle=(left+right)>>1;if(plan[middle].start<=time)left=middle+1;else right=middle;}
  const item=plan[left-1];
  return item && time<item.end ? item : {visual:'allow',audio:'allow',blur:0};
}

export function coverageEnd(coverage,time) {
  return coverage.find(([start,end])=>start<=time+.001 && end>time)?.[1] ?? time;
}

export function audioSchedule(plan,coverage,time,duration,horizon=1) {
  // The final zero remains scheduled even if the main thread stops responding.
  const covered=coverageEnd(coverage,time), end=Math.min(time+horizon,covered>=duration-.01?duration:covered-.12);
  if(end<=time) return [{time,source:0,beep:0}];
  const bounds=new Set([time,end]);
  for(const e of plan) if(e.audio!=='allow' && e.end+.06>time && e.start-.06<end) {
    bounds.add(Math.max(time,e.start-.06)); bounds.add(Math.min(end,e.end+.06));
  }
  const result=[];
  for(const t of [...bounds].sort((a,b)=>a-b)) {
    let mode='allow';
    for(const e of plan) if(e.start-.06<=t+1e-6 && e.end+.06>t+1e-6 && audios.indexOf(e.audio)>audios.indexOf(mode)) mode=e.audio;
    result.push({time:t,source:t>=end?0:mode==='allow'?1:0,beep:t>=end?0:mode==='beep'?.08:0});
  }
  return result;
}
