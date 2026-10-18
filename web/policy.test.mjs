import {test} from 'node:test';
import assert from 'node:assert/strict';
import {defaultPolicy,compilePlan,actionAt,coverageEnd,audioSchedule,validPolicy} from './policy.mjs';

test('uncertain and legacy CLIP suggestions cannot filter until explicitly confirmed',()=>{
  for(const detail of [{review_required:true},{source:'CLIP experimental'},{source:'CLIP experimental',review_required:false}]){
    const event={start:1,end:2,category:'sexual_activity',confidence:1,...detail};
    assert.deepEqual(compilePlan([event],defaultPolicy(),3),[]);
    assert.equal(actionAt(compilePlan([{...event,confirmed:true}],defaultPolicy(),3),1.5).visual,'hide');
    assert.deepEqual(compilePlan([{...event,confirmed:true,enabled:false}],defaultPolicy(),3),[]);
  }
});

test('overlapping picture and sound resolve independently, with half-open boundaries',()=>{
  const policy=defaultPolicy();policy.pre_padding=policy.post_padding=0;
  const events=[{start:1,end:4,category:'violence',confidence:1},{start:2,end:3,category:'sexual_activity',confidence:1},{start:0,end:5,category:'language',confidence:1}];
  const plan=compilePlan(events,policy,6);
  assert.equal(actionAt(plan,1.5).visual,'blur');assert.equal(actionAt(plan,1.5).audio,'beep');
  assert.equal(actionAt(plan,2.5).visual,'hide');assert.equal(actionAt(plan,2.5).audio,'mute');
  assert.equal(actionAt(plan,3).visual,'blur');assert.equal(actionAt(plan,5).audio,'allow');
});
test('coverage holes stay closed after a seek',()=>{
  assert.equal(coverageEnd([[0,6],[18,30]],10),10);
  assert.deepEqual(audioSchedule([],[[0,6],[18,30]],10,30),[{time:10,source:0,beep:0}]);
});
test('audio has an independently scheduled fail-closed deadline',()=>{
  const points=audioSchedule([],[[0,6]],5.5,30,2);
  assert.deepEqual(points[0],{time:5.5,source:1,beep:0});
  assert.deepEqual(points.at(-1),{time:5.88,source:0,beep:0});
});
test('audio boundaries include scheduling guard; mute beats beep',()=>{
  const plan=[{start:.5,end:1,visual:'allow',audio:'beep'},{start:.7,end:.8,visual:'allow',audio:'mute'}];
  const points=audioSchedule(plan,[[0,4]],0,4,2);
  assert.equal(points.find(p=>Math.abs(p.time-.44)<1e-6).beep,.08);
  assert.equal(points.find(p=>Math.abs(p.time-.64)<1e-6).source,0);
  assert.equal(points.find(p=>Math.abs(p.time-.64)<1e-6).beep,0);
  assert.equal(points.at(-1).source,0);
});
test('disabled/low-score events do not filter; imported invalid policies fail',()=>{
  const p=defaultPolicy();
  assert.deepEqual(compilePlan([{start:0,end:3,category:'nudity',confidence:.1},{start:0,end:3,category:'nudity',confidence:1,enabled:false}],p,4),[]);
  p.categories.nudity.blur=Infinity;assert.throws(()=>validPolicy(p));
});

test('localized masks preserve full frame, fall back without boxes, and respect full concealment',()=>{
  const p=defaultPolicy();p.pre_padding=p.post_padding=0;
  const events=[{start:1,end:3,category:'nudity',confidence:1,region:[.1,.2,.3,.4]}, {start:2,end:4,category:'sexual_activity',confidence:1}];
  const plan=compilePlan(events,p,5);
  assert.equal(actionAt(plan,1.5).visual,'allow');assert.deepEqual(actionAt(plan,1.5).regions[0].box,[.1,.2,.3,.4]);
  assert.equal(actionAt(plan,2.5).visual,'hide');assert.equal(actionAt(plan,2.5).regions,undefined);
  assert.equal(actionAt(compilePlan([{start:0,end:1,category:'nudity',confidence:1}],p,2),.5).visual,'hide');
});

test('balanced scene thresholds reject weak suggestions and manual object masks work',()=>{
  const p=defaultPolicy();
  assert.deepEqual(compilePlan([{start:0,end:2,category:'sexual_activity',confidence:.6}],p,3),[]);
  const action=actionAt(compilePlan([{start:0,end:2,category:'violence',confidence:1,source:'manual',region:[.1,.1,.2,.2]}],p,3),1);
  assert.equal(action.visual,'allow');assert.equal(action.regions[0].visual,'blur');
});

test("shot bounds stop padding at a cut",()=>{
 const p=defaultPolicy();
 const plan=compilePlan([{start:.7,end:1,category:"nudity",confidence:1,time_bounds:[0,1],region:[.1,.1,.2,.2]}],p,3);
 assert.equal(plan.at(-1).end,1);
 assert.equal(actionAt(plan,1.01).visual,"allow");
 assert.equal(actionAt(plan,1.01).regions,undefined);
});
