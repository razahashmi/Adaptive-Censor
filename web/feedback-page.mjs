import {feedbackApi} from './feedback.mjs';
import {installQueue} from './feedback-queue.mjs';
let token='';
const request=feedbackApi({getToken:()=>token,setToken:value=>{token=value;}});
const queue=installQueue({document,request});
await queue.load();
setInterval(()=>queue.poll(),1500);
