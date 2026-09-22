const $=id=>document.getElementById(id);
const video=$('video'),empty=$('empty'),file=$('file'),camera=$('camera'),record=$('record'),processButton=$('process'),status=$('status'),bar=$('bar'),download=$('download'),canvas=$('skeleton'),ctx=canvas.getContext('2d'),view=$('view'),legend=$('legend');
let sourceURL=null,stream=null,recorder=null,chunks=[],job=null,playing=false,viewYaw=null,latestPose=null;

function message(text,progress){status.textContent=text;if(progress!==undefined)bar.style.width=`${Math.max(0,Math.min(100,progress))}%`}
async function ready(){
  if(!Number.isFinite(video.duration)){video.currentTime=Number.MAX_SAFE_INTEGER;await new Promise(resolve=>video.addEventListener('seeked',resolve,{once:true}));video.currentTime=0}
  empty.hidden=true;video.classList.add('ready');processButton.disabled=false;message(`Ready · ${video.videoWidth}×${video.videoHeight} · ${Number.isFinite(video.duration)?video.duration.toFixed(1):'?'} s`,0)
}
function useBlob(blob){playing=false;clearPreview();if(sourceURL)URL.revokeObjectURL(sourceURL);sourceURL=URL.createObjectURL(blob);video.srcObject=null;video.src=sourceURL;video.load();video.onloadedmetadata=()=>ready().catch(e=>message(e.message,0))}

const mode=$('mode'),liveStart=$('live-start'),liveStop=$('live-stop'),cameraDevice=$('camera-device');
let liveID=null,liveEpoch=0,liveAbort=null,liveRunning=false,liveDisplayed=null,offlineBusy=false,lastLivePose=0,liveStopping=false;
function updatePreview(){
  const showPose=!!latestPose&&(liveRunning||playing);
  canvas.hidden=!showPose;canvas.parentElement.classList.toggle('show-pose',showPose);
  view.disabled=!showPose;legend.hidden=!showPose||view.value!=='overlay';
  $('playback-controls').hidden=mode.value!=='offline'||!playing;
}
function clearPreview(){latestPose=null;viewYaw=null;legend.hidden=true;ctx.fillStyle='#0b1020';ctx.fillRect(0,0,canvas.width,canvas.height);updatePreview()}
function cameraControls(){
  const live=mode.value==='live';
  $('detect-setting').hidden=!live;$('detect-interval').disabled=liveRunning||liveStopping;
  $('file-label').hidden=live;record.hidden=live;liveStart.hidden=!live;liveStop.hidden=!live;
  processButton.hidden=live;download.hidden=live;$('live-metrics').hidden=!live;
  $('rate-label').textContent=live?'Rate cap':'Sample rate';
  liveStop.textContent=liveRunning?'Stop inference':'Close camera';
  updatePreview();
  liveStart.disabled=liveRunning||liveStopping||offlineBusy;liveStop.disabled=liveStopping||(!liveRunning&&!stream);
  camera.disabled=liveRunning||liveStopping||offlineBusy||!!stream;cameraDevice.disabled=liveRunning||liveStopping||offlineBusy;
  record.disabled=!stream||live;mode.disabled=offlineBusy||liveStopping;file.disabled=offlineBusy;
  $('capture-hint').textContent=live?'Keep the camera still and your whole body in view. Between detections, stay in the same area while moving your limbs. Set 1 to detect every frame.':'Record a clip or choose a video, then build motion using the complete sequence.';
}
function closeCamera(){if(stream)stream.getTracks().forEach(t=>t.stop());stream=null;video.srcObject=null;video.controls=true;video.classList.remove('ready');empty.hidden=false;cameraControls()}
async function stopLive(note='Live stopped.',keepCamera=false){
  ++liveEpoch;liveStopping=true;liveRunning=false;liveAbort?.abort();liveAbort=null;
  const id=liveID;liveID=null;if(!keepCamera)closeCamera();liveDisplayed=null;clearPreview();$('live-metrics').textContent='';cameraControls();message(note,0);
  try{if(id)await fetch(`/api/live/${id}`,{method:'DELETE',keepalive:true})}catch{}finally{liveStopping=false;cameraControls()}
}
async function openCamera(){
  if(!window.isSecureContext||!navigator.mediaDevices?.getUserMedia){$('camera-notice').hidden=false;throw new Error('Camera access requires HTTPS or localhost.')}
  video.onloadedmetadata=null;video.loop=false;playing=false;clearPreview();
  const epoch=liveEpoch;
  const acquired=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:1280},height:{ideal:720},...(cameraDevice.value?{deviceId:{exact:cameraDevice.value}}:{})},audio:false});
  if(epoch!==liveEpoch){acquired.getTracks().forEach(t=>t.stop());return false}
  stream=acquired;video.removeAttribute('src');video.srcObject=stream;video.controls=false;await video.play();
  if(epoch!==liveEpoch)return false;
  empty.hidden=true;video.classList.add('ready');processButton.disabled=true;
  const devices=await navigator.mediaDevices.enumerateDevices(),selected=stream.getVideoTracks()[0].getSettings().deviceId;
  cameraDevice.replaceChildren(new Option('Default camera',''),...devices.filter(d=>d.kind==='videoinput').map((d,i)=>new Option(d.label||`Camera ${i+1}`,d.deviceId)));
  cameraDevice.value=selected||'';
  stream.getVideoTracks()[0].onended=()=>stopLive('Camera disconnected. Start again to reconnect.');
  cameraControls();return true;
}
file.onchange=()=>{if(file.files[0]){closeCamera();useBlob(file.files[0])}};
camera.onclick=async()=>{camera.disabled=true;try{if(await openCamera())message(mode.value==='live'?'Camera ready. Press Start live.':'Camera ready. Record a short clip, then process it.',0)}catch(e){message(`Camera: ${e.message}`,0)}finally{cameraControls()}};
cameraDevice.onchange=async()=>{await stopLive('Camera changed. Press Start live or open the camera.');};
mode.onchange=async()=>{$('fps').value=mode.value==='live'?'20':'10';if(recorder?.state==='recording'){recorder.onstop=null;recorder.stop();record.textContent='Start recording'}await stopLive(mode.value==='live'?'Open your camera or press Start live.':'Choose a video or record a clip.');playing=false;job=null;video.removeAttribute('src');video.load();video.classList.remove('ready');empty.hidden=false;processButton.disabled=true;download.classList.add('disabled');cameraControls()};
liveStop.onclick=()=>liveRunning?stopLive('Inference stopped. Camera preview is active.',true):stopLive('Camera closed.');
liveStart.onclick=async()=>{
  const epoch=++liveEpoch;liveRunning=true;playing=false;job=null;clearPreview();download.classList.add('disabled');cameraControls();
  liveAbort=new AbortController();
  try{
    if(!stream&&!await openCamera())return;
    if(epoch!==liveEpoch)return;
    message('Loading live models…',0);
    const detectionInterval=Number($('detect-interval').value);
    if(!Number.isInteger(detectionInterval)||detectionInterval<1||detectionInterval>30)throw new Error('Detection interval must be a whole number from 1 to 30.');
    const session=await api(`/api/live?detect_interval=${detectionInterval}&pipeline=2`,{method:'POST',signal:liveAbort.signal});
    if(epoch!==liveEpoch){fetch(`/api/live/${session.id}`,{method:'DELETE'}).catch(()=>{});return}
    liveID=session.id;
    const ratio=Math.min(1,960/Math.max(video.videoWidth,video.videoHeight)),width=Math.max(8,Math.round(video.videoWidth*ratio)),height=Math.max(8,Math.round(video.videoHeight*ratio));
    // Two separate image buffers retain the exact source image for each result.
    // Start preparing the next frame shortly before the current inference ends.
    const buffers=Array.from({length:2},()=>{const c=document.createElement('canvas');c.width=width;c.height=height;return c});
    liveDisplayed=document.createElement('canvas');liveDisplayed.width=width;liveDisplayed.height=height;
    job={width,height};view.value='overlay';let previousResult=null,encodeMs=5,transportMs=4,plainMs=55,detectMs=110,nextDetection=1;
    const sourceOrigin=performance.now();
    const signal=liveAbort.signal,url=`/api/live/${liveID}/frame`;
    const submit=async(index,earliest=0)=>{
      await new Promise(resolve=>setTimeout(resolve,Math.max(0,earliest-performance.now())));
      if(epoch!==liveEpoch)return null;
      if(video.readyState<2)throw new Error('Camera stopped delivering frames.');
      const capture=buffers[(index-1)%2],captured=performance.now();
      capture.getContext('2d',{alpha:false}).drawImage(video,0,0,width,height);
      const blob=await jpeg(capture);if(epoch!==liveEpoch)return null;
      const sent=performance.now();encodeMs=.75*encodeMs+.25*(sent-captured);
      const result=(async()=>{
        const response=await fetch(url,{method:'PUT',headers:{'Content-Type':'image/jpeg','X-GEMX-Frame':String(index),'X-GEMX-Source-Time-Us':String(Math.round((captured-sourceOrigin)*1000))},body:blob,signal});
        if(!response.ok){const error=await response.json();throw new Error(error.error||'Live inference failed')}
        if(Number(response.headers.get('X-GEMX-Sequence'))!==index)throw new Error('Live frame sequence mismatch');
        const pose=response.status===204?null:parsePose(await response.arrayBuffer());
        const queued=Number(response.headers.get('Server-Timing')?.match(/(?:^|,\s*)queue;dur=([\d.]+)/)?.[1]||0);
        return {pose,queued,now:performance.now(),inference:Number(response.headers.get('X-GEMX-Inference-Ms')),people:Number(response.headers.get('X-GEMX-People'))};
      })().then(value=>({value}),error=>({error}));
      return {index,capture,captured,sent,result};
    };
    // Always handle a lookahead rejection, including stop/restart during encoding.
    const outcome=p=>p.then(value=>({value}),error=>({error}));
    let current=await submit(1);
    while(current&&epoch===liveEpoch){
      const detecting=current.index>=nextDetection;
      const cap=Math.max(1,Math.min(30,Number($('fps').value)||10));
      const predicted=detecting?detectMs:plainMs;
      const next=outcome(submit(current.index+1,Math.max(current.captured+1000/cap,Math.max(current.sent,previousResult||0)+predicted-encodeMs-transportMs-4)));
      const received=await current.result;if(received.error)throw received.error;
      if(epoch!==liveEpoch)break;
      const {pose,queued,now,inference,people}=received.value,age=Math.round(now-current.captured);
      transportMs=.75*transportMs+.25*Math.max(0,now-current.sent-inference-queued);
      if(detecting)detectMs=.75*detectMs+.25*inference;else plainMs=.75*plainMs+.25*inference;
      if(detecting)nextDetection=current.index+(people?detectionInterval:1);
      if(!pose){clearPreview();message('Warming up · waiting for the second frame…',0)}
      else {
        liveDisplayed.getContext('2d').drawImage(current.capture,0,0);lastLivePose=performance.now();draw(pose);
        message(people?'Live · keep your whole body in view':'Live · no person detected; using the full frame',100);
      }
      $('live-metrics').textContent=`${previousResult?(1000/(now-previousResult)).toFixed(1):'—'} fps · ${age} ms frame age · ${inference} ms inference · 30-frame context`;
      previousResult=now;
      const prepared=await next;if(prepared.error)throw prepared.error;current=prepared.value;
    }
  }catch(e){if(epoch===liveEpoch)await stopLive(e.name==='AbortError'?'Live stopped.':e.message)}
};
document.addEventListener('visibilitychange',()=>{if(document.hidden&&mode.value==='live'&&(liveRunning||stream))stopLive('Live paused while the tab is hidden. Press Start live to resume.')});
setInterval(()=>{if(liveRunning&&latestPose&&performance.now()-lastLivePose>2000){clearPreview();message('Waiting for a fresh camera pose…',0)}},500);
window.addEventListener('pagehide',()=>{liveAbort?.abort();if(liveID)fetch(`/api/live/${liveID}`,{method:'DELETE',keepalive:true}).catch(()=>{});stream?.getTracks().forEach(t=>t.stop())});
cameraControls();

record.onclick=()=>{if(!recorder||recorder.state==='inactive'){
  chunks=[];const type=['video/webm;codecs=vp9','video/webm;codecs=vp8','video/webm'].find(value=>MediaRecorder.isTypeSupported(value))||'';recorder=new MediaRecorder(stream,type?{mimeType:type}:undefined);
  recorder.ondataavailable=e=>{if(e.data.size)chunks.push(e.data)};recorder.onstop=()=>{const blob=new Blob(chunks,{type:recorder.mimeType});stream.getTracks().forEach(t=>t.stop());stream=null;video.controls=true;record.disabled=true;record.textContent='Start recording';cameraControls();useBlob(blob)};
  recorder.start(250);record.textContent='Stop recording';message('Recording… inference remains idle.',0)
}else recorder.stop()};

function seek(time){return new Promise((resolve,reject)=>{const done=()=>{cleanup();resolve()},bad=()=>{cleanup();reject(new Error('could not decode this point in the clip'))},cleanup=()=>{video.removeEventListener('seeked',done);video.removeEventListener('error',bad)};video.addEventListener('seeked',done,{once:true});video.addEventListener('error',bad,{once:true});video.currentTime=Math.min(time,Math.max(0,video.duration-.001))})}
function jpeg(canvas){return new Promise((resolve,reject)=>canvas.toBlob(v=>v?resolve(v):reject(new Error('could not encode frame')),'image/jpeg',.88))}
async function api(path,options={}){const response=await fetch(path,options);let value;try{value=await response.json()}catch{value={error:await response.text()}}if(!response.ok)throw new Error(value.error||`HTTP ${response.status}`);return value}
async function waitForJob(id){for(;;){const value=await api(`/api/jobs/${id}`);message(value.error||value.stage,value.state==='complete'?100:74);if(value.state==='complete')return value;if(value.state==='failed')throw new Error(value.error);await new Promise(r=>setTimeout(r,700))}}

processButton.onclick=async()=>{offlineBusy=true;cameraControls();try{
  if(!Number.isFinite(video.duration)||video.duration<=0)throw new Error('wait for the recorded video duration to become available');
  processButton.disabled=true;download.classList.add('disabled');playing=false;clearPreview();
  const fps=Math.max(1,Math.min(30,Number($('fps').value)||10)),frames=Math.max(1,Math.min(120,Math.floor(video.duration*fps)));
  const ratio=Math.min(1,960/Math.max(video.videoWidth,video.videoHeight)),width=Math.max(8,Math.round(video.videoWidth*ratio)),height=Math.max(8,Math.round(video.videoHeight*ratio));
  const capture=document.createElement('canvas');capture.width=width;capture.height=height;const captureContext=capture.getContext('2d',{alpha:false});
  job=await api('/api/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:file.files[0]?.name||'webcam recording',width,height,frames,fps})});
  video.pause();for(let i=0;i<frames;i++){await seek(i/fps);captureContext.drawImage(video,0,0,width,height);const blob=await jpeg(capture);await api(`/api/jobs/${job.id}/frames/${i}`,{method:'PUT',headers:{'Content-Type':'image/jpeg'},body:blob});message(`Collecting complete clip · ${i+1}/${frames}`,Math.round((i+1)/frames*65))}
  await api(`/api/jobs/${job.id}/process`,{method:'POST'});job=await waitForJob(job.id);download.href=`/api/jobs/${job.id}/motion.glb`;download.classList.remove('disabled');view.value='overlay';legend.hidden=false;playing=true;video.loop=true;video.currentTime=0;video.play().catch(()=>{});animate(job)
}catch(e){message(e.message,0)}finally{offlineBusy=false;cameraControls();processButton.disabled=false}};

function parsePose(buffer){
  const data=new DataView(buffer);let magic='';for(let i=0;i<8;i++)magic+=String.fromCharCode(data.getUint8(i));
  if(magic!=='GEMPOSE1'&&magic!=='GEMPOSE2')throw new Error('invalid pose result');
  let at=8;const vector=()=>{const values=[];for(let i=0;i<77;i++){values.push([data.getFloat32(at,true),data.getFloat32(at+4,true),data.getFloat32(at+8,true)]);at+=12}return values};
  const positions=vector(),cameraPositions=magic==='GEMPOSE2'?vector():null;at+=77*4*4+77*3*4;
  const parents=[];for(let i=0;i<77;i++){parents.push(data.getInt32(at,true));at+=4}
  const camera=[data.getFloat32(at,true),data.getFloat32(at+4,true),data.getFloat32(at+8,true)];at+=12;
  const keypoints=vector();if(at!==buffer.byteLength)throw new Error('invalid pose result size');
  return {positions,cameraPositions,parents,camera,keypoints};
}
function bones(points,parents,color,width,visible=()=>true){ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();for(let i=0;i<77;i++){const parent=parents[i];if(parent<0||parent>=77||!points[i]||!points[parent]||!visible(i)||!visible(parent))continue;ctx.moveTo(...points[i]);ctx.lineTo(...points[parent])}ctx.stroke()}
function joints(points,color,radius,visible=()=>true){ctx.fillStyle=color;for(let i=0;i<points.length;i++){if(!points[i]||!visible(i))continue;ctx.beginPath();ctx.arc(...points[i],radius,0,Math.PI*2);ctx.fill()}}
function drawGlobal(pose){const {positions,parents}=pose,W=canvas.width,H=canvas.height;if(viewYaw===null){const dx=positions[1][0]-positions[2][0]+positions[16][0]-positions[17][0],dz=positions[1][2]-positions[2][2]+positions[16][2]-positions[17][2];viewYaw=Math.atan2(dz,dx)+Math.PI/4}const projected=positions.map(p=>[Math.cos(viewYaw)*p[0]+Math.sin(viewYaw)*p[2],p[1]]);ctx.fillStyle='#0b1020';ctx.fillRect(0,0,W,H);let minX=Infinity,maxX=-Infinity,minY=Infinity,maxY=-Infinity;for(const p of projected){minX=Math.min(minX,p[0]);maxX=Math.max(maxX,p[0]);minY=Math.min(minY,p[1]);maxY=Math.max(maxY,p[1])}const scale=.78*Math.min(W/Math.max(.2,maxX-minX),H/Math.max(.2,maxY-minY)),cx=(minX+maxX)/2,cy=(minY+maxY)/2,points=projected.map(p=>[W/2+(p[0]-cx)*scale,H/2+(p[1]-cy)*scale]);bones(points,parents,'#5dd9f3',3);joints(points,'#ece9e2',3.3)}
function drawOverlay(pose){
  if(!pose.cameraPositions){drawGlobal(pose);return}
  const W=canvas.width,H=canvas.height,iw=job.width,ih=job.height,scale=Math.min(W/iw,H/ih),dw=iw*scale,dh=ih*scale,ox=(W-dw)/2,oy=(H-dh)/2;
  ctx.fillStyle='#0b1020';ctx.fillRect(0,0,W,H);if(liveDisplayed&&liveRunning)ctx.drawImage(liveDisplayed,ox,oy,dw,dh);else if(video.readyState>=2)ctx.drawImage(video,ox,oy,dw,dh);
  const focal=Math.max(iw,ih),map=p=>[ox+p[0]*scale,oy+p[1]*scale],predicted=pose.cameraPositions.map(p=>{const z=p[2]+pose.camera[2];return z>1e-5?map([(p[0]+pose.camera[0])/z*focal+iw/2,(p[1]+pose.camera[1])/z*focal+ih/2]):null}),observed=pose.keypoints.map(map),confident=i=>pose.keypoints[i][2]>.5;
  ctx.save();ctx.shadowColor='#0b1020';ctx.shadowBlur=3;bones(predicted,pose.parents,'#5dd9f3',Math.max(2,scale*2.5));joints(predicted,'#5dd9f3',Math.max(2.5,scale*3));bones(observed,pose.parents,'#ffac5b',Math.max(1.2,scale*1.6),confident);joints(observed,'#ffac5b',Math.max(2,scale*2.4),confident);ctx.restore();
}
function draw(pose){latestPose=pose;updatePreview();if(view.value==='overlay')drawOverlay(pose);else drawGlobal(pose)}
view.onchange=()=>{legend.hidden=view.value!=='overlay'||!latestPose;if(latestPose)draw(latestPose)};
async function animate(value){const cache=new Map();let previous=-1,fallback=0,last=performance.now();while(playing&&job?.id===value.id){const now=performance.now();let index;if(video.readyState>=2&&Number.isFinite(video.currentTime))index=Math.min(value.frames-1,Math.floor(video.currentTime*value.fps));else if(now-last>=1000/value.fps){last=now;index=fallback++%value.frames}else index=previous;if(index>=0&&index!==previous){previous=index;try{let pose=cache.get(index);if(!pose){const response=await fetch(`/api/jobs/${value.id}/poses/${String(index).padStart(6,'0')}`);if(!response.ok)throw new Error('pose unavailable');pose=parsePose(await response.arrayBuffer());cache.set(index,pose)}if(!playing||job?.id!==value.id)break;draw(pose)}catch(e){if(!playing||job?.id!==value.id)break;message(e.message);playing=false;clearPreview()}}await new Promise(requestAnimationFrame)}}

$('playback').onclick=()=>video.paused?video.play().catch(e=>message(e.message)):video.pause();
video.addEventListener('play',()=>{$('playback').textContent='Pause playback'});
video.addEventListener('pause',()=>{$('playback').textContent='Play'});
video.addEventListener('timeupdate',()=>{if(Number.isFinite(video.duration)){$('playhead').max=video.duration;$('playhead').value=video.currentTime}});
$('playhead').oninput=()=>{video.currentTime=Number($('playhead').value)};
