import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from 'react';
import {
  Activity, AudioLines, Camera, Check, ChevronDown, CircleHelp, Compass, Crosshair,
  Eye, Gauge, Layers3, Mic, MicOff, Pause, Play, Radar, RefreshCw, Scan,
  Settings2, ShieldAlert, Volume2, VolumeX, Wifi, X
} from 'lucide-react';

type Detection = {
  id: string; class_id: number; label: string; confidence: number;
  bbox: [number, number, number, number]; distance_m: number | null;
  distance_quality?: string; risk?: string; direction?: string;
};
type ApiResult = { detections: Detection[]; inference_ms: number; image: {width:number;height:number}; distance_note?: string };

type ObjectStyle = CSSProperties & { '--object-color': string };

const API = (import.meta.env.VITE_API_URL || (import.meta.env.DEV ? 'http://localhost:8000' : '')).replace(/\/+$/, '');
const LOCAL_MODEL_URL = import.meta.env.VITE_LOCAL_MODEL_URL || '';
const COLORS = ['#44D7B6','#55A7FF','#B69CFF','#FFB45E','#FF718B','#70D6FF'];
const pretty = (s: string) => s.replace(/[_-]/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
function errorMessage(error: unknown, fallback: string) {
  return error instanceof Error ? error.message : fallback;
}
function directionFor(b: Detection) {
  const center = (b.bbox[0] + b.bbox[2]) / 2;
  return center < .36 ? 'Left' : center > .64 ? 'Right' : 'Ahead';
}
function riskFor(d: Detection) {
  const label = d.label.toLowerCase();
  if (/stairs|staircase|pothole|hole|curb|step|barrier/.test(label)) return 'Caution';
  if (d.distance_m !== null && d.distance_m < 1.1) return 'Near';
  return 'Detected';
}
function distanceText(d: Detection) {
  return typeof d.distance_m === 'number' && Number.isFinite(d.distance_m) ? `${d.distance_m.toFixed(1)} m` : 'Distance unknown';
}

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const busyRef = useRef(false);
  const lastSpokenRef = useRef<Map<string, number>>(new Map());
  const lastResultRef = useRef<Detection[]>([]);
  const [cameraOn, setCameraOn] = useState(false);
  const [cameraError, setCameraError] = useState('');
  const [detections, setDetections] = useState<Detection[]>([]);
  const [apiMs, setApiMs] = useState<number | null>(null);
  const [fps, setFps] = useState(0);
  const [voiceOn, setVoiceOn] = useState(true);
  const [paused, setPaused] = useState(false);
  const [mode, setMode] = useState<'fast'|'balanced'>('fast');
  const [activeTab, setActiveTab] = useState<'live'|'objects'|'map'>('live');
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [sensitivity, setSensitivity] = useState(.25);
  const [status, setStatus] = useState<'idle'|'starting'|'live'|'error'>('idle');
  const [lastUpdate, setLastUpdate] = useState('');
  const [localModelState, setLocalModelState] = useState<'off'|'loading'|'ready'|'failed'>('off');
  const [localModelError, setLocalModelError] = useState('');

  const selected = detections.find(d => d.id === selectedId) || detections[0] || null;
  const nearest = useMemo(() => detections.reduce<Detection|null>((a,d) => {
    if (d.distance_m == null) return a;
    return !a || (a.distance_m ?? Infinity) > d.distance_m ? d : a;
  }, null), [detections]);

  const speak = useCallback((text: string, key: string) => {
    if (!voiceOn || !('speechSynthesis' in window)) return;
    const now = Date.now(), prev = lastSpokenRef.current.get(key) || 0;
    if (now - prev < 5000) return;
    lastSpokenRef.current.set(key, now);
    // Keep alerts short. Do not queue a backlog from stale frames.
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1.05; utterance.pitch = 1; utterance.volume = 1;
    window.speechSynthesis.speak(utterance);
  }, [voiceOn]);

  const startCamera = useCallback(async () => {
    setCameraError(''); setStatus('starting');
    let stream: MediaStream | null = null;
    try {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error('Camera access requires HTTPS or localhost in a supported browser.');
      stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30, max: 30 } }
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setCameraOn(true); setStatus('live');
    } catch (error: unknown) {
      stream?.getTracks().forEach(track => track.stop());
      streamRef.current = null;
      setStatus('error'); setCameraError(errorMessage(error, 'Could not start camera.'));
    }
  }, []);

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach(t => t.stop());
    streamRef.current = null;
    if (videoRef.current) { videoRef.current.pause(); videoRef.current.srcObject = null; }
    window.speechSynthesis?.cancel();
    lastResultRef.current = [];
    setDetections([]);
    setCameraError('');
    setCameraOn(false); setStatus('idle'); setPaused(false);
  }, []);

  const drawBoxes = useCallback((items: Detection[]) => {
    const canvas = canvasRef.current, video = videoRef.current;
    if (!canvas || !video || !video.videoWidth || !video.videoHeight) return;
    canvas.width = video.clientWidth * devicePixelRatio;
    canvas.height = video.clientHeight * devicePixelRatio;
    const ctx = canvas.getContext('2d'); if (!ctx) return;
    ctx.scale(devicePixelRatio, devicePixelRatio);
    const w = video.clientWidth, h = video.clientHeight;
    const scale = Math.max(w / video.videoWidth, h / video.videoHeight);
    const offsetX = (w - video.videoWidth * scale) / 2;
    const offsetY = (h - video.videoHeight * scale) / 2;
    items.forEach((d,i) => {
      const [x1,y1,x2,y2] = d.bbox;
      const x = x1*video.videoWidth*scale + offsetX;
      const y = y1*video.videoHeight*scale + offsetY;
      const bw = (x2-x1)*video.videoWidth*scale, bh = (y2-y1)*video.videoHeight*scale;
      const color = COLORS[i % COLORS.length];
      ctx.strokeStyle = color; ctx.lineWidth = d.id === selectedId ? 3 : 2;
      ctx.shadowColor = color; ctx.shadowBlur = 8; ctx.strokeRect(x,y,bw,bh); ctx.shadowBlur = 0;
      const label = `${pretty(d.label)}  ${distanceText(d)}`;
      ctx.font = '600 12px Inter, system-ui, sans-serif';
      const tw = ctx.measureText(label).width;
      const tagW = Math.min(w - x, tw + 18), tagH = 26, tagY = Math.max(4, y - tagH);
      ctx.fillStyle = color; ctx.beginPath();
      const r = 7; ctx.roundRect(x, tagY, tagW, tagH, r); ctx.fill();
      ctx.fillStyle = '#06111d'; ctx.fillText(label, x+9, tagY+17, tagW-14);
    });
  }, [selectedId]);

  const infer = useCallback(async () => {
    if (!cameraOn || paused || busyRef.current) return;
    const video = videoRef.current;
    if (!video || video.readyState < 2 || !video.videoWidth) return;
    const canvas = document.createElement('canvas');
    const maxW = 960, scale = Math.min(1, maxW/video.videoWidth);
    canvas.width = Math.round(video.videoWidth*scale); canvas.height = Math.round(video.videoHeight*scale);
    const c = canvas.getContext('2d'); if (!c) return;
    c.drawImage(video, 0, 0, canvas.width, canvas.height);
    busyRef.current = true;
    const started = performance.now();
    try {
      const blob = await new Promise<Blob>((resolve,reject) => canvas.toBlob(b => b ? resolve(b) : reject(new Error('Frame encoding failed')), 'image/jpeg', .78));
      const form = new FormData(); form.append('image', blob, 'frame.jpg');
      form.append('confidence', String(sensitivity));
      if (!API) throw new Error('Detection backend is not configured. Set VITE_API_URL in Vercel to your deployed backend URL, then redeploy.');
      const res = await fetch(`${API}/api/detect`, { method: 'POST', body: form });
      if (!res.ok) {
        const body: { detail?: unknown } = await res.json();
        const detail = typeof body.detail === 'string' ? body.detail : `Detection backend returned ${res.status}`;
        throw new Error(detail);
      }
      const data: ApiResult = await res.json();
      const items = (data.detections || []).filter(d => d.confidence >= sensitivity).slice(0, 40)
        .map((d,i) => ({...d, id: `${d.class_id}-${Math.round(d.bbox[0]*100)}-${Math.round(d.bbox[1]*100)}-${i}`, direction: directionFor(d), risk: riskFor(d)}));
      lastResultRef.current = items;
      setDetections(items); setApiMs(data.inference_ms ?? performance.now()-started);
      setCameraError('');
      setLastUpdate(new Date().toLocaleTimeString());
      setFps(v => Math.min(60, Math.round(v*.7 + (1000/(performance.now()-started))*.3)));
      if (voiceOn) {
        const urgent = items.find(d => d.distance_m != null && d.distance_m < 1.1);
        if (urgent) speak(`${pretty(urgent.label)} nearby, distance estimate ${distanceText(urgent)}. ${urgent.direction}.`, urgent.id);
      }
    } catch (error: unknown) {
      // A temporary inference/network failure should not crash or kill the camera loop.
      setCameraError(errorMessage(error, 'Inference unavailable. Check that the backend is running.'));
    } finally {
      busyRef.current = false;
      drawBoxes(lastResultRef.current);
    }
  }, [cameraOn, paused, sensitivity, voiceOn, speak, drawBoxes]);

  useEffect(() => {
    if (!cameraOn || paused) return;
    const interval = window.setInterval(() => { void infer(); }, mode === 'balanced' ? 250 : 120);
    return () => window.clearInterval(interval);
  }, [cameraOn, paused, infer, mode]);

  useEffect(() => {
    drawBoxes(detections);
  }, [detections, drawBoxes]);

  useEffect(() => () => {
    streamRef.current?.getTracks().forEach(t => t.stop());
    window.speechSynthesis?.cancel();
  }, []);

  const enableLocalModel = async () => {
    if (!LOCAL_MODEL_URL) return;
    setLocalModelState('loading');
    setLocalModelError('');
    try {
      const ort = await import('onnxruntime-web');
      const session = await ort.InferenceSession.create(LOCAL_MODEL_URL, { executionProviders: ['webgpu', 'wasm'] });
      // Session is initialized to verify that the optional browser model is loadable.
      // Plug the exact model-specific preprocessing/decoder into this session before local inference.
      window.__astrixOrtSession = session;
      setLocalModelState('ready');
    } catch (error: unknown) {
      setLocalModelState('failed');
      setLocalModelError(errorMessage(error, 'Could not load the browser model.'));
    }
  };

  const riskCount = detections.filter(d => d.risk === 'Near' || d.risk === 'Caution').length;

  return <div className="app-shell">
    <aside className="sidebar">
      <div className="brand-lockup">
        <img src="/astrix-logo.jpeg" className="brand-logo" alt="ASTRIX logo" onError={e => (e.currentTarget.style.display='none')} />
        <div><div className="brand-name">ASTRIX</div><div className="brand-tag">SEE · UNDERSTAND · MOVE SAFELY</div></div>
      </div>
      <div className="side-label">WORKSPACE</div>
      <button className={`nav-item ${activeTab==='live'?'active':''}`} onClick={()=>setActiveTab('live')}><Scan size={18}/> Live detection <span className="nav-live"/></button>
      <button className={`nav-item ${activeTab==='objects'?'active':''}`} onClick={()=>setActiveTab('objects')}><Layers3 size={18}/> Detected objects <span className="nav-count">{detections.length}</span></button>
      <button className={`nav-item ${activeTab==='map'?'active':''}`} onClick={()=>setActiveTab('map')}><Radar size={18}/> Surrounding view</button>
      <div className="side-label side-label-spaced">SYSTEM</div>
      <button className="nav-item" onClick={()=>setSettingsOpen(true)}><Settings2 size={18}/> Preferences</button>
      <div className="sidebar-bottom">
        <div className="connection-card"><div className="connection-dot"/><div><b>{cameraOn ? 'Vision system active' : 'Ready when you are'}</b><span>{cameraOn ? 'Camera session running' : 'Camera is not transmitting'}</span></div></div>
        <div className="privacy-note"><ShieldAlert size={15}/> Frames are sent to your configured backend for inference.</div>
        <div className="version">ASTRIX WEB · PREVIEW 0.1</div>
      </div>
    </aside>

    <main className="main">
      <header className="topbar">
        <div><div className="eyebrow">AI-POWERED ASSISTIVE VISION</div><h1>{activeTab==='live'?'Live environment':activeTab==='objects'?'Detected objects':'Surrounding view'}</h1><p>Real-time awareness, designed around you.</p></div>
        <div className="top-actions"><div className={`status-pill ${cameraOn?'is-live':''}`}><span/>{cameraOn?'CAMERA LIVE':'SYSTEM STANDBY'}</div><button className="icon-button" title="Settings" onClick={()=>setSettingsOpen(true)}><Settings2 size={18}/></button></div>
      </header>

      {activeTab==='live' && <section className="workspace-grid">
        <div className="vision-panel panel">
          <div className="panel-heading"><div><span className="tiny-label">LIVE ENVIRONMENT</span><h2>Vision feed</h2></div><div className="feed-actions"><span className="feed-chip"><Activity size={13}/>{cameraOn?'Tracking active':'Awaiting camera'}</span><button className="mini-icon" onClick={()=>drawBoxes(detections)} title="Refresh overlays"><RefreshCw size={15}/></button></div></div>
          <div className={`camera-stage ${cameraOn?'camera-active':''}`}>
            <video ref={videoRef} playsInline muted className={cameraOn?'video-visible':''}/>
            <canvas ref={canvasRef} className="overlay-canvas"/>
            {!cameraOn && <div className="camera-empty"><div className="camera-orbit"><div className="orbit-ring"/><Camera size={34}/></div><h3>Your world, in focus.</h3><p>Start the camera to detect objects, estimate distance<br/>and hear important warnings.</p><button className="primary-button" onClick={startCamera}><Camera size={17}/> Start live detection</button>{cameraError && <div className="error-message">{cameraError}</div>}</div>}
            {cameraOn && <div className="camera-hud"><div className="hud-pill"><span className="live-dot"/> LIVE · {fps || '—'} FPS</div><div className="hud-corner"><Crosshair size={19}/></div><div className="camera-bottom-hud"><span><Wifi size={14}/> SERVER INFERENCE · {mode==='fast'?'FAST':'BALANCED'}</span><span>{apiMs===null?'--':`${Math.round(apiMs)} ms`} inference</span></div></div>}
            {cameraOn && cameraError && <div className="camera-inference-error" role="alert">{cameraError}</div>}
          </div>
          <div className="vision-controls">
            <div className="control-left"><button className={`control-button ${voiceOn?'selected':''}`} onClick={()=>setVoiceOn(v=>!v)}>{voiceOn?<Volume2 size={16}/>:<VolumeX size={16}/>} Voice alerts <span className={`toggle ${voiceOn?'on':''}`}/></button><button className="control-button" onClick={()=>setMode(m=>m==='fast'?'balanced':'fast')}><RefreshCw size={15}/>{mode==='fast'?'Fast inference':'Balanced inference'}<ChevronDown size={14}/></button></div>
            {cameraOn && <div className="camera-action-group"><button className="primary-button compact" onClick={()=>setPaused(p=>!p)}>{paused?<Play size={16}/>:<Pause size={16}/>} {paused?'Resume':'Pause'}</button><button className="icon-button stop-camera" onClick={stopCamera} title="Stop camera"><X size={15}/></button></div>}
          </div>
        </div>
        <div className="insight-column">
          <div className="metric-card nearest-card"><div className="metric-top"><span>NEAREST MEASURED OBJECT</span><Compass size={16}/></div><div className="metric-value">{nearest ? distanceText(nearest) : '— —'}</div><div className="metric-caption">{nearest ? `${pretty(nearest.label)} · ${nearest.direction}` : 'Distance unavailable until calibrated'}</div><div className="metric-footer"><span className="metric-dot mint"/>{nearest?'Estimated distance':'No metric distance'}</div></div>
          <div className="metric-card scene-card"><div className="metric-top"><span>SCENE SUMMARY</span><Eye size={16}/></div><div className="metric-value">{detections.length.toString().padStart(2,'0')}</div><div className="metric-caption">Objects in current frame</div><div className="scene-bars"><span style={{width:`${Math.min(100,detections.length*12)}%`}}/></div><div className="metric-footer"><span className="metric-dot blue"/>{riskCount} items flagged for attention</div></div>
          <div className="objects-panel panel"><div className="panel-heading compact-heading"><div><span className="tiny-label">LIVE TRACKING</span><h2>Object list</h2></div><span className="count-badge">{detections.length} objects</span></div>
            <div className="object-list">{detections.length===0?<div className="empty-list"><Radar size={25}/><b>{cameraOn?'No objects detected':'No objects yet'}</b><span>{cameraOn?'Try lowering detection sensitivity or check whether the object is supported by this model.':'Start the camera to populate this list.'}</span></div>:detections.slice(0,8).map((d,i)=><button key={d.id} className={`object-row ${selectedId===d.id?'object-selected':''}`} onClick={()=>setSelectedId(d.id)}><div className="object-symbol" style={{'--object-color':COLORS[i%COLORS.length]} as ObjectStyle}><Crosshair size={19}/></div><div className="object-main"><b>{pretty(d.label)}</b><span>{d.direction} · {Math.round(d.confidence*100)}% confidence</span><em className={`risk-tag ${d.risk==='Near'?'risk-high':d.risk==='Caution'?'risk-medium':'risk-low'}`}>{d.risk}</em></div><div className="object-distance">{distanceText(d)}<span>{d.distance_m==null?'uncalibrated':'estimated'}</span></div></button>)}</div>
            <button className="text-button" onClick={()=>setActiveTab('objects')}>View all detections <span>↗</span></button>
          </div>
          <div className="voice-card"><div className="voice-icon"><AudioLines size={19}/></div><div><b>Voice guidance</b><span>{voiceOn?'Ready for urgent obstacle alerts':'Voice alerts are muted'}</span></div><button className="mini-icon" onClick={()=>speak('ASTRIX voice guidance is ready.', 'manual-test')} title="Test voice"><Play size={15}/></button></div>
        </div>
      </section>}

      {activeTab==='objects' && <section className="panel full-panel"><div className="panel-heading"><div><span className="tiny-label">FRAME ANALYSIS</span><h2>All detected objects</h2></div><span className="count-badge">{detections.length} tracked</span></div>{detections.length===0?<div className="empty-list roomy"><Radar size={32}/><b>Nothing detected yet</b><span>Start the live camera to inspect your surroundings.</span><button className="primary-button" onClick={()=>setActiveTab('live')}>Go to live view</button></div>:<div className="all-objects">{detections.map((d,i)=><div className="detail-row" key={d.id}><span className="object-index" style={{color:COLORS[i%COLORS.length]}}>0{i+1}</span><div><b>{pretty(d.label)}</b><span>{d.direction} · {Math.round(d.confidence*100)}% confidence · {d.risk}</span></div><strong>{distanceText(d)}</strong><button className="mini-icon" onClick={()=>speak(`${pretty(d.label)}. ${distanceText(d)}. ${d.direction}.`,'detail-'+d.id)}><Volume2 size={16}/></button></div>)}</div>}</section>}

      {activeTab==='map' && <section className="panel full-panel map-panel"><div className="panel-heading"><div><span className="tiny-label">SPATIAL OVERVIEW</span><h2>Surrounding view</h2></div><span className="count-badge">{detections.length} objects</span></div><div className="radar-view"><div className="radar-grid"/><div className="radar-center"><div className="radar-arrow">▲</div><span>YOU</span></div>{detections.map((d,i)=>{const x=(d.bbox[0]+d.bbox[2])/2;const y=(d.bbox[1]+d.bbox[3])/2;return <div key={d.id} className="radar-object" style={{left:`${Math.max(8,Math.min(92,x*100))}%`,top:`${Math.max(8,Math.min(92,y*100))}%`,'--object-color':COLORS[i%COLORS.length]} as ObjectStyle} title={`${pretty(d.label)} · ${distanceText(d)}`}><span/>{pretty(d.label)}<small>{distanceText(d)}</small></div>})}</div><p className="map-note"><CircleHelp size={15}/> This is a directional visualization inferred from image position, not a calibrated 360° map. A single forward-facing camera cannot see behind you.</p></section>}

      <footer className="footer"><span>© ASTRIX · Assistive vision technology</span><span><span className="footer-dot"/> {status==='live'?'Systems operational':status==='error'?'Needs attention':'Standing by'} · Last update {lastUpdate||'—'}</span></footer>
    </main>

    {settingsOpen && <div className="modal-backdrop" onClick={()=>setSettingsOpen(false)}><div className="settings-modal" onClick={e=>e.stopPropagation()}><div className="modal-header"><div><span className="tiny-label">CONFIGURATION</span><h2>Vision preferences</h2></div><button className="icon-button" onClick={()=>setSettingsOpen(false)}><X size={18}/></button></div>
      <label className="setting-label">Detection sensitivity <strong>{Math.round(sensitivity*100)}%</strong></label><input type="range" min=".15" max=".8" step=".05" value={sensitivity} onChange={e=>setSensitivity(Number(e.target.value))}/><p className="setting-help">Lower values detect more objects but may increase false positives. Detection may still be limited by the model's classes.</p>
      <div className="setting-line"><div><b>Voice warnings</b><span>Spoken obstacle alerts</span></div><button className={`switch ${voiceOn?'switch-on':''}`} onClick={()=>setVoiceOn(v=>!v)}><span/></button></div>
      <div className="setting-line"><div><b>Optional browser model</b><span>{localModelState==='ready'?'Model file loaded; live detection uses the server':localModelState==='loading'?'Loading model':localModelState==='failed'?'Browser model failed to load':LOCAL_MODEL_URL?'Available for browser loading':'Not configured; live detection uses the server'}</span></div><button className="secondary-button" onClick={enableLocalModel} disabled={!LOCAL_MODEL_URL || localModelState==='loading'}>{localModelState==='loading'?'Loading…':LOCAL_MODEL_URL?'Load model':'Not configured'}</button></div>
      {localModelError && <p className="setting-help" role="alert">{localModelError}</p>}
      <div className="setting-warning"><ShieldAlert size={16}/><span>Distances are shown as unknown until a calibrated metric-depth pipeline is connected. Do not use this prototype as your only mobility aid.</span></div>
      <button className="primary-button modal-done" onClick={()=>setSettingsOpen(false)}><Check size={16}/> Save preferences</button>
    </div></div>}
  </div>;
}
