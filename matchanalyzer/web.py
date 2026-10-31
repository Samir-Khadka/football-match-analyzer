"""Football Match Analyzer — web UI (v0.8).

A coach uploads a single-camera match video, the pipeline runs in the
background, and they get the HTML coach report. Honest limits:
- CPU inference is SLOW (~5-10 s/frame with YOLO on CPU). A 90-min match
  takes hours. For now this is a demo/MVP; GPU or a hosted API is the
  real path (see README).
- Calibration assumes the pitch fills the frame (wide-angle footage).
  Click-to-calibrate is v0.9.
- One job at a time (MVP); uploads are deleted after processing.

Run: uvicorn matchanalyzer.web:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import shutil
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from .calibrate import PitchCalibrator
from .detect import YoloDetector
from .pipeline import Pipeline

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
JOBS: dict[str, dict] = {}
JOB_LOCK = threading.Lock()
MODEL_PATH = Path("models/yolov8n.pt")

app = FastAPI(title="Football Match Analyzer")


INDEX_HTML = """<!doctype html><html lang="en-GB"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Football Match Analyzer</title>
<style>
body{font-family:system-ui,sans-serif;max-width:640px;margin:2rem auto;
padding:0 1rem;background:#faf6ed;color:#2b2b2b}
h1{color:#4c1d95} .card{background:#fff;border:1px solid #e5dcc3;
border-radius:12px;padding:1.2rem;margin:1rem 0}
button{background:#4c1d95;color:#fff;border:0;border-radius:8px;
padding:.7rem 1.4rem;font-size:1rem;cursor:pointer}
#status{margin-top:1rem;font-variant-numeric:tabular-nums}
.note{color:#6b6b6b;font-size:.85rem}
</style></head><body>
<h1>⚽ Football Match Analyzer</h1>
<div class="card">
<p>Upload a single-camera match video (wide angle, pitch filling the frame).
You'll get possession, shots + xG, tackles, passes, set pieces, heatmaps,
a pass network, momentum chart and highlight clips.</p>
<input type="file" id="file" accept="video/*">
<button onclick="upload()">Analyze match</button>
<p style="margin:1rem 0">— or paste a link —</p>
<input type="url" id="url" placeholder="YouTube / video URL / live stream link"
 style="width:100%;padding:.6rem;border:1px solid #e5dcc3;border-radius:8px">
<button onclick="uploadUrl()" style="margin-top:.5rem">Analyze from link</button>
<div id="status"></div>
<p class="note">CPU processing is slow (minutes for a short clip, hours for a
full match). One job at a time. Your video is deleted after processing.</p>
</div>
<script>
async function upload(){
  const f=document.getElementById('file').files[0];
  if(!f){alert('Choose a video first');return;}
  const st=document.getElementById('status');
  st.textContent='Uploading…';
  const fd=new FormData(); fd.append('video',f);
  const r=await fetch('/upload',{method:'POST',body:fd});
  const {job_id}=await r.json();
  location.href='/calibrate/'+job_id;
}
async function uploadUrl(){
  const u=document.getElementById('url').value.trim();
  if(!u){alert('Paste a video or stream link first');return;}
  const st=document.getElementById('status');
  st.textContent='Resolving link… (downloads can take a while)';
  const r=await fetch('/from-url',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({url:u})});
  const j=await r.json();
  if(j.error){st.textContent='Error: '+j.error;return;}
  if(j.kind==='stream'){location.href='/live/'+j.job_id;}
  else{location.href='/calibrate/'+j.job_id;}
}
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


@app.post("/upload")
def upload_video(video: UploadFile = File(...)):
    job_id = uuid.uuid4().hex[:8]
    dest = UPLOAD_DIR / f"{job_id}_{Path(video.filename or 'match.mp4').name}"
    with dest.open("wb") as f:
        shutil.copyfileobj(video.file, f)
    # extract first frame for click-to-calibrate
    frame_path = UPLOAD_DIR / f"{job_id}_frame.jpg"
    dims = _extract_first_frame(dest, frame_path)
    with JOB_LOCK:
        JOBS[job_id] = {"status": "awaiting_calibration", "progress": 0.0,
                        "video_path": str(dest),
                        "frame_path": str(frame_path),
                        "frame_dims": dims}
    return {"job_id": job_id}


@app.post("/from-url")
def from_url(payload: dict):
    from . import source as source_mod
    url = (payload.get("url") or "").strip()
    if not url or not source_mod.is_url(url):
        return JSONResponse({"error": "not a valid http(s) URL"}, 400)
    job_id = uuid.uuid4().hex[:8]
    try:
        kind, location = source_mod.resolve(url, UPLOAD_DIR / job_id)
    except RuntimeError as e:
        return JSONResponse({"error": str(e)[:200]}, 400)
    if kind == "stream":
        with JOB_LOCK:
            JOBS[job_id] = {"status": "live", "progress": 0.0,
                            "stream_url": location,
                            "live_stats": {"frames": 0}}
        t = threading.Thread(target=_live_job, args=(job_id,), daemon=True)
        t.start()
        return {"job_id": job_id, "kind": "stream"}
    # downloaded file -> normal calibrate flow
    frame_path = UPLOAD_DIR / f"{job_id}_frame.jpg"
    dims = _extract_first_frame(Path(location), frame_path)
    with JOB_LOCK:
        JOBS[job_id] = {"status": "awaiting_calibration", "progress": 0.0,
                        "video_path": location,
                        "frame_path": str(frame_path),
                        "frame_dims": dims}
    return {"job_id": job_id, "kind": "file"}


def _extract_first_frame(video_path: Path, out_path: Path):
    """Save the first frame as JPEG; return (width, height)."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if ok:
        cv2.imwrite(str(out_path), frame)
    return (w, h)


CALIBRATE_HTML = """<!doctype html><html lang="en-GB"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Calibrate pitch — Football Match Analyzer</title>
<style>
body{font-family:system-ui,sans-serif;max-width:800px;margin:1rem auto;
padding:0 1rem;background:#faf6ed;color:#2b2b2b}
h1{color:#4c1d95;font-size:1.3rem}
canvas{max-width:100%;border:2px solid #4c1d95;border-radius:8px;cursor:crosshair}
button{background:#4c1d95;color:#fff;border:0;border-radius:8px;
padding:.7rem 1.4rem;font-size:1rem;cursor:pointer;margin-top:.8rem}
button:disabled{opacity:.4}
.note{color:#6b6b6b;font-size:.85rem}
#msg{margin-top:.5rem;font-weight:600}
</style></head><body>
<h1>Step 2 — mark the pitch corners</h1>
<p>Click the <b>four corners of the pitch</b> in this order:
top-left, top-right, bottom-right, bottom-left.</p>
<canvas id="c"></canvas>
<div id="msg"></div>
<button id="go" disabled onclick="submit()">Start analysis</button>
<p class="note">Or <a href="#" onclick="skip();return false">skip</a> to assume
the pitch fills the frame (less accurate).</p>
<script>
const jobId="{job_id}";
const img=new Image();
const pts=[];
img.onload=()=>{
  const c=document.getElementById('c');
  c.width=img.width; c.height=img.height;
  c.getContext('2d').drawImage(img,0,0);
};
img.src="/frame/"+jobId;
document.getElementById('c').onclick=(e)=>{
  if(pts.length>=4)return;
  const r=e.target.getBoundingClientRect();
  const x=(e.clientX-r.left)*(e.target.width/r.width);
  const y=(e.clientY-r.top)*(e.target.height/r.height);
  pts.push([Math.round(x),Math.round(y)]);
  const ctx=e.target.getContext('2d');
  ctx.fillStyle='#ef4444';
  ctx.beginPath();ctx.arc(x,y,8,0,7);ctx.fill();
  ctx.fillStyle='#fff';ctx.font='bold 14px sans-serif';
  ctx.fillText(pts.length,x-5,y+5);
  document.getElementById('msg').textContent=
    pts.length+'/4 corners marked'+(pts.length<4?'':' — ready!');
  document.getElementById('go').disabled=pts.length<4;
};
async function submit(){
  await fetch('/api/jobs/'+jobId+'/calibrate',{
    method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({corners:pts})});
  location.href='/progress/'+jobId;
}
async function skip(){
  await fetch('/api/jobs/'+jobId+'/calibrate',{
    method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({corners:null})});
  location.href='/progress/'+jobId;
}
</script></body></html>"""


@app.get("/calibrate/{job_id}", response_class=HTMLResponse)
def calibrate_page(job_id: str):
    return CALIBRATE_HTML.replace("{job_id}", job_id)


@app.get("/frame/{job_id}")
def get_frame(job_id: str):
    from fastapi.responses import FileResponse
    with JOB_LOCK:
        job = JOBS.get(job_id)
    if not job:
        return JSONResponse({"status": "not_found"}, status_code=404)
    return FileResponse(job["frame_path"], media_type="image/jpeg")


@app.post("/api/jobs/{job_id}/calibrate")
def set_calibration(job_id: str, payload: dict):
    with JOB_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return JSONResponse({"status": "not_found"}, status_code=404)
        job["corners"] = payload.get("corners")
        job["status"] = "queued"
    t = threading.Thread(target=_process_job, args=(job_id,), daemon=True)
    t.start()
    return {"ok": True}


PROGRESS_HTML = """<!doctype html><html><head><meta charset="utf-8">
<title>Processing…</title><style>
body{font-family:system-ui;max-width:480px;margin:3rem auto;text-align:center;
background:#faf6ed}h1{color:#4c1d95}#bar{height:18px;background:#e5dcc3;
border-radius:9px;overflow:hidden;margin:1rem 0}
#fill{height:100%;width:0;background:#4c1d95}</style></head><body>
<h1>Analyzing your match…</h1><div id="bar"><div id="fill"></div></div>
<div id="pct">0%</div>
<script>
const jobId="{job_id}";
const iv=setInterval(async()=>{
  const j=await (await fetch('/api/jobs/'+jobId)).json();
  if(j.status==='done'){clearInterval(iv);
    document.body.innerHTML='<h1>Done!</h1><a href="/report/'+jobId+
    '">View the full report</a>';}
  else if(j.status==='error'){clearInterval(iv);
    document.body.innerHTML='<h1>Error</h1><p>'+(j.error||'failed')+'</p>';}
  else{const p=Math.round((j.progress||0)*100);
    document.getElementById('fill').style.width=p+'%';
    document.getElementById('pct').textContent=p+'%';}
},2000);
</script></body></html>"""


@app.get("/progress/{job_id}", response_class=HTMLResponse)
def progress_page(job_id: str):
    return PROGRESS_HTML.replace("{job_id}", job_id)


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    with JOB_LOCK:
        job = JOBS.get(job_id)
    if not job:
        return JSONResponse({"status": "not_found"}, status_code=404)
    return job


@app.get("/report/{job_id}", response_class=HTMLResponse)
def get_report(job_id: str):
    with JOB_LOCK:
        job = JOBS.get(job_id)
    if not job or job["status"] != "done":
        return HTMLResponse("<p>Report not ready yet.</p>", status_code=404)
    return Path(job["report_path"]).read_text(encoding="utf-8")


def _set_progress(job_id: str, p: float, status: str = "processing"):
    with JOB_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(progress=p, status=status)


def _process_job(job_id: str):
    with JOB_LOCK:
        job = JOBS.get(job_id, {})
    video_path = Path(job["video_path"])
    try:
        import cv2
        _set_progress(job_id, 0.02)
        w, h = job.get("frame_dims") or (1280, 720)
        fps = 10.0
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 10.0)
            cap.release()

        cal = PitchCalibrator()
        corners = job.get("corners")
        if corners and len(corners) == 4:
            # user clicked: TL, TR, BR, BL in video pixels
            (x1, y1), (x2, y2), (x3, y3), (x4, y4) = corners
            cal.set_from_corners((x1, y1), (x2, y2), (x3, y3), (x4, y4))
        else:
            # skipped: pitch fills the frame (documented approximation)
            cal.set_from_corners((0, 0), (w, 0), (w, h), (0, h))

        det = YoloDetector(str(MODEL_PATH) if MODEL_PATH.exists()
                           else "yolov8n.pt")
        out_dir = UPLOAD_DIR / job_id
        out_dir.mkdir(exist_ok=True)

        def progress(p):
            _set_progress(job_id, 0.05 + 0.9 * p)

        summary = Pipeline(det, cal, fps=fps, progress_cb=progress).run(
            str(video_path),
            report_path=str(out_dir / "report.html"),
            clips_dir=str(out_dir / "clips"))
        _set_progress(job_id, 0.98)

        # report.html references clips relatively; serve from out_dir
        with JOB_LOCK:
            JOBS[job_id].update(status="done", progress=1.0,
                                report_path=str(out_dir / "report.html"),
                                summary={k: v for k, v in summary.items()
                                         if k in ("possession_pct", "n_shots",
                                                  "n_goals", "n_tackles")})
    except Exception as e:  # noqa: BLE001 — surfaced to the UI
        with JOB_LOCK:
            JOBS[job_id].update(status="error", error=str(e)[:300])
    finally:
        video_path.unlink(missing_ok=True)


LIVE_HTML = """<!doctype html><html lang="en-GB"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Live analysis</title>
<style>
body{font-family:system-ui,sans-serif;max-width:560px;margin:2rem auto;
padding:0 1rem;background:#0f172a;color:#e2e8f0}
h1{color:#f8fafc}.live{display:inline-block;background:#ef4444;color:#fff;
border-radius:6px;padding:.2rem .6rem;font-size:.8rem;font-weight:700;
animation:blink 1.2s infinite}
@keyframes blink{50%{opacity:.4}}
.card{background:#1e293b;border-radius:12px;padding:1.2rem;margin:1rem 0}
.big{font-size:2rem;font-weight:700}
.note{color:#94a3b8;font-size:.85rem}
</style></head><body>
<h1><span class="live">LIVE</span> Match analysis</h1>
<div class="card"><div class="note">Frames sampled</div>
<div class="big" id="frames">0</div></div>
<div class="card"><div class="note">Possession</div>
<div class="big" id="poss">–</div></div>
<div class="card">Shots <b id="shots">0</b> · Goals <b id="goals">0</b> ·
Tackles <b id="tackles">0</b></div>
<p class="note">Sampling ~1 frame/sec on CPU (YOLO is heavy). For true
real-time, run on a GPU. Stats update every 10 samples.</p>
<script>
const jobId="{job_id}";
setInterval(async()=>{
  const j=await (await fetch('/api/jobs/'+jobId)).json();
  const s=j.live_stats||{};
  document.getElementById('frames').textContent=s.frames||0;
  const p=s.possession_pct;
  document.getElementById('poss').textContent=p?
    `A ${p.A.toFixed(0)}% – B ${p.B.toFixed(0)}%`:'–';
  document.getElementById('shots').textContent=s.n_shots||0;
  document.getElementById('goals').textContent=s.n_goals||0;
  document.getElementById('tackles').textContent=s.n_tackles||0;
},3000);
</script></body></html>"""


@app.get("/live/{job_id}", response_class=HTMLResponse)
def live_page(job_id: str):
    return LIVE_HTML.replace("{job_id}", job_id)


def _live_job(job_id: str):
    """Live stream processing: samples ~1 fps on CPU, updates running
    stats every 10 samples. Honest limit: not true 25fps real-time."""
    import cv2
    import time
    with JOB_LOCK:
        stream_url = JOBS[job_id]["stream_url"]
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        with JOB_LOCK:
            JOBS[job_id].update(status="error",
                                error="could not open stream")
        return
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    cal = PitchCalibrator()
    cal.set_from_corners((0, 0), (w, 0), (w, h), (0, h))
    det = YoloDetector(str(MODEL_PATH) if MODEL_PATH.exists()
                       else "yolov8n.pt")
    from .track import Tracker
    from . import events as events_mod
    tracker = Tracker(fps=1)
    frames_data, n = [], 0
    last = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(1)
            continue
        if time.time() - last < 1.0:
            continue
        last = time.time()
        try:
            dets = det.detect(frame)
        except Exception:
            continue
        players, ball = tracker.update(dets)
        pitch_players = {}
        for p in players:
            try:
                pitch_players[p["id"]] = cal.px_to_pitch(float(p["cx"]),
                                                         float(p["cy"]))
            except Exception:
                continue
        bp = None
        if ball is not None:
            try:
                bp = cal.px_to_pitch(float(ball["cx"]), float(ball["cy"]))
            except Exception:
                pass
        frames_data.append({"t": float(n), "ball": bp,
                            "players": pitch_players})
        frames_data = frames_data[-300:]
        n += 1
        with JOB_LOCK:
            JOBS[job_id]["live_stats"]["frames"] = n
        if n % 10 == 0 and len(frames_data) >= 20:
            try:
                # live team split: no color data, so split by average x
                # (Team A = left half on average). Crude but functional.
                avg_x = {}
                for f in frames_data:
                    for pid, (px, py) in f["players"].items():
                        s = avg_x.setdefault(pid, [0.0, 0])
                        s[0] += px
                        s[1] += 1
                teams = {pid: ("A" if s[0] / s[1] < 0 else "B")
                         for pid, s in avg_x.items()}
                summ = events_mod.summarize(frames_data, teams, fps=1.0)
                with JOB_LOCK:
                    JOBS[job_id]["live_stats"].update({
                        "possession_pct": summ.get("possession_pct"),
                        "n_shots": summ.get("n_shots"),
                        "n_goals": summ.get("n_goals"),
                        "n_tackles": summ.get("n_tackles"),
                    })
            except Exception:
                pass
