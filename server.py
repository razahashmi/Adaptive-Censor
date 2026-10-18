"""Loopback-only application server. Run: python server.py --port 8502."""
from contextlib import asynccontextmanager
from pathlib import Path
import argparse
import asyncio
import json
import math
import secrets
import shutil
import threading
import time
import uuid

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field

from analysis_engine import Engine, ScanConfig, CHUNK, normalize_media
from core import make_demo, compile_plan, render_video, validate_events
from model_setup import prepare_environment
from live import LiveCapture
import labeling
from scene_verifier import installed as verifier_installed
from learned_runtime import available_heads, runtime_ready
from provenance import file_hash

ROOT = prepare_environment()
SESSIONS = ROOT / "sessions-v2"
SESSIONS.mkdir(exist_ok=True)
TOKEN = secrets.token_urlsafe(32)
engine = Engine()
exports = {}
captures = {}
export_lock = threading.Lock()
upload_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app):
    yield
    for capture in captures.values():
        capture.stop()
    feedback_clips.close()
    engine.close()


app = FastAPI(title="Adaptive Censor", docs_url=None, redoc_url=None, lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])


@app.middleware("http")
async def local_security(request: Request, call_next):
    if request.url.path.startswith("/api/") and request.url.path != "/api/bootstrap":
        if not secrets.compare_digest(request.cookies.get("censor_session", ""), TOKEN):
            return JSONResponse({"detail": "Open the app on this server before using its API"}, status_code=403)
        if request.method not in ("GET", "HEAD"):
            if not secrets.compare_digest(request.headers.get("x-censor-token", ""), TOKEN):
                return JSONResponse({"detail": "Missing application token"}, status_code=403)
            origin = request.headers.get("origin")
            if origin and origin != str(request.base_url).rstrip("/"):
                return JSONResponse({"detail": "Cross-origin writes are disabled"}, status_code=403)
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
    response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; media-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
    response.headers["Cache-Control"] = "no-store"
    return response


def get_job(job_id):
    job = engine.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Media session not found")
    return job


def snapshot(job):
    with engine.condition:
        return job.snapshot(engine.models)


def available_roi_models():
    from roi_gate import validate_head
    expected=file_hash(ROOT/'clip/onnx/vision_model_quantized.onnx')
    try:
        selections=labeling.read_json(ROOT/'roi-models'/'review-options.json',{})
    except (OSError,ValueError):
        return []
    if not isinstance(selections,dict):
        return []
    names=[]
    for path in sorted((ROOT/'roi-models').glob('*.json')):
        try:
            if not path.stem or len(path.stem)>64 or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in path.stem):
                continue
            data=json.loads(path.read_text(encoding='utf-8'))
            if selections.get(path.stem)!=file_hash(path):
                continue
            if not isinstance(data,dict):continue
            validate_head(data)
            if data.get('anatomy_model')=='640m' and data.get('anatomy_identity')!=file_hash(ROOT/'scene-verifier/640m.onnx'):
                continue
            if isinstance(data,dict) and expected and data.get('clip_identity')==expected and all(k in data for k in ('mean','scale','coef','intercept','decision_threshold')):
                names.append(path.stem)
        except (OSError,ValueError,TypeError,KeyError):
            continue
    return names


@app.get("/api/bootstrap")
def bootstrap():
    roi_names=available_roi_models()
    response = JSONResponse(dict(token=TOKEN, version="0.3.0", sessions=[{"id": j.id, "title": j.title} for j in engine.jobs.values() if not j.live],
                                 verifier_ready=verifier_installed(ROOT) or verifier_installed(ROOT,'8B'),
                                 verifier_4b_ready=verifier_installed(ROOT),
                                 verifier_8b_ready=verifier_installed(ROOT,'8B'),
                                 anatomy_640_ready=(ROOT/'scene-verifier/640m.onnx').is_file(),
                                 learned_heads=available_heads(ROOT) if runtime_ready(ROOT) else [],
                                 anatomy_roi_models=roi_names,
                                 anatomy_roi_requirements={name:labeling.read_json(ROOT/'roi-models'/f'{name}.json').get('anatomy_model','') for name in roi_names},
                                 models_ready=(ROOT / "clip/onnx/vision_model_quantized.onnx").exists()))
    response.set_cookie("censor_session", TOKEN, httponly=True, samesite="strict")
    return response


@app.get('/api/readiness')
def readiness():
    from readiness import inspect
    return inspect(ROOT)


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    if upload_lock.locked():
        raise HTTPException(409, "Another upload is being prepared")
    async with upload_lock:
        folder = SESSIONS / uuid.uuid4().hex
        folder.mkdir()
        source = folder / "input.bin"
        try:
            size = 0
            with source.open("wb") as stream:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > 8 * 1024**3:
                        raise HTTPException(413, "Maximum upload size is 8 GiB")
                    stream.write(chunk)
            info = await asyncio.to_thread(normalize_media, source, folder / "source.mp4")
            if info["duration"] > 8 * 3600:
                raise ValueError("Maximum duration is 8 hours")
            source.unlink()
            job = engine.add(folder / "source.mp4", (file.filename or "Local media")[:200])
            return snapshot(job)
        except Exception as error:
            shutil.rmtree(folder, ignore_errors=True)
            if isinstance(error, HTTPException):
                raise
            raise HTTPException(422, str(error)[-700:]) from error
        finally:
            await file.close()


@app.post("/api/demo")
def demo():
    folder = SESSIONS / uuid.uuid4().hex
    folder.mkdir()
    events = make_demo(folder / "source.mp4")
    job = engine.add(folder / "source.mp4", "Scripted controls demo · not an accuracy test")
    job.config = ScanConfig(speech=False, nudity=False, scenes=False)
    for index in range(math.ceil(job.duration / CHUNK)):
        job.chunks[index] = []
    job.manual = [dict(e, id=uuid.uuid4().hex[:16]) for e in events]
    job.status = "complete"
    job.updated += 1
    for index in job.chunks:
        engine.persist(job, index)
    return snapshot(job)


@app.get("/api/media/{job_id}")
def media(job_id: str, since: int = -1):
    job = get_job(job_id)
    job.last_seen = time.monotonic()
    if since == job.updated:
        return dict(unchanged=True, status=job.status, error=job.error, generation=job.generation)
    return snapshot(job)


@app.get("/api/media/{job_id}/source")
def source(job_id: str):
    job = get_job(job_id)
    if job.live:
        raise HTTPException(400, "Live media uses the segmented playback endpoint")
    return FileResponse(job.path, media_type="video/mp4")


class LiveRequest(BaseModel):
    url: str = Field(max_length=4000)
    config: ScanConfig = Field(default_factory=ScanConfig)


@app.post("/api/live")
def start_live(request: LiveRequest):
    if any(capture.job.capture_running for capture in captures.values()):
        raise HTTPException(409, "Stop the existing live input first")
    try:
        capture = LiveCapture(engine, SESSIONS / uuid.uuid4().hex, request.url, request.config)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    captures[capture.job.id] = capture
    return snapshot(capture.job)


@app.post("/api/media/{job_id}/stop-live")
def stop_live(job_id: str):
    capture = captures.get(job_id)
    if not capture:
        raise HTTPException(404)
    capture.stop()
    return snapshot(capture.job)


@app.get("/api/media/{job_id}/manifest")
def live_manifest(job_id: str):
    if job_id not in captures:
        raise HTTPException(404)
    return captures[job_id].manifest()


@app.get("/api/media/{job_id}/fragment/{part}")
def live_fragment(job_id: str, part: str):
    job = get_job(job_id)
    if not job.live:
        raise HTTPException(404)
    if part == "init":
        path = job.path.parent / "init.mp4"
    elif part.isdecimal() and int(part) in job.chunks and int(part) in job.segments:
        path = job.segments[int(part)]
    else:
        raise HTTPException(404, "Fragment is not analyzed or has expired")
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="video/mp4")


@app.post("/api/media/{job_id}/configure")
def configure(job_id: str, config: ScanConfig):
    job = get_job(job_id)
    engine.configure(job, config)
    return snapshot(job)


class Cursor(BaseModel):
    position: float = Field(ge=0, allow_inf_nan=False)
    active: bool = True
    full_scan: bool = False


@app.post("/api/media/{job_id}/cursor")
def cursor(job_id: str, cursor: Cursor):
    job = get_job(job_id)
    with engine.condition:
        job.position = min(cursor.position, max(0, job.duration - .001))
        job.active = cursor.active
        job.complete_scan = cursor.full_scan
        job.last_seen = time.monotonic()
        engine.condition.notify_all()
    return {"ok": True}


class ManualEvent(BaseModel):
    start: float
    end: float
    category: str
    confidence: float = 1
    note: str = Field(default="Manual correction", max_length=300)
    region: list[float] | None = Field(default=None, min_length=4, max_length=4)


@app.post("/api/media/{job_id}/events")
def add_event(job_id: str, event: ManualEvent):
    job = get_job(job_id)
    try:
        validated = validate_events([event.model_dump()], job.duration)[0]
    except (ValueError, TypeError) as error:
        raise HTTPException(422, str(error)) from error
    validated.update(id=uuid.uuid4().hex[:16], source="manual")
    with engine.condition:
        job.manual.append(validated)
        job.updated += 1
        engine.persist(job)
    return snapshot(job)


class Review(BaseModel):
    id: str
    enabled: bool
    confirmed: bool | None = None


@app.post("/api/media/{job_id}/review")
def review(job_id: str, review: Review):
    job = get_job(job_id)
    with engine.condition:
        if not any(e['id'] == review.id for e in job.snapshot(engine.models)['events']):
            raise HTTPException(404, 'Detection not found')
        if review.enabled:
            job.ignored.discard(review.id)
        else:
            job.ignored.add(review.id)
        if review.confirmed is True:
            job.confirmed.add(review.id)
        elif review.confirmed is False:
            job.confirmed.discard(review.id)
        job.updated += 1
        engine.persist(job)
    return snapshot(job)


@app.get("/api/media/{job_id}/timeline")
def timeline(job_id: str):
    return JSONResponse(snapshot(get_job(job_id)), headers={"Content-Disposition": 'attachment; filename="detection-timeline.json"'})


@app.post("/api/media/{job_id}/export")
def export(job_id: str, policy: dict):
    job = get_job(job_id)
    if job.live:
        raise HTTPException(400, "Filtered export currently supports finite files only")
    state = snapshot(job)
    if state["analyzed_seconds"] < job.duration - .01:
        raise HTTPException(409, "Analyze the full file before exporting")
    try:
        plan = compile_plan(state["events"], policy, job.duration)
    except (ValueError, TypeError, KeyError) as error:
        raise HTTPException(422, str(error)) from error
    if not export_lock.acquire(blocking=False):
        raise HTTPException(409, "An export is already running")
    export_id = uuid.uuid4().hex
    exports[export_id] = {"status": "rendering", "path": job.path.parent / f"{export_id}.mp4"}
    def run():
        try:
            render_video(job.path, exports[export_id]["path"], plan)
            exports[export_id]["status"] = "complete"
        except Exception as error:
            exports[export_id].update(status="error", error=str(error)[-500:])
        finally:
            export_lock.release()
    threading.Thread(target=run, name="export-worker", daemon=True).start()
    return {"id": export_id, "status": "rendering"}


@app.get("/api/exports/{export_id}")
def export_status(export_id: str):
    if export_id not in exports:
        raise HTTPException(404)
    return {k: v for k, v in exports[export_id].items() if k != "path"}


@app.get("/api/exports/{export_id}/download")
def download_export(export_id: str):
    item = exports.get(export_id)
    if not item or item["status"] != "complete":
        raise HTTPException(404, "Export is not ready")
    return FileResponse(item["path"], media_type="video/mp4", filename="filtered.mp4")


@app.delete("/api/media/{job_id}")
def delete_media(job_id: str):
    job = get_job(job_id)
    # Retention takes these locks in the same order when resolving a source.
    with feedback_clips.lock, engine.condition:
        if feedback_clips.busy_session(job_id):
            raise HTTPException(409, 'A reported clip is still saving. Wait for it to finish before removing this media.')
        if job.status == "analyzing" or job.capture_running or export_lock.locked():
            raise HTTPException(409, "Pause analysis and wait for the current chunk or export to finish")
        folder = job.path.parent.resolve()
        if folder.parent != SESSIONS.resolve():
            raise HTTPException(400, "Invalid session directory")
        try:
            shutil.rmtree(folder)
        except OSError as error:
            raise HTTPException(409, "Media is still open; stop playback and retry") from error
        del engine.jobs[job_id]
    return {"ok": True}


def feedback_context(job_id):
    # Metadata only: no decoding, inference or full-file hashing on the playback path.
    with engine.condition:
        job = get_job(job_id)
        stat = job.path.stat() if job.path.is_file() else None
        return dict(snapshot=job.snapshot(engine.models), analysis_identity=job.analysis_identity,
                    source_ref=dict(session_id=job.id, filename=job.path.name,
                                    size=stat.st_size if stat else None, mtime_ns=stat.st_mtime_ns if stat else None))


def feedback_available(report):
    job = engine.jobs.get(report['session_id'])
    if job is None or job.live:
        return False
    try:
        stat = job.path.stat()
    except OSError:
        return False
    source = report['context']['source_ref']
    return stat.st_size == source['size'] and stat.st_mtime_ns == source['mtime_ns']


def feedback_source(report):
    with engine.condition:
        job = get_job(report['session_id'])
        if job.live:
            raise HTTPException(422, 'Keeping clips from a live buffer is not supported yet.')
        if not feedback_available(report):
            raise HTTPException(409, 'The original video is unavailable or changed. Keep the local session until clips finish saving.')
        return job.path


import playback_feedback
from feedback_evidence import Evidence
feedback_clips = Evidence(ROOT, feedback_source)
app.include_router(playback_feedback.router(ROOT, feedback_context, feedback_available, feedback_clips))
app.include_router(labeling.router(ROOT))
import roi_labeling
app.include_router(roi_labeling.router(ROOT))

import segmentation_labels
app.include_router(segmentation_labels.router(ROOT))
app.mount("/", StaticFiles(directory=Path(__file__).parent / "web", html=True), name="web")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8502)
    args = parser.parse_args()
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
