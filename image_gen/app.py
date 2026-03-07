from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


PROVIDER = os.getenv("LOCALAI_IMAGE_GEN_PROVIDER", "mock").strip() or "mock"
ARTIFACT_STORE = os.getenv("LOCALAI_IMAGE_GEN_ARTIFACT_STORE", "filesystem").strip() or "filesystem"
DATA_DIR = Path(os.getenv("LOCALAI_IMAGE_GEN_DATA_DIR", "/data")).resolve()
PUBLIC_BASE_URL = os.getenv("LOCALAI_IMAGE_GEN_PUBLIC_BASE_URL", "http://127.0.0.1:8090").strip() or "http://127.0.0.1:8090"
MAX_CONCURRENCY = max(1, int(os.getenv("LOCALAI_IMAGE_GEN_CONCURRENCY", "1")))

ARTIFACTS_DIR = DATA_DIR / "artifacts"
JOBS_FILE = DATA_DIR / "jobs.json"

app = FastAPI(title="LocalAI Image Generation Service", version="0.1.0")


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    negative_prompt: str = ""
    width: int = Field(default=1024, ge=64, le=2048)
    height: int = Field(default=1024, ge=64, le=2048)
    steps: int = Field(default=20, ge=1, le=200)
    seed: int | None = None
    format: Literal["svg"] = "svg"


class JobRecord(BaseModel):
    id: str
    status: Literal["queued", "running", "done", "failed"]
    provider: str
    created_at: str
    updated_at: str
    prompt: str
    negative_prompt: str
    width: int
    height: int
    steps: int
    seed: int | None
    artifact_id: str = ""
    artifact_path: str = ""
    artifact_url: str = ""
    error: str = ""


JOBS: dict[str, JobRecord] = {}
QUEUE: asyncio.Queue[str] = asyncio.Queue()
WORKERS_STARTED = False


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_text(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _persist_jobs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {job_id: job.model_dump() for job_id, job in JOBS.items()}
    JOBS_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_jobs() -> None:
    if not JOBS_FILE.exists():
        return
    raw = json.loads(JOBS_FILE.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return
    for job_id, data in raw.items():
        if isinstance(data, dict):
            try:
                JOBS[job_id] = JobRecord(**data)
            except Exception:
                continue


def _render_svg(job: JobRecord) -> str:
    prompt = _safe_text(job.prompt[:180])
    negative = _safe_text(job.negative_prompt[:120]) if job.negative_prompt else "(none)"
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{job.width}" height="{job.height}" viewBox="0 0 {job.width} {job.height}">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#0f172a" />
      <stop offset="100%" stop-color="#1d4ed8" />
    </linearGradient>
  </defs>
  <rect width="100%" height="100%" fill="url(#bg)" />
  <rect x="24" y="24" width="{job.width - 48}" height="{job.height - 48}" rx="18" fill="#020617" fill-opacity="0.72" stroke="#38bdf8" stroke-opacity="0.4" />
  <text x="44" y="78" fill="#e2e8f0" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace" font-size="24">localai image-gen preview</text>
  <text x="44" y="124" fill="#bae6fd" font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI" font-size="18">prompt: {prompt}</text>
  <text x="44" y="160" fill="#93c5fd" font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI" font-size="16">negative: {negative}</text>
  <text x="44" y="{job.height - 52}" fill="#cbd5e1" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace" font-size="14">provider={_safe_text(job.provider)} steps={job.steps} seed={job.seed if job.seed is not None else "auto"}</text>
</svg>
"""


async def _process_job(job_id: str) -> None:
    job = JOBS[job_id]
    job.status = "running"
    job.updated_at = _now_iso()
    _persist_jobs()
    try:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        artifact_id = f"{job.id}.svg"
        artifact_path = ARTIFACTS_DIR / artifact_id
        artifact_path.write_text(_render_svg(job), encoding="utf-8")

        job.status = "done"
        job.updated_at = _now_iso()
        job.artifact_id = artifact_id
        job.artifact_path = str(artifact_path)
        job.artifact_url = f"{PUBLIC_BASE_URL.rstrip('/')}/api/images/assets/{artifact_id}"
        _persist_jobs()
    except Exception as e:  # noqa: BLE001
        job.status = "failed"
        job.error = str(e)
        job.updated_at = _now_iso()
        _persist_jobs()


async def _worker() -> None:
    while True:
        job_id = await QUEUE.get()
        try:
            await _process_job(job_id)
        finally:
            QUEUE.task_done()


@app.on_event("startup")
async def startup() -> None:
    global WORKERS_STARTED
    _load_jobs()
    if not WORKERS_STARTED:
        for _ in range(MAX_CONCURRENCY):
            asyncio.create_task(_worker())
        WORKERS_STARTED = True


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    queued = sum(1 for j in JOBS.values() if j.status == "queued")
    running = sum(1 for j in JOBS.values() if j.status == "running")
    done = sum(1 for j in JOBS.values() if j.status == "done")
    failed = sum(1 for j in JOBS.values() if j.status == "failed")
    return {
        "status": "ok",
        "provider": PROVIDER,
        "artifact_store": ARTIFACT_STORE,
        "queue_depth": QUEUE.qsize(),
        "jobs": {"queued": queued, "running": running, "done": done, "failed": failed},
    }


@app.post("/api/images/generate")
async def generate(req: GenerateRequest) -> dict[str, Any]:
    job_id = uuid.uuid4().hex
    now = _now_iso()
    job = JobRecord(
        id=job_id,
        status="queued",
        provider=PROVIDER,
        created_at=now,
        updated_at=now,
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        width=req.width,
        height=req.height,
        steps=req.steps,
        seed=req.seed,
    )
    JOBS[job_id] = job
    _persist_jobs()
    await QUEUE.put(job_id)
    return {"ok": True, "job_id": job_id, "status": "queued"}


@app.get("/api/images/jobs/{job_id}")
async def job_status(job_id: str) -> dict[str, Any]:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return {"ok": True, "job": job.model_dump()}


@app.get("/api/images/assets/{asset_id}")
async def get_asset(asset_id: str):
    path = (ARTIFACTS_DIR / asset_id).resolve()
    if not path.exists() or not str(path).startswith(str(ARTIFACTS_DIR)):
        raise HTTPException(status_code=404, detail="asset not found")
    return FileResponse(path, media_type="image/svg+xml", filename=asset_id)
