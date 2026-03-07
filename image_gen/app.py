from __future__ import annotations

import asyncio
import base64
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from minio import Minio
from minio.error import S3Error
from pydantic import BaseModel, Field


PROVIDER = os.getenv("LOCALAI_IMAGE_GEN_PROVIDER", "mock").strip().lower() or "mock"
ARTIFACT_STORE = os.getenv("LOCALAI_IMAGE_GEN_ARTIFACT_STORE", "filesystem").strip().lower() or "filesystem"
DATA_DIR = Path(os.getenv("LOCALAI_IMAGE_GEN_DATA_DIR", "/data")).resolve()
PUBLIC_BASE_URL = os.getenv("LOCALAI_IMAGE_GEN_PUBLIC_BASE_URL", "http://127.0.0.1:8090").strip() or "http://127.0.0.1:8090"
MAX_CONCURRENCY = max(1, int(os.getenv("LOCALAI_IMAGE_GEN_CONCURRENCY", "1")))
A1111_BASE_URL = os.getenv("LOCALAI_IMAGE_GEN_A1111_URL", "").strip().rstrip("/")

MINIO_ENDPOINT = os.getenv("LOCALAI_MINIO_ENDPOINT", "http://minio:9000").strip() or "http://minio:9000"
MINIO_BUCKET = os.getenv("LOCALAI_MINIO_BUCKET", "generated").strip() or "generated"
MINIO_USER = os.getenv("LOCALAI_MINIO_ROOT_USER", "localai").strip() or "localai"
MINIO_PASSWORD = os.getenv("LOCALAI_MINIO_ROOT_PASSWORD", "localai-localai-localai").strip() or "localai-localai-localai"

ARTIFACTS_DIR = DATA_DIR / "artifacts"
JOBS_FILE = DATA_DIR / "jobs.json"

app = FastAPI(title="LocalAI Image Generation Service", version="0.2.0")


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    negative_prompt: str = ""
    width: int = Field(default=1024, ge=64, le=2048)
    height: int = Field(default=1024, ge=64, le=2048)
    steps: int = Field(default=20, ge=1, le=200)
    seed: int | None = None


class OpenAIImageGenerateRequest(BaseModel):
    model: str = "localai-imagegen"
    prompt: str = Field(min_length=1, max_length=4000)
    n: int = Field(default=1, ge=1, le=4)
    size: str = "1024x1024"
    response_format: Literal["url", "b64_json"] = "url"


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
    artifact_mime: str = ""
    artifact_url: str = ""
    artifact_key: str = ""
    error: str = ""


JOBS: dict[str, JobRecord] = {}
QUEUE: asyncio.Queue[str] = asyncio.Queue()
WORKERS_STARTED = False
MINIO_CLIENT: Minio | None = None


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


def _parse_size(size: str) -> tuple[int, int]:
    raw = size.strip().lower()
    if "x" not in raw:
        raise HTTPException(status_code=400, detail="size must be WIDTHxHEIGHT")
    w, h = raw.split("x", 1)
    try:
        width = int(w)
        height = int(h)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"invalid size: {size}") from e
    if not (64 <= width <= 2048 and 64 <= height <= 2048):
        raise HTTPException(status_code=400, detail="size out of allowed range (64-2048)")
    return width, height


def _mime_ext(mime: str) -> str:
    if mime == "image/png":
        return "png"
    return "svg"


def _render_svg(req: GenerateRequest) -> bytes:
    prompt = _safe_text(req.prompt[:180])
    negative = _safe_text(req.negative_prompt[:120]) if req.negative_prompt else "(none)"
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{req.width}" height="{req.height}" viewBox="0 0 {req.width} {req.height}">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#0f172a" />
      <stop offset="100%" stop-color="#1d4ed8" />
    </linearGradient>
  </defs>
  <rect width="100%" height="100%" fill="url(#bg)" />
  <rect x="24" y="24" width="{req.width - 48}" height="{req.height - 48}" rx="18" fill="#020617" fill-opacity="0.72" stroke="#38bdf8" stroke-opacity="0.4" />
  <text x="44" y="78" fill="#e2e8f0" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace" font-size="24">localai image-gen preview</text>
  <text x="44" y="124" fill="#bae6fd" font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI" font-size="18">prompt: {prompt}</text>
  <text x="44" y="160" fill="#93c5fd" font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI" font-size="16">negative: {negative}</text>
  <text x="44" y="{req.height - 52}" fill="#cbd5e1" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace" font-size="14">provider=mock steps={req.steps} seed={req.seed if req.seed is not None else "auto"}</text>
</svg>
"""
    return svg.encode("utf-8")


async def _generate_with_a1111(req: GenerateRequest) -> tuple[str, bytes]:
    if not A1111_BASE_URL:
        raise RuntimeError("LOCALAI_IMAGE_GEN_A1111_URL is required when provider=automatic1111")
    payload = {
        "prompt": req.prompt,
        "negative_prompt": req.negative_prompt,
        "steps": req.steps,
        "width": req.width,
        "height": req.height,
        "seed": req.seed if req.seed is not None else -1,
        "sampler_name": "Euler a",
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(f"{A1111_BASE_URL}/sdapi/v1/txt2img", json=payload)
    if resp.status_code >= 400:
        raise RuntimeError(f"automatic1111 returned {resp.status_code}: {resp.text[:240]}")
    data = resp.json() if resp.content else {}
    images = data.get("images", []) if isinstance(data, dict) else []
    if not images:
        raise RuntimeError("automatic1111 returned no images")
    first = str(images[0]).strip()
    if "," in first and first.startswith("data:"):
        first = first.split(",", 1)[1]
    try:
        return "image/png", base64.b64decode(first, validate=False)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"automatic1111 returned invalid base64 image: {e}") from e


async def _provider_generate(req: GenerateRequest) -> tuple[str, bytes]:
    if PROVIDER in {"automatic1111", "a1111"}:
        return await _generate_with_a1111(req)
    return "image/svg+xml", _render_svg(req)


def _init_minio() -> None:
    global MINIO_CLIENT
    endpoint = MINIO_ENDPOINT
    secure = endpoint.startswith("https://")
    host = endpoint.replace("https://", "").replace("http://", "")
    client = Minio(host, access_key=MINIO_USER, secret_key=MINIO_PASSWORD, secure=secure)
    try:
        if not client.bucket_exists(MINIO_BUCKET):
            client.make_bucket(MINIO_BUCKET)
    except S3Error as e:
        raise RuntimeError(f"minio init failed: {e}") from e
    MINIO_CLIENT = client


def _save_artifact_bytes(job_id: str, mime: str, content: bytes) -> tuple[str, str]:
    ext = _mime_ext(mime)
    artifact_id = f"{job_id}.{ext}"
    key = f"artifacts/{artifact_id}"

    if ARTIFACT_STORE == "minio":
        if MINIO_CLIENT is None:
            raise RuntimeError("minio client not initialized")
        try:
            from io import BytesIO

            data = BytesIO(content)
            MINIO_CLIENT.put_object(
                MINIO_BUCKET,
                key,
                data,
                length=len(content),
                content_type=mime,
            )
            return artifact_id, key
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"minio put failed: {e}") from e

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS_DIR / artifact_id
    path.write_bytes(content)
    return artifact_id, str(path)


def _read_artifact_bytes(job: JobRecord) -> bytes:
    if ARTIFACT_STORE == "minio":
        if MINIO_CLIENT is None:
            raise HTTPException(status_code=500, detail="minio client not initialized")
        try:
            resp = MINIO_CLIENT.get_object(MINIO_BUCKET, job.artifact_key)
            try:
                return resp.read()
            finally:
                resp.close()
                resp.release_conn()
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=404, detail=f"artifact not found: {e}") from e

    path = Path(job.artifact_key)
    if not path.exists():
        raise HTTPException(status_code=404, detail="artifact not found")
    return path.read_bytes()


async def _generate_and_store(req: GenerateRequest, job_id: str) -> tuple[str, str, str, str]:
    mime, content = await _provider_generate(req)
    artifact_id, key = _save_artifact_bytes(job_id, mime, content)
    artifact_url = f"{PUBLIC_BASE_URL.rstrip('/')}/api/images/assets/{artifact_id}"
    return artifact_id, mime, key, artifact_url


async def _process_job(job_id: str) -> None:
    job = JOBS[job_id]
    job.status = "running"
    job.updated_at = _now_iso()
    _persist_jobs()

    req = GenerateRequest(
        prompt=job.prompt,
        negative_prompt=job.negative_prompt,
        width=job.width,
        height=job.height,
        steps=job.steps,
        seed=job.seed,
    )
    try:
        artifact_id, artifact_mime, artifact_key, artifact_url = await _generate_and_store(req, job.id)
        job.status = "done"
        job.updated_at = _now_iso()
        job.artifact_id = artifact_id
        job.artifact_mime = artifact_mime
        job.artifact_key = artifact_key
        job.artifact_url = artifact_url
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


async def _provider_ready() -> tuple[bool, str]:
    if PROVIDER in {"automatic1111", "a1111"}:
        if not A1111_BASE_URL:
            return False, "LOCALAI_IMAGE_GEN_A1111_URL missing"
        try:
            async with httpx.AsyncClient(timeout=6.0) as client:
                resp = await client.get(f"{A1111_BASE_URL}/sdapi/v1/options")
            if resp.status_code >= 400:
                return False, f"automatic1111 returned {resp.status_code}"
            return True, "ok"
        except Exception as e:  # noqa: BLE001
            return False, str(e)
    return True, "ok"


@app.on_event("startup")
async def startup() -> None:
    global WORKERS_STARTED
    _load_jobs()
    if ARTIFACT_STORE == "minio":
        _init_minio()
    else:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

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
    provider_ok, provider_detail = await _provider_ready()
    return {
        "status": "ok" if provider_ok else "degraded",
        "provider": PROVIDER,
        "provider_ready": provider_ok,
        "provider_detail": provider_detail,
        "artifact_store": ARTIFACT_STORE,
        "bucket": MINIO_BUCKET if ARTIFACT_STORE == "minio" else "",
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
    for job in JOBS.values():
        if job.artifact_id == asset_id and job.status == "done":
            content = _read_artifact_bytes(job)
            return Response(content=content, media_type=job.artifact_mime or "application/octet-stream")
    raise HTTPException(status_code=404, detail="asset not found")


@app.post("/v1/images/generations")
async def openai_images(req: OpenAIImageGenerateRequest) -> dict[str, Any]:
    width, height = _parse_size(req.size)
    now = int(datetime.now(timezone.utc).timestamp())
    data: list[dict[str, str]] = []
    for _ in range(req.n):
        one = GenerateRequest(
            prompt=req.prompt,
            width=width,
            height=height,
            steps=20,
            seed=None,
        )
        job_id = uuid.uuid4().hex
        artifact_id, artifact_mime, artifact_key, artifact_url = await _generate_and_store(one, job_id)

        job_now = _now_iso()
        JOBS[job_id] = JobRecord(
            id=job_id,
            status="done",
            provider=PROVIDER,
            created_at=job_now,
            updated_at=job_now,
            prompt=one.prompt,
            negative_prompt=one.negative_prompt,
            width=one.width,
            height=one.height,
            steps=one.steps,
            seed=one.seed,
            artifact_id=artifact_id,
            artifact_mime=artifact_mime,
            artifact_key=artifact_key,
            artifact_url=artifact_url,
        )

        if req.response_format == "b64_json":
            # OpenAI-compatible key.
            if artifact_mime == "image/svg+xml":
                content = _read_artifact_bytes(JOBS[job_id])
                encoded = base64.b64encode(content).decode("ascii")
            else:
                content = _read_artifact_bytes(JOBS[job_id])
                encoded = base64.b64encode(content).decode("ascii")
            data.append({"b64_json": encoded})
        else:
            data.append({"url": artifact_url})

    _persist_jobs()
    return {"created": now, "data": data}
