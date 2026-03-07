from __future__ import annotations

import base64
import asyncio
import json
import os
import re
import secrets
from urllib.parse import quote_plus, urlparse
from pathlib import Path
from time import monotonic, perf_counter
from typing import Any, Literal

import httpx
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434").rstrip("/")
OLLAMA_LIBRARY_URL = os.getenv("OLLAMA_LIBRARY_URL", "https://ollama.com/library")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


LOCALAI_HOST_RAM_GB = max(8, _env_int("LOCALAI_HOST_RAM_GB", 32))
LOCALAI_CPU_LOGICAL = max(0, _env_int("LOCALAI_CPU_LOGICAL", 0))
LOCALAI_CPU_PHYSICAL = max(0, _env_int("LOCALAI_CPU_PHYSICAL", 0))
LOCALAI_CPU_PERF = max(0, _env_int("LOCALAI_CPU_PERF", 0))
LOCALAI_MACHINE = os.getenv("LOCALAI_MACHINE", "arm64")
LOCALAI_HW_MODEL = os.getenv("LOCALAI_HW_MODEL", "unknown")
LOCALAI_GPU_BACKEND = os.getenv("LOCALAI_GPU_BACKEND", "Metal")
LOCALAI_GPU_NAME = os.getenv("LOCALAI_GPU_NAME", "Apple Silicon Integrated GPU")
LOCALAI_GPU_CORES = os.getenv("LOCALAI_GPU_CORES", "unknown")
LOCALAI_OLLAMA_NUM_PARALLEL = max(1, _env_int("LOCALAI_OLLAMA_NUM_PARALLEL", 4))
LOCALAI_OLLAMA_MAX_LOADED_MODELS = max(1, _env_int("LOCALAI_OLLAMA_MAX_LOADED_MODELS", 1))
LOCALAI_OLLAMA_KEEP_ALIVE = os.getenv("LOCALAI_OLLAMA_KEEP_ALIVE", "30m")
LOCALAI_OLLAMA_MAX_QUEUE = max(1, _env_int("LOCALAI_OLLAMA_MAX_QUEUE", 160))
LOCALAI_BOOST_ACTIVE = os.getenv("LOCALAI_BOOST_ACTIVE", "0") == "1"
LOCALAI_RAG_PRESET = os.getenv("LOCALAI_RAG_PRESET", "fast").strip().lower() or "fast"
LOCALAI_QDRANT_ENABLED = os.getenv("LOCALAI_QDRANT_ENABLED", "1") == "1"
LOCALAI_WEB_ENABLED = _env_bool("LOCALAI_WEB_ENABLED", False)
LOCALAI_SEARXNG_QUERY_URL = os.getenv("LOCALAI_SEARXNG_QUERY_URL", "http://searxng:8080/search?q=<query>&format=json").strip()
LOCALAI_WEB_REDIS_MAXMEMORY_MB = max(1, _env_int("LOCALAI_WEB_REDIS_MAXMEMORY_MB", 256))
LOCALAI_SEARXNG_REDIS_URL = os.getenv("LOCALAI_SEARXNG_REDIS_URL", "redis://redis:6379/0").strip()
LOCALAI_VISION_ENABLED = os.getenv("LOCALAI_VISION_ENABLED", "0") == "1"
LOCALAI_VISION_DEFAULT_MODEL = os.getenv("LOCALAI_VISION_DEFAULT_MODEL", "llava:latest").strip() or "llava:latest"
LOCALAI_VISION_MAX_IMAGE_MB = max(1, _env_int("LOCALAI_VISION_MAX_IMAGE_MB", 10))
LOCALAI_VISION_BENCHMARK_DATASET = os.getenv("LOCALAI_VISION_BENCHMARK_DATASET", "tests/fixtures/vision/smoke.jsonl").strip()
LOCALAI_IMAGE_GEN_ENABLED = os.getenv("LOCALAI_IMAGE_GEN_ENABLED", "0") == "1"
LOCALAI_IMAGE_GEN_PROVIDER = os.getenv("LOCALAI_IMAGE_GEN_PROVIDER", "none").strip() or "none"
LOCALAI_IMAGE_GEN_BACKEND_URL = os.getenv("LOCALAI_IMAGE_GEN_BACKEND_URL", "http://image-gen:8090").strip() or "http://image-gen:8090"
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333").rstrip("/")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip()
MODEL_ADMIN_USERNAME = os.getenv("MODEL_ADMIN_USERNAME", "").strip()
MODEL_ADMIN_PASSWORD = os.getenv("MODEL_ADMIN_PASSWORD", "")

POPULAR_MODELS: list[dict[str, str]] = [
    {"name": "llama3.2", "tag": "3b", "description": "Fast general-purpose model"},
    {"name": "llama3.3", "tag": "latest", "description": "Large-capability Llama model"},
    {"name": "qwen2.5", "tag": "7b", "description": "Strong coding + reasoning balance"},
    {"name": "deepseek-r1", "tag": "latest", "description": "Reasoning-focused model family"},
    {"name": "mistral", "tag": "latest", "description": "Compact high-quality instruction model"},
    {"name": "mistral-small", "tag": "latest", "description": "Lower-latency Mistral variant"},
    {"name": "gemma3", "tag": "latest", "description": "Google Gemma open model line"},
    {"name": "phi4", "tag": "latest", "description": "Small but capable model"},
    {"name": "codestral", "tag": "latest", "description": "Code-first model"},
    {"name": "qwen2.5-coder", "tag": "latest", "description": "Coding-specialized Qwen"},
    {"name": "nomic-embed-text", "tag": "latest", "description": "Embedding model for RAG"},
    {"name": "bge-m3", "tag": "latest", "description": "Multilingual embedding model"},
    {"name": "llava", "tag": "latest", "description": "Vision-language model"},
]

app = FastAPI(title="LocalAI Model Admin", version="0.4.0")

ACTION_STATE: dict[str, Any] = {
    "active_jobs": 0,
    "completed_jobs": 0,
    "failed_jobs": 0,
    "last_action": "",
    "last_model": "",
    "last_duration_ms": 0,
    "last_error": "",
}

QDRANT_METRICS_CACHE: dict[str, Any] = {
    "expires_at": 0.0,
    "payload": None,
}
WEB_METRICS_CACHE: dict[str, Any] = {
    "expires_at": 0.0,
    "payload": None,
}
WEB_METRICS_STATE: dict[str, int] = {
    "search_ok": 0,
    "search_err": 0,
}


def _is_auth_enabled() -> bool:
    return bool(MODEL_ADMIN_USERNAME and MODEL_ADMIN_PASSWORD)


def _basic_unauthorized() -> Response:
    return Response(
        status_code=401,
        content="Unauthorized",
        headers={"WWW-Authenticate": 'Basic realm="Model Admin"'},
    )


def _auth_ok(header_value: str) -> bool:
    if not header_value.startswith("Basic "):
        return False
    try:
        raw = base64.b64decode(header_value.split(" ", 1)[1]).decode("utf-8")
    except Exception:
        return False
    user, sep, pwd = raw.partition(":")
    if not sep:
        return False
    return secrets.compare_digest(user, MODEL_ADMIN_USERNAME) and secrets.compare_digest(pwd, MODEL_ADMIN_PASSWORD)


@app.middleware("http")
async def require_auth(request: Request, call_next):
    if not _is_auth_enabled():
        return await call_next(request)
    if request.url.path == "/healthz":
        return await call_next(request)
    auth = request.headers.get("Authorization", "")
    if not _auth_ok(auth):
        return _basic_unauthorized()
    return await call_next(request)


class ModelAction(BaseModel):
    model: str = Field(min_length=1)


class ConsoleRequest(BaseModel):
    op: Literal["tags", "ps", "show", "generate", "pull"]
    model: str = ""
    prompt: str = ""


class SmokeTestRequest(BaseModel):
    model: str = ""
    prompt: str = "Reply with exactly: ok"
    timeout_seconds: float = Field(default=20.0, ge=1.0, le=120.0)


class BenchmarkRequest(BaseModel):
    model: str = ""
    prompt: str = "Summarize why local-first AI stacks are useful in one sentence."
    iterations: int = Field(default=5, ge=1, le=40)
    timeout_seconds: float = Field(default=60.0, ge=1.0, le=180.0)


class VisionAnalyzeRequest(BaseModel):
    model: str = ""
    prompt: str = "Describe the image in one paragraph."
    image_base64: str = ""
    image_url: str = ""


class VisionSmokeRequest(BaseModel):
    model: str = ""
    prompt: str = "Return exactly: ok"
    image_base64: str = ""
    image_url: str = ""
    timeout_seconds: float = Field(default=30.0, ge=1.0, le=180.0)


class VisionBenchmarkRequest(BaseModel):
    model: str = ""
    dataset_path: str = ""
    iterations: int = Field(default=5, ge=1, le=1000)
    timeout_seconds: float = Field(default=120.0, ge=1.0, le=300.0)


def _sum_int(items: list[dict[str, Any]], key: str) -> int:
    total = 0
    for item in items:
        value = item.get(key, 0)
        if isinstance(value, int):
            total += value
    return total


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int((q / 100.0) * len(ordered) - 0.000001)))
    return ordered[idx]


def _by_name(items: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for item in items:
        name = item.get("name", "").strip().lower()
        if name:
            out[name] = item
    return out


def _classify_model(name: str) -> str:
    n = name.lower()
    if any(x in n for x in ["embed", "bge", "e5", "nomic"]):
        return "embed"
    if any(x in n for x in ["coder", "codestral", "codegemma", "code"]):
        return "code"
    if any(x in n for x in ["llava", "vision", "moondream", "vl"]):
        return "vision"
    if any(x in n for x in ["r1", "reason"]):
        return "reasoning"
    return "chat"


def _estimate_ram_gb(model_name: str, tag: str, model_class: str) -> float:
    if model_class == "embed":
        return 2.0

    match = re.search(r"(\d+)(?:\.\d+)?b", tag.lower())
    if match:
        params_b = float(match.group(1))
        return max(2.0, round(params_b * 0.8, 1))

    if "llama3.3" in model_name:
        return 32.0
    if "phi" in model_name or "mistral-small" in model_name:
        return 4.0
    return 8.0


def _fit_tier(required_ram_gb: float, host_ram_gb: int) -> str:
    if required_ram_gb <= host_ram_gb * 0.35:
        return "great"
    if required_ram_gb <= host_ram_gb * 0.60:
        return "good"
    if required_ram_gb <= host_ram_gb * 0.85:
        return "tight"
    return "heavy"


def _is_generate_capable_model(model_name: str) -> bool:
    return _classify_model(model_name) != "embed"


def _require_vision_enabled() -> None:
    if not LOCALAI_VISION_ENABLED:
        raise HTTPException(status_code=503, detail="vision lane disabled; set LOCALAI_VISION_ENABLED=1")


def _resolve_vision_model(model: str) -> str:
    chosen = model.strip() or LOCALAI_VISION_DEFAULT_MODEL
    if _classify_model(chosen) != "vision":
        raise HTTPException(status_code=400, detail=f"model '{chosen}' does not appear vision-capable")
    return chosen


def _validate_vision_image_input(image_base64: str, image_url: str) -> None:
    if not image_base64.strip() and not image_url.strip():
        raise HTTPException(status_code=400, detail="one of image_base64 or image_url is required")


def _strip_data_url_prefix(image_base64: str) -> str:
    raw = image_base64.strip()
    if raw.startswith("data:") and "," in raw:
        return raw.split(",", 1)[1].strip()
    return raw


def _validate_base64_bytes(image_b64: str) -> bytes:
    try:
        return base64.b64decode(image_b64, validate=True)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"invalid image_base64 payload: {e}") from e


async def _resolve_image_base64(image_base64: str, image_url: str) -> str:
    if image_base64.strip():
        normalized = _strip_data_url_prefix(image_base64)
        image_bytes = _validate_base64_bytes(normalized)
        if len(image_bytes) > LOCALAI_VISION_MAX_IMAGE_MB * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"image exceeds max size ({LOCALAI_VISION_MAX_IMAGE_MB} MB)")
        return normalized

    parsed = urlparse(image_url.strip())
    if parsed.scheme not in {"http", "https"}:
        raise HTTPException(status_code=400, detail="image_url must use http or https")

    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            resp = await client.get(image_url.strip())
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"failed to fetch image_url: {e}") from e

    if resp.status_code >= 400:
        raise HTTPException(status_code=400, detail=f"image_url fetch failed with status {resp.status_code}")
    image_bytes = resp.content or b""
    if not image_bytes:
        raise HTTPException(status_code=400, detail="image_url returned empty content")
    if len(image_bytes) > LOCALAI_VISION_MAX_IMAGE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"image exceeds max size ({LOCALAI_VISION_MAX_IMAGE_MB} MB)")
    return base64.b64encode(image_bytes).decode("ascii")


def _extract_response_text(out: dict[str, Any]) -> str:
    text = str(out.get("response", "") or "").strip()
    if text:
        return text
    message = out.get("message")
    if isinstance(message, dict):
        content = str(message.get("content", "") or "").strip()
        if content:
            return content
    return ""


async def _vision_infer(model: str, prompt: str, image_b64: str, timeout_seconds: float) -> dict[str, Any]:
    out = await _ollama_request(
        "POST",
        "/api/generate",
        {
            "model": model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
        },
        timeout_seconds=timeout_seconds,
    )
    eval_count = int(out.get("eval_count", 0) or 0)
    eval_duration_ns = int(out.get("eval_duration", 0) or 0)
    total_duration_ns = int(out.get("total_duration", 0) or 0)
    token_s = round((eval_count / (eval_duration_ns / 1_000_000_000.0)), 2) if eval_duration_ns > 0 else 0.0
    response_text = _extract_response_text(out)
    return {
        "response": response_text,
        "raw": out,
        "eval_count": eval_count,
        "eval_duration_ms": round(eval_duration_ns / 1_000_000.0, 2),
        "total_duration_ms": round(total_duration_ns / 1_000_000.0, 2),
        "tokens_per_second": token_s,
    }


def _resolve_dataset_path(dataset_path: str) -> Path:
    raw = dataset_path.strip() or LOCALAI_VISION_BENCHMARK_DATASET
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    if not p.exists():
        raise HTTPException(status_code=400, detail=f"dataset file not found: {p}")
    return p


def _select_benchmark_model(models: list[dict[str, Any]], requested: str) -> str:
    model = requested.strip()
    if model:
        if not _is_generate_capable_model(model):
            raise HTTPException(
                status_code=400,
                detail=f"model '{model}' appears to be embedding-only; choose a chat/code/reasoning model for generate benchmarks",
            )
        return model

    discovered: list[str] = []
    for item in models:
        if isinstance(item, dict):
            name = str(item.get("name", "")).strip()
            if name:
                discovered.append(name)

    for name in discovered:
        if _is_generate_capable_model(name):
            return name

    raise HTTPException(status_code=400, detail="no generation-capable model available to benchmark")


async def _ollama_get(path: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(f"{OLLAMA_BASE_URL}{path}")
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return resp.json()


def _empty_qdrant_metrics(*, error: str = "") -> dict[str, Any]:
    return {
        "enabled": LOCALAI_QDRANT_ENABLED,
        "up": False,
        "latency_ms": -1,
        "collections": 0,
        "points_total": 0,
        "indexed_vectors_total": 0,
        "segments_total": 0,
        "error": error,
    }


async def _qdrant_metrics() -> dict[str, Any]:
    if not LOCALAI_QDRANT_ENABLED:
        return {
            "enabled": False,
            "up": False,
            "latency_ms": -1,
            "collections": 0,
            "points_total": 0,
            "indexed_vectors_total": 0,
            "segments_total": 0,
            "error": "qdrant disabled",
        }

    now = monotonic()
    cached = QDRANT_METRICS_CACHE.get("payload")
    if cached and now < float(QDRANT_METRICS_CACHE.get("expires_at", 0.0)):
        return cached

    headers: dict[str, str] = {}
    if QDRANT_API_KEY:
        headers["api-key"] = QDRANT_API_KEY

    start = perf_counter()
    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            list_resp = await client.get(f"{QDRANT_URL}/collections", headers=headers)
            list_resp.raise_for_status()
            latency_ms = int((perf_counter() - start) * 1000)
            list_data = list_resp.json() if list_resp.content else {}

            collections_raw = ((list_data.get("result") or {}).get("collections") or []) if isinstance(list_data, dict) else []
            names = [str(item.get("name", "")).strip() for item in collections_raw if isinstance(item, dict)]
            names = [name for name in names if name]

            points_total = 0
            indexed_total = 0
            segments_total = 0
            for name in names:
                detail = await client.get(f"{QDRANT_URL}/collections/{name}", headers=headers)
                detail.raise_for_status()
                detail_data = detail.json() if detail.content else {}
                result = detail_data.get("result", {}) if isinstance(detail_data, dict) else {}
                points_total += int(result.get("points_count", 0) or 0)
                indexed_total += int(result.get("indexed_vectors_count", 0) or 0)
                segments_total += int(result.get("segments_count", 0) or 0)

        payload = {
            "enabled": True,
            "up": True,
            "latency_ms": latency_ms,
            "collections": len(names),
            "points_total": points_total,
            "indexed_vectors_total": indexed_total,
            "segments_total": segments_total,
            "error": "",
        }
    except Exception as e:  # noqa: BLE001
        payload = _empty_qdrant_metrics(error=str(e))

    QDRANT_METRICS_CACHE["payload"] = payload
    QDRANT_METRICS_CACHE["expires_at"] = now + 5.0
    return payload


def _build_searxng_probe_url(query: str = "latest ai news") -> str:
    template = LOCALAI_SEARXNG_QUERY_URL or "http://searxng:8080/search?q=<query>&format=json"
    encoded = quote_plus(query)
    if "<query>" in template:
        url = template.replace("<query>", encoded)
    else:
        sep = "&" if "?" in template else "?"
        url = f"{template}{sep}q={encoded}"
    if "format=" not in url:
        join = "&" if "?" in url else "?"
        url = f"{url}{join}format=json"
    return url


def _redis_parse_info(raw: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or ":" not in s:
            continue
        k, v = s.split(":", 1)
        out[k.strip()] = v.strip()
    return out


async def _redis_send_command(reader, writer, *parts: str) -> str:  # type: ignore[no-untyped-def]
    payload = f"*{len(parts)}\r\n"
    for part in parts:
        b = part.encode("utf-8")
        payload += f"${len(b)}\r\n{part}\r\n"
    writer.write(payload.encode("utf-8"))
    await writer.drain()

    lead = await reader.readexactly(1)
    if lead == b"+":
        return (await reader.readline()).decode("utf-8", errors="ignore").strip()
    if lead == b":":
        return (await reader.readline()).decode("utf-8", errors="ignore").strip()
    if lead == b"$":
        line = await reader.readline()
        n = int(line.decode("utf-8", errors="ignore").strip() or "-1")
        if n < 0:
            return ""
        data = await reader.readexactly(n)
        await reader.readexactly(2)  # trailing CRLF
        return data.decode("utf-8", errors="ignore")
    if lead == b"-":
        msg = (await reader.readline()).decode("utf-8", errors="ignore").strip()
        raise RuntimeError(msg)
    raise RuntimeError("unsupported redis reply type")


def _empty_web_metrics(*, error: str = "") -> dict[str, Any]:
    return {
        "enabled": LOCALAI_WEB_ENABLED,
        "searxng_up": False,
        "searxng_latency_ms": -1,
        "last_results_count": -1,
        "search_ok": WEB_METRICS_STATE["search_ok"],
        "search_err": WEB_METRICS_STATE["search_err"],
        "search_error_rate": 0.0,
        "redis_up": False,
        "redis_latency_ms": -1,
        "redis_used_memory_mb": -1.0,
        "redis_maxmemory_mb": float(LOCALAI_WEB_REDIS_MAXMEMORY_MB),
        "redis_memory_pct": -1.0,
        "redis_connected_clients": -1,
        "redis_hit_ratio": -1.0,
        "error": error,
    }


async def _web_metrics() -> dict[str, Any]:
    if not LOCALAI_WEB_ENABLED:
        return _empty_web_metrics(error="web search disabled")

    now = monotonic()
    cached = WEB_METRICS_CACHE.get("payload")
    if cached and now < float(WEB_METRICS_CACHE.get("expires_at", 0.0)):
        return cached

    payload = _empty_web_metrics()
    errors: list[str] = []

    # SearxNG probe.
    searx_url = _build_searxng_probe_url()
    try:
        start = perf_counter()
        async with httpx.AsyncClient(timeout=6.0) as client:
            resp = await client.get(searx_url)
            resp.raise_for_status()
            latency_ms = int((perf_counter() - start) * 1000)
            body = resp.json() if resp.content else {}
        results = body.get("results", []) if isinstance(body, dict) else []
        payload["searxng_up"] = True
        payload["searxng_latency_ms"] = latency_ms
        payload["last_results_count"] = len(results) if isinstance(results, list) else 0
        WEB_METRICS_STATE["search_ok"] += 1
    except Exception as e:  # noqa: BLE001
        WEB_METRICS_STATE["search_err"] += 1
        errors.append(f"searxng: {e}")

    # Redis probe.
    try:
        parsed = urlparse(LOCALAI_SEARXNG_REDIS_URL or "redis://redis:6379/0")
        host = parsed.hostname or "redis"
        port = parsed.port or 6379
        db = 0
        if parsed.path and parsed.path != "/":
            try:
                db = int(parsed.path.strip("/"))
            except ValueError:
                db = 0
        password = parsed.password or ""

        start = perf_counter()
        reader, writer = await asyncio.open_connection(host, port)
        try:
            if password:
                await _redis_send_command(reader, writer, "AUTH", password)
            if db > 0:
                await _redis_send_command(reader, writer, "SELECT", str(db))
            pong = await _redis_send_command(reader, writer, "PING")
            info_memory = await _redis_send_command(reader, writer, "INFO", "memory")
            info_stats = await _redis_send_command(reader, writer, "INFO", "stats")
            info_clients = await _redis_send_command(reader, writer, "INFO", "clients")
        finally:
            writer.close()
            await writer.wait_closed()

        payload["redis_latency_ms"] = int((perf_counter() - start) * 1000)
        payload["redis_up"] = str(pong).upper() == "PONG"

        mem = _redis_parse_info(info_memory)
        stats = _redis_parse_info(info_stats)
        clients = _redis_parse_info(info_clients)
        used_bytes = int(mem.get("used_memory", "0") or 0)
        maxmemory_bytes = int(mem.get("maxmemory", "0") or 0)
        hits = int(stats.get("keyspace_hits", "0") or 0)
        misses = int(stats.get("keyspace_misses", "0") or 0)
        total = hits + misses

        used_mb = used_bytes / (1024 * 1024)
        max_mb = maxmemory_bytes / (1024 * 1024) if maxmemory_bytes > 0 else float(LOCALAI_WEB_REDIS_MAXMEMORY_MB)
        payload["redis_used_memory_mb"] = round(used_mb, 2)
        payload["redis_maxmemory_mb"] = round(max_mb, 2)
        payload["redis_memory_pct"] = round((used_mb / max_mb) * 100.0, 1) if max_mb > 0 else -1.0
        payload["redis_connected_clients"] = int(clients.get("connected_clients", "0") or 0)
        payload["redis_hit_ratio"] = round((hits / total) * 100.0, 1) if total > 0 else -1.0
    except Exception as e:  # noqa: BLE001
        errors.append(f"redis: {e}")

    payload["search_ok"] = WEB_METRICS_STATE["search_ok"]
    payload["search_err"] = WEB_METRICS_STATE["search_err"]
    total_search = payload["search_ok"] + payload["search_err"]
    payload["search_error_rate"] = round((payload["search_err"] / total_search) * 100.0, 1) if total_search > 0 else 0.0
    payload["error"] = "; ".join(errors)

    WEB_METRICS_CACHE["payload"] = payload
    WEB_METRICS_CACHE["expires_at"] = now + 5.0
    return payload


async def _ollama_stream_post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    last: dict[str, Any] = {}
    raw_lines: list[str] = []
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", f"{OLLAMA_BASE_URL}{path}", json=payload) as resp:
            if resp.status_code >= 400:
                body = await resp.aread()
                raise HTTPException(status_code=resp.status_code, detail=body.decode("utf-8", errors="ignore"))

            async for line in resp.aiter_lines():
                if not line:
                    continue
                raw_lines.append(line)
                try:
                    item = json.loads(line)
                    if isinstance(item, dict):
                        last = item
                except Exception:
                    continue

    return {
        "ok": True,
        "last": last,
        "events": raw_lines[-15:],
    }


async def _ollama_request(method: str, path: str, payload: dict[str, Any], *, timeout_seconds: float = 30.0) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        resp = await client.request(method, f"{OLLAMA_BASE_URL}{path}", json=payload)
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    if resp.content:
        try:
            return resp.json()
        except Exception:
            return {"ok": True, "raw": resp.text}
    return {"ok": True}


async def _tracked_action(
    action: str,
    model: str,
    path: str,
    payload: dict[str, Any],
    *,
    stream: bool,
    method: str = "POST",
) -> dict[str, Any]:
    ACTION_STATE["active_jobs"] += 1
    ACTION_STATE["last_action"] = action
    ACTION_STATE["last_model"] = model
    ACTION_STATE["last_error"] = ""

    start = perf_counter()
    try:
        if stream:
            result = await _ollama_stream_post(path, payload)
        else:
            try:
                result = await _ollama_request(method, path, payload)
            except HTTPException as e:
                # Compatibility path for older/newer Ollama API variants.
                if method == "DELETE" and e.status_code == 405:
                    result = await _ollama_request("POST", path, payload)
                else:
                    raise
        ACTION_STATE["completed_jobs"] += 1
        return result
    except Exception as e:  # noqa: BLE001
        ACTION_STATE["failed_jobs"] += 1
        ACTION_STATE["last_error"] = str(e)
        raise
    finally:
        ACTION_STATE["active_jobs"] -= 1
        ACTION_STATE["last_duration_ms"] = int((perf_counter() - start) * 1000)


async def _discover_library_names(limit: int = 120) -> list[str]:
    async with httpx.AsyncClient(timeout=8.0) as client:
        resp = await client.get(OLLAMA_LIBRARY_URL)
    if resp.status_code >= 400:
        return []

    names = re.findall(r'href="/library/([a-zA-Z0-9._-]+)"', resp.text)
    dedup: list[str] = []
    seen: set[str] = set()
    for name in names:
        n = name.strip().lower()
        if not n or n in seen:
            continue
        seen.add(n)
        dedup.append(n)
        if len(dedup) >= limit:
            break
    return dedup


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/models")
async def list_models() -> dict[str, Any]:
    return await _ollama_get("/api/tags")


@app.get("/api/catalog")
async def catalog(q: str = Query("", max_length=80), limit: int = Query(60, ge=5, le=200)) -> dict[str, Any]:
    q_norm = q.strip().lower()

    tags_data = await _ollama_get("/api/tags")
    local_models = tags_data.get("models", []) if isinstance(tags_data, dict) else []

    local_by_base: dict[str, list[str]] = {}
    for model in local_models:
        raw = str(model.get("name", "")).strip()
        if not raw:
            continue
        base = raw.split(":", 1)[0].lower()
        local_by_base.setdefault(base, []).append(raw)

    curated = _by_name(POPULAR_MODELS)
    discovered = await _discover_library_names(limit=limit * 2)

    merged_names: list[str] = []
    seen: set[str] = set()
    for name in list(curated.keys()) + discovered:
        n = name.strip().lower()
        if not n or n in seen:
            continue
        seen.add(n)
        merged_names.append(n)

    items: list[dict[str, Any]] = []
    for name in merged_names:
        if q_norm and q_norm not in name:
            continue

        curated_item = curated.get(name)
        tag = (curated_item or {}).get("tag", "latest")
        desc = (curated_item or {}).get("description", "")
        pull_ref = f"{name}:{tag}"
        installed_tags = sorted(local_by_base.get(name, []))

        model_class = _classify_model(name)
        est_ram_gb = _estimate_ram_gb(name, tag, model_class)
        fit = _fit_tier(est_ram_gb, LOCALAI_HOST_RAM_GB)

        items.append(
            {
                "name": name,
                "tag": tag,
                "pull_ref": pull_ref,
                "description": desc,
                "installed": len(installed_tags) > 0,
                "installed_tags": installed_tags,
                "source": "curated" if curated_item else "library",
                "model_class": model_class,
                "est_ram_gb": est_ram_gb,
                "fit": fit,
            }
        )

        if len(items) >= limit:
            break

    return {
        "items": items,
        "query": q,
        "total": len(items),
        "host_ram_gb": LOCALAI_HOST_RAM_GB,
        "library_url": OLLAMA_LIBRARY_URL,
    }


@app.get("/api/metrics")
async def metrics() -> dict[str, Any]:
    qdrant = await _qdrant_metrics()
    web = await _web_metrics()
    tags_latency_start = perf_counter()
    ollama_up = True

    try:
        tags_data = await _ollama_get("/api/tags")
        tags_latency_ms = int((perf_counter() - tags_latency_start) * 1000)
    except Exception as e:  # noqa: BLE001
        ollama_up = False
        tags_latency_ms = -1
        return {
            "ollama_up": ollama_up,
            "tags_latency_ms": tags_latency_ms,
            "models_total": 0,
            "models_store_bytes": 0,
            "loaded_models": 0,
            "loaded_ram_bytes": 0,
            "loaded_vram_bytes": 0,
            "loaded_names": [],
            "actions": ACTION_STATE,
            "host_ram_gb": LOCALAI_HOST_RAM_GB,
            "cpu_logical": LOCALAI_CPU_LOGICAL,
            "cpu_physical": LOCALAI_CPU_PHYSICAL,
            "cpu_perf": LOCALAI_CPU_PERF,
            "machine": LOCALAI_MACHINE,
            "hw_model": LOCALAI_HW_MODEL,
            "gpu_backend": LOCALAI_GPU_BACKEND,
            "gpu_name": LOCALAI_GPU_NAME,
            "gpu_cores": LOCALAI_GPU_CORES,
            "num_parallel": LOCALAI_OLLAMA_NUM_PARALLEL,
            "max_loaded_models": LOCALAI_OLLAMA_MAX_LOADED_MODELS,
            "keep_alive": LOCALAI_OLLAMA_KEEP_ALIVE,
            "max_queue": LOCALAI_OLLAMA_MAX_QUEUE,
            "boost_active": LOCALAI_BOOST_ACTIVE,
            "rag_preset": LOCALAI_RAG_PRESET,
            "qdrant": qdrant,
            "web": web,
            "error": str(e),
        }

    try:
        ps_data = await _ollama_get("/api/ps")
    except Exception:
        ps_data = {"models": []}

    models = tags_data.get("models", []) if isinstance(tags_data, dict) else []
    loaded = ps_data.get("models", []) if isinstance(ps_data, dict) else []

    return {
        "ollama_up": ollama_up,
        "tags_latency_ms": tags_latency_ms,
        "models_total": len(models),
        "models_store_bytes": _sum_int(models, "size"),
        "loaded_models": len(loaded),
        "loaded_ram_bytes": _sum_int(loaded, "size"),
        "loaded_vram_bytes": _sum_int(loaded, "size_vram"),
        "loaded_names": [m.get("name", "") for m in loaded if isinstance(m, dict)],
        "actions": ACTION_STATE,
        "host_ram_gb": LOCALAI_HOST_RAM_GB,
        "cpu_logical": LOCALAI_CPU_LOGICAL,
        "cpu_physical": LOCALAI_CPU_PHYSICAL,
        "cpu_perf": LOCALAI_CPU_PERF,
        "machine": LOCALAI_MACHINE,
        "hw_model": LOCALAI_HW_MODEL,
        "gpu_backend": LOCALAI_GPU_BACKEND,
        "gpu_name": LOCALAI_GPU_NAME,
        "gpu_cores": LOCALAI_GPU_CORES,
        "num_parallel": LOCALAI_OLLAMA_NUM_PARALLEL,
        "max_loaded_models": LOCALAI_OLLAMA_MAX_LOADED_MODELS,
        "keep_alive": LOCALAI_OLLAMA_KEEP_ALIVE,
        "max_queue": LOCALAI_OLLAMA_MAX_QUEUE,
        "boost_active": LOCALAI_BOOST_ACTIVE,
        "rag_preset": LOCALAI_RAG_PRESET,
        "qdrant": qdrant,
        "web": web,
    }


@app.post("/api/models/pull")
async def pull_model(action: ModelAction) -> dict[str, Any]:
    return await _tracked_action(
        "pull",
        action.model,
        "/api/pull",
        {"model": action.model, "stream": True},
        stream=True,
    )


@app.post("/api/models/update")
async def update_model(action: ModelAction) -> dict[str, Any]:
    return await _tracked_action(
        "update",
        action.model,
        "/api/pull",
        {"model": action.model, "stream": True},
        stream=True,
    )


@app.post("/api/models/delete")
async def delete_model(action: ModelAction) -> dict[str, Any]:
    return await _tracked_action(
        "delete",
        action.model,
        "/api/delete",
        {"model": action.model},
        stream=False,
        method="DELETE",
    )


@app.post("/api/console")
async def console(req: ConsoleRequest) -> dict[str, Any]:
    start = perf_counter()

    if req.op in {"show", "generate", "pull"} and not req.model.strip():
        raise HTTPException(status_code=400, detail="model is required for this operation")
    if req.op == "generate" and not req.prompt.strip():
        raise HTTPException(status_code=400, detail="prompt is required for generate")

    if req.op == "tags":
        result = await _ollama_get("/api/tags")
    elif req.op == "ps":
        result = await _ollama_get("/api/ps")
    elif req.op == "show":
        result = await _ollama_request("POST", "/api/show", {"name": req.model.strip()})
    elif req.op == "generate":
        result = await _ollama_request(
            "POST",
            "/api/generate",
            {"model": req.model.strip(), "prompt": req.prompt, "stream": False},
        )
    elif req.op == "pull":
        result = await _ollama_stream_post("/api/pull", {"model": req.model.strip(), "stream": True})
    else:
        raise HTTPException(status_code=400, detail="unsupported operation")

    return {
        "op": req.op,
        "model": req.model.strip(),
        "duration_ms": int((perf_counter() - start) * 1000),
        "result": result,
    }


@app.post("/api/tests/smoke")
async def run_smoke_tests(req: SmokeTestRequest) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    tags: dict[str, Any] = {}
    base_model = req.model.strip()

    start = perf_counter()
    try:
        t0 = perf_counter()
        tags = await _ollama_get("/api/tags")
        checks.append({"name": "ollama_tags", "ok": True, "duration_ms": int((perf_counter() - t0) * 1000), "detail": ""})
    except Exception as e:  # noqa: BLE001
        checks.append({"name": "ollama_tags", "ok": False, "duration_ms": int((perf_counter() - start) * 1000), "detail": str(e)})
        return {
            "ok": False,
            "checks": checks,
            "summary": {
                "passed": 0,
                "failed": 1,
                "skipped": 0,
                "duration_ms": int((perf_counter() - start) * 1000),
            },
        }

    t0 = perf_counter()
    try:
        await _ollama_get("/api/ps")
        checks.append({"name": "ollama_ps", "ok": True, "duration_ms": int((perf_counter() - t0) * 1000), "detail": ""})
    except Exception as e:  # noqa: BLE001
        checks.append({"name": "ollama_ps", "ok": False, "duration_ms": int((perf_counter() - t0) * 1000), "detail": str(e)})

    models = tags.get("models", []) if isinstance(tags, dict) else []
    if not base_model and models:
        try:
            base_model = _select_benchmark_model(models, "")
        except HTTPException:
            base_model = ""

    if base_model:
        t0 = perf_counter()
        try:
            out = await _ollama_request(
                "POST",
                "/api/generate",
                {"model": base_model, "prompt": req.prompt.strip() or "Reply with exactly: ok", "stream": False},
            )
            ok = bool((out.get("response", "") or "").strip())
            checks.append(
                {
                    "name": "ollama_generate",
                    "ok": ok,
                    "duration_ms": int((perf_counter() - t0) * 1000),
                    "detail": "" if ok else "empty response",
                    "model": base_model,
                }
            )
        except Exception as e:  # noqa: BLE001
            checks.append(
                {
                    "name": "ollama_generate",
                    "ok": False,
                    "duration_ms": int((perf_counter() - t0) * 1000),
                    "detail": str(e),
                    "model": base_model,
                }
            )
    else:
        checks.append({"name": "ollama_generate", "ok": True, "duration_ms": 0, "detail": "skipped (no local model found)", "skipped": True})

    if LOCALAI_QDRANT_ENABLED:
        t0 = perf_counter()
        qdrant = await _qdrant_metrics()
        checks.append(
            {
                "name": "qdrant_health",
                "ok": bool(qdrant.get("up", False)),
                "duration_ms": int((perf_counter() - t0) * 1000),
                "detail": qdrant.get("error", ""),
            }
        )
    else:
        checks.append({"name": "qdrant_health", "ok": True, "duration_ms": 0, "detail": "skipped (qdrant disabled)", "skipped": True})

    if LOCALAI_WEB_ENABLED:
        t0 = perf_counter()
        web = await _web_metrics()
        checks.append(
            {
                "name": "web_stack_health",
                "ok": bool(web.get("searxng_up", False)) and bool(web.get("redis_up", False)),
                "duration_ms": int((perf_counter() - t0) * 1000),
                "detail": web.get("error", ""),
            }
        )
    else:
        checks.append({"name": "web_stack_health", "ok": True, "duration_ms": 0, "detail": "skipped (web disabled)", "skipped": True})

    passed = sum(1 for c in checks if c.get("ok"))
    skipped = sum(1 for c in checks if c.get("skipped"))
    failed = len(checks) - passed
    return {
        "ok": failed == 0,
        "checks": checks,
        "summary": {
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "duration_ms": int((perf_counter() - start) * 1000),
        },
    }


@app.post("/api/benchmarks/run")
async def run_benchmark(req: BenchmarkRequest) -> dict[str, Any]:
    tags = await _ollama_get("/api/tags")
    models = tags.get("models", []) if isinstance(tags, dict) else []
    model = _select_benchmark_model(models, req.model)

    samples: list[dict[str, Any]] = []
    latencies: list[float] = []
    tokens_per_second_values: list[float] = []
    failures = 0

    for i in range(req.iterations):
        t0 = perf_counter()
        try:
            out = await _ollama_request(
                "POST",
                "/api/generate",
                {"model": model, "prompt": req.prompt, "stream": False},
            )
            wall_ms = round((perf_counter() - t0) * 1000.0, 2)
            eval_count = int(out.get("eval_count", 0) or 0)
            eval_duration_ns = int(out.get("eval_duration", 0) or 0)
            token_s = round((eval_count / (eval_duration_ns / 1_000_000_000.0)), 2) if eval_duration_ns > 0 else 0.0
            latencies.append(wall_ms)
            if token_s > 0:
                tokens_per_second_values.append(token_s)
            samples.append(
                {
                    "run": i + 1,
                    "ok": True,
                    "latency_ms": wall_ms,
                    "eval_count": eval_count,
                    "eval_duration_ms": round(eval_duration_ns / 1_000_000.0, 2),
                    "total_duration_ms": round(int(out.get("total_duration", 0) or 0) / 1_000_000.0, 2),
                    "tokens_per_second": token_s,
                }
            )
        except Exception as e:  # noqa: BLE001
            failures += 1
            samples.append(
                {
                    "run": i + 1,
                    "ok": False,
                    "latency_ms": round((perf_counter() - t0) * 1000.0, 2),
                    "error": str(e),
                }
            )

    return {
        "model": model,
        "summary": {
            "iterations": req.iterations,
            "successful_runs": len(latencies),
            "failed_runs": failures,
            "latency_ms_avg": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
            "latency_ms_p50": round(_percentile(latencies, 50), 2) if latencies else 0.0,
            "latency_ms_p95": round(_percentile(latencies, 95), 2) if latencies else 0.0,
            "tokens_per_second_avg": round(sum(tokens_per_second_values) / len(tokens_per_second_values), 2)
            if tokens_per_second_values
            else 0.0,
        },
        "samples": samples,
    }


@app.post("/api/vision/analyze")
async def vision_analyze(req: VisionAnalyzeRequest) -> dict[str, Any]:
    _require_vision_enabled()
    model = _resolve_vision_model(req.model)
    _validate_vision_image_input(req.image_base64, req.image_url)
    image_b64 = await _resolve_image_base64(req.image_base64, req.image_url)
    started = perf_counter()
    result = await _vision_infer(model, req.prompt.strip() or "Describe the image in one paragraph.", image_b64, 60.0)
    return {
        "ok": bool(result["response"]),
        "status": "ready",
        "model": model,
        "prompt": req.prompt,
        "image_source": "base64" if req.image_base64.strip() else "url",
        "max_image_mb": LOCALAI_VISION_MAX_IMAGE_MB,
        "duration_ms": int((perf_counter() - started) * 1000),
        "response": result["response"],
        "eval_count": result["eval_count"],
        "eval_duration_ms": result["eval_duration_ms"],
        "total_duration_ms": result["total_duration_ms"],
        "tokens_per_second": result["tokens_per_second"],
        "raw": result["raw"],
    }


@app.post("/api/tests/vision-smoke")
async def run_vision_smoke(req: VisionSmokeRequest) -> dict[str, Any]:
    _require_vision_enabled()
    model = _resolve_vision_model(req.model)
    _validate_vision_image_input(req.image_base64, req.image_url)
    image_b64 = await _resolve_image_base64(req.image_base64, req.image_url)
    started = perf_counter()
    checks: list[dict[str, Any]] = []
    try:
        out = await _vision_infer(model, req.prompt.strip() or "Return exactly: ok", image_b64, req.timeout_seconds)
        ok = bool(out["response"])
        checks.append(
            {
                "name": "vision_inference",
                "ok": ok,
                "duration_ms": int((perf_counter() - started) * 1000),
                "detail": "" if ok else "empty response",
                "model": model,
            }
        )
    except Exception as e:  # noqa: BLE001
        checks.append(
            {
                "name": "vision_inference",
                "ok": False,
                "duration_ms": int((perf_counter() - started) * 1000),
                "detail": str(e),
                "model": model,
            }
        )

    passed = sum(1 for c in checks if c.get("ok"))
    failed = len(checks) - passed
    return {
        "ok": failed == 0,
        "status": "ready",
        "summary": {
            "passed": passed,
            "failed": failed,
            "skipped": 0,
            "duration_ms": int((perf_counter() - started) * 1000),
        },
        "checks": checks,
        "dataset_hint": LOCALAI_VISION_BENCHMARK_DATASET,
    }


@app.post("/api/tests/vision-benchmark")
async def run_vision_benchmark(req: VisionBenchmarkRequest) -> dict[str, Any]:
    _require_vision_enabled()
    model = _resolve_vision_model(req.model)
    path = _resolve_dataset_path(req.dataset_path)
    started = perf_counter()

    rows: list[dict[str, Any]] = []
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        s = line.strip()
        if not s:
            continue
        try:
            row = json.loads(s)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"invalid JSONL at line {idx}: {e}") from e
        if not isinstance(row, dict):
            raise HTTPException(status_code=400, detail=f"invalid JSONL object at line {idx}")
        rows.append(row)

    if not rows:
        raise HTTPException(status_code=400, detail="benchmark dataset is empty")

    target = rows[: req.iterations]
    samples: list[dict[str, Any]] = []
    latencies: list[float] = []
    token_rates: list[float] = []
    failures = 0
    passes = 0

    for i, row in enumerate(target):
        prompt = str(row.get("prompt", "")).strip() or "Describe the image in one paragraph."
        expected_contains = str(row.get("expected_contains", "")).strip()
        image_b64 = str(row.get("image_base64", "")).strip()
        image_url = str(row.get("image_url", "")).strip()

        if not image_b64 and not image_url:
            failures += 1
            samples.append(
                {
                    "run": i + 1,
                    "ok": False,
                    "latency_ms": 0.0,
                    "score_pass": False,
                    "error": "dataset row missing image_base64/image_url",
                }
            )
            continue

        t0 = perf_counter()
        try:
            normalized = await _resolve_image_base64(image_b64, image_url)
            out = await _vision_infer(model, prompt, normalized, req.timeout_seconds)
            latency_ms = round((perf_counter() - t0) * 1000.0, 2)
            response = out["response"]
            score_pass = bool(response)
            if expected_contains:
                score_pass = expected_contains.lower() in response.lower()
            if score_pass:
                passes += 1
            latencies.append(latency_ms)
            if out["tokens_per_second"] > 0:
                token_rates.append(float(out["tokens_per_second"]))
            samples.append(
                {
                    "run": i + 1,
                    "ok": True,
                    "latency_ms": latency_ms,
                    "score_pass": score_pass,
                    "expected_contains": expected_contains,
                    "response": response,
                    "eval_count": out["eval_count"],
                    "eval_duration_ms": out["eval_duration_ms"],
                    "total_duration_ms": out["total_duration_ms"],
                    "tokens_per_second": out["tokens_per_second"],
                }
            )
        except Exception as e:  # noqa: BLE001
            failures += 1
            samples.append(
                {
                    "run": i + 1,
                    "ok": False,
                    "latency_ms": round((perf_counter() - t0) * 1000.0, 2),
                    "score_pass": False,
                    "error": str(e),
                }
            )

    executed = len(samples)
    return {
        "ok": failures == 0,
        "status": "ready",
        "model": model,
        "dataset_path": str(path),
        "summary": {
            "iterations_requested": req.iterations,
            "iterations_executed": executed,
            "successful_runs": executed - failures,
            "failed_runs": failures,
            "latency_ms_avg": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
            "latency_ms_p95": round(_percentile(latencies, 95), 2) if latencies else 0.0,
            "tokens_per_second_avg": round(sum(token_rates) / len(token_rates), 2) if token_rates else 0.0,
            "score_passed": passes,
            "score_total": executed,
            "score_pass_rate": round((passes / executed) * 100.0, 2) if executed else 0.0,
            "duration_ms": int((perf_counter() - started) * 1000),
        },
        "samples": samples,
        "message": "vision benchmark completed",
    }


@app.get("/api/image-gen/health")
async def image_gen_health() -> dict[str, Any]:
    if not LOCALAI_IMAGE_GEN_ENABLED:
        return {
            "enabled": False,
            "provider": LOCALAI_IMAGE_GEN_PROVIDER,
            "backend_url": LOCALAI_IMAGE_GEN_BACKEND_URL,
            "ready": False,
            "status": "disabled",
            "message": "image generation lane disabled",
        }

    started = perf_counter()
    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            resp = await client.get(f"{LOCALAI_IMAGE_GEN_BACKEND_URL.rstrip('/')}/healthz")
        latency_ms = int((perf_counter() - started) * 1000)
        if resp.status_code >= 400:
            return {
                "enabled": True,
                "provider": LOCALAI_IMAGE_GEN_PROVIDER,
                "backend_url": LOCALAI_IMAGE_GEN_BACKEND_URL,
                "ready": False,
                "status": "down",
                "latency_ms": latency_ms,
                "message": f"health endpoint returned status {resp.status_code}",
            }
        payload = resp.json() if resp.content else {}
        return {
            "enabled": True,
            "provider": LOCALAI_IMAGE_GEN_PROVIDER,
            "backend_url": LOCALAI_IMAGE_GEN_BACKEND_URL,
            "ready": True,
            "status": "ready",
            "latency_ms": latency_ms,
            "details": payload,
        }
    except Exception as e:  # noqa: BLE001
        return {
            "enabled": True,
            "provider": LOCALAI_IMAGE_GEN_PROVIDER,
            "backend_url": LOCALAI_IMAGE_GEN_BACKEND_URL,
            "ready": False,
            "status": "down",
            "message": str(e),
        }




def _render_layout(title: str, subtitle: str, active_tab: str, content: str, script: str) -> str:
    def tab(path: str, label: str, key: str) -> str:
        active = "tab active" if active_tab == key else "tab"
        return f'<a class="{active}" href="{path}">{label}</a>'

    nav = "".join(
        [
            tab("/runtime", "Hardware/Runtime", "runtime"),
            tab("/models", "Model Management", "models"),
            tab("/tests", "Tests/Benchmarking", "tests"),
        ]
    )

    return f"""<!doctype html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>{title}</title>
    <style>
      :root {{
        --bg: #0a0a0a;
        --surface: #111111;
        --surface-2: #151515;
        --border: #262626;
        --text: #f3f4f6;
        --muted: #a3a3a3;
      }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        color: var(--text);
        font-family: Inter, ui-sans-serif, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
        background: #090909;
      }}
      .page {{ max-width: 1160px; margin: 20px auto; padding: 0 14px; }}
      .shell {{ border: 1px solid var(--border); border-radius: 12px; background: var(--surface); padding: 16px; }}
      .top {{ display: flex; align-items: center; justify-content: space-between; gap: 16px; }}
      .title {{ margin: 0; font-size: 36px; font-weight: 700; }}
      .subtitle {{ margin: 5px 0 0; color: var(--muted); }}
      .tabs {{ display: flex; gap: 8px; flex-wrap: wrap; margin-top: 12px; margin-bottom: 14px; }}
      .tab {{
        border: 1px solid var(--border);
        border-radius: 999px;
        padding: 7px 12px;
        text-decoration: none;
        color: var(--muted);
        background: var(--surface-2);
      }}
      .tab.active {{ color: var(--text); border-color: #3b82f6; }}
      .status {{ display: inline-flex; align-items: center; gap: 8px; border: 1px solid var(--border); border-radius: 999px; padding: 7px 12px; background: var(--surface-2); color: var(--muted); }}
      .dot {{ width: 8px; height: 8px; border-radius: 50%; background: #ef4444; }}
      .dot.up {{ background: #22c55e; }}
      .grid {{ margin-top: 12px; display: grid; gap: 8px; grid-template-columns: repeat(4, minmax(140px, 1fr)); }}
      .card {{ border: 1px solid var(--border); border-radius: 10px; background: #121212; padding: 10px; }}
      .label {{ font-size: 11px; color: var(--muted); margin-bottom: 4px; }}
      .value {{ font-size: 29px; font-weight: 650; line-height: 1.15; }}
      .subvalue {{ margin-top: 2px; font-size: 11px; color: var(--muted); min-height: 14px; }}
      .section-title {{ margin-top: 18px; padding-top: 12px; border-top: 1px solid var(--border); font-size: 21px; font-weight: 700; color: #e5e7eb; }}
      .hwbar {{ margin-top: 12px; border: 1px solid var(--border); border-radius: 10px; background: #101010; padding: 9px 11px; display: flex; flex-wrap: wrap; gap: 12px; }}
      .hwitem {{ color: var(--muted); font-size: 12px; }}
      .hwitem b {{ color: var(--text); font-weight: 600; }}
      .spark {{ margin-top: 7px; width: 100%; height: 28px; display: block; }}
      .spark path.grid {{ stroke: #2b2b2b; stroke-width: 1; }}
      .spark polyline.line {{ fill: none; stroke: #d4d4d8; stroke-width: 1.8; stroke-linecap: round; stroke-linejoin: round; }}
      .controls {{ margin-top: 12px; display: flex; gap: 8px; flex-wrap: wrap; }}
      input, textarea, select, button {{ border-radius: 8px; border: 1px solid var(--border); color: var(--text); background: #0f0f10; padding: 9px 11px; }}
      input {{ width: min(360px, 100%); }}
      textarea {{ width: 100%; resize: vertical; }}
      button {{ cursor: pointer; }}
      button.ok {{ border-color: #14532d; background: #0d1f12; }}
      button.warn {{ border-color: #78350f; background: #251605; }}
      button.danger {{ border-color: #7f1d1d; background: #2a0a0a; }}
      .catalog {{ margin-top: 12px; border: 1px solid var(--border); border-radius: 10px; padding: 10px; background: #101010; }}
      .catalog-head {{ display: flex; align-items: center; justify-content: space-between; gap: 8px; margin-bottom: 8px; }}
      .catalog-head h2 {{ margin: 0; font-size: 15px; }}
      .catalog-tools {{ display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }}
      .catalog-table {{ width: 100%; border-collapse: collapse; }}
      .catalog-table th, .catalog-table td {{ text-align: left; border-bottom: 1px solid var(--border); padding: 7px; vertical-align: top; }}
      .catalog-table th {{ color: var(--muted); font-weight: 500; font-size: 12px; }}
      .chip {{ display: inline-flex; align-items: center; justify-content: center; border: 1px solid var(--border); color: var(--muted); padding: 1px 9px; min-height: 24px; border-radius: 999px; font-size: 11px; line-height: 1; }}
      .chip.installed {{ color: #86efac; border-color: #14532d; }}
      .chip.class {{ color: #c4b5fd; border-color: #4c1d95; }}
      .chip.fit-great {{ color: #86efac; border-color: #166534; }}
      .chip.fit-good {{ color: #bef264; border-color: #4d7c0f; }}
      .chip.fit-tight {{ color: #fcd34d; border-color: #92400e; }}
      .chip.fit-heavy {{ color: #fca5a5; border-color: #991b1b; }}
      .advanced {{ margin-top: 12px; border: 1px solid var(--border); border-radius: 10px; background: #0e0e0f; padding: 10px; }}
      .advanced summary {{ cursor: pointer; font-size: 14px; font-weight: 600; color: #d4d4d8; }}
      .advanced-body {{ margin-top: 10px; }}
      .advanced-controls {{ display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }}
      table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
      th, td {{ text-align: left; border-bottom: 1px solid var(--border); padding: 8px 7px; }}
      th {{ color: var(--muted); font-weight: 500; font-size: 12px; }}
      pre {{ margin-top: 12px; margin-bottom: 0; min-height: 72px; border: 1px solid var(--border); border-radius: 10px; background: #0c0c0d; color: #d4d4d8; padding: 11px; overflow: auto; white-space: pre-wrap; }}
      .muted {{ color: var(--muted); }}
      @media (max-width: 960px) {{ .grid {{ grid-template-columns: repeat(2, minmax(140px, 1fr)); }} }}
      @media (max-width: 560px) {{ .top {{ flex-direction: column; align-items: flex-start; }} .grid {{ grid-template-columns: 1fr; }} }}
    </style>
  </head>
  <body>
    <div class=\"page\">
      <div class=\"shell\">
        <div class=\"top\">
          <div>
            <h1 class=\"title\">{title}</h1>
            <p class=\"subtitle\">{subtitle}</p>
          </div>
          <div class=\"status\"><span id=\"status-dot\" class=\"dot\"></span><span id=\"status-text\">Checking Ollama...</span></div>
        </div>
        <div class=\"tabs\">{nav}</div>
        {content}
      </div>
    </div>
    <script>
      async function refreshHeaderStatus() {{
        try {{
          const r = await fetch('/api/metrics');
          const m = await r.json();
          const dot = document.getElementById('status-dot');
          const text = document.getElementById('status-text');
          if (!dot || !text) return;
          dot.className = m.ollama_up ? 'dot up' : 'dot';
          text.textContent = m.ollama_up ? 'Ollama online' : 'Ollama unavailable';
        }} catch (e) {{
          const text = document.getElementById('status-text');
          if (text) text.textContent = 'Metrics unavailable';
        }}
      }}
      refreshHeaderStatus();
      setInterval(refreshHeaderStatus, 3000);
    </script>
    {script}
  </body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return _runtime_page_html()


@app.get("/runtime", response_class=HTMLResponse)
async def runtime_page() -> str:
    return _runtime_page_html()


def _runtime_page_html() -> str:
    content = """
      <div class=\"hwbar\">
        <div class=\"hwitem\"><b>CPU:</b> <span id=\"hw-cpu\">-</span></div>
        <div class=\"hwitem\"><b>GPU:</b> <span id=\"hw-gpu\">-</span></div>
        <div class=\"hwitem\"><b>System RAM:</b> <span id=\"hw-ram\">-</span></div>
        <div class=\"hwitem\"><b>Runtime:</b> <span id=\"hw-runtime\">-</span></div>
      </div>
      <div class=\"hwbar\">
        <div class=\"hwitem\"><b>RAG Preset:</b> <span id=\"rag-preset\">-</span></div>
        <div class=\"hwitem\"><b>Qdrant:</b> <span id=\"rag-qdrant\">-</span></div>
        <div class=\"hwitem\"><b>Collections:</b> <span id=\"rag-collections\">-</span></div>
        <div class=\"hwitem\"><b>Vectors:</b> <span id=\"rag-vectors\">-</span></div>
        <div class=\"hwitem\"><b>Qdrant Latency:</b> <span id=\"rag-latency\">-</span></div>
      </div>
      <div class=\"hwbar\">
        <div class=\"hwitem\"><b>Web Search:</b> <span id=\"web-enabled\">-</span></div>
        <div class=\"hwitem\"><b>SearxNG:</b> <span id=\"web-searxng\">-</span></div>
        <div class=\"hwitem\"><b>SearxNG Latency:</b> <span id=\"web-latency\">-</span></div>
        <div class=\"hwitem\"><b>Last Results:</b> <span id=\"web-results\">-</span></div>
        <div class=\"hwitem\"><b>Search Errors:</b> <span id=\"web-errors\">-</span></div>
        <div class=\"hwitem\"><b>Redis:</b> <span id=\"web-redis\">-</span></div>
      </div>

      <div class=\"grid\">
        <div class=\"card\"><div class=\"label\">Stored Models</div><div id=\"m-models\" class=\"value\">-</div><div class=\"subvalue\">count</div><svg id=\"spark-models\" class=\"spark\" viewBox=\"0 0 160 28\" preserveAspectRatio=\"none\"></svg></div>
        <div class=\"card\"><div class=\"label\">Model Store Size</div><div id=\"m-store\" class=\"value\">-</div><div class=\"subvalue\">disk footprint</div><svg id=\"spark-store\" class=\"spark\" viewBox=\"0 0 160 28\" preserveAspectRatio=\"none\"></svg></div>
        <div class=\"card\"><div class=\"label\">Loaded Models</div><div id=\"m-loaded\" class=\"value\">-</div><div id=\"m-loaded-sub\" class=\"subvalue\">none</div><svg id=\"spark-loaded\" class=\"spark\" viewBox=\"0 0 160 28\" preserveAspectRatio=\"none\"></svg></div>
        <div class=\"card\"><div class=\"label\">API Latency</div><div id=\"m-latency\" class=\"value\">-</div><div id=\"m-latency-sub\" class=\"subvalue\">/api/tags</div><svg id=\"spark-latency\" class=\"spark\" viewBox=\"0 0 160 28\" preserveAspectRatio=\"none\"></svg></div>
        <div class=\"card\"><div class=\"label\">Loaded RAM</div><div id=\"m-ram\" class=\"value\">-</div><div id=\"m-ram-sub\" class=\"subvalue\">model resident set</div><svg id=\"spark-ram\" class=\"spark\" viewBox=\"0 0 160 28\" preserveAspectRatio=\"none\"></svg></div>
        <div class=\"card\"><div class=\"label\">Loaded VRAM</div><div id=\"m-vram\" class=\"value\">-</div><div id=\"m-vram-sub\" class=\"subvalue\">accelerator memory</div><svg id=\"spark-vram\" class=\"spark\" viewBox=\"0 0 160 28\" preserveAspectRatio=\"none\"></svg></div>
        <div class=\"card\"><div class=\"label\">Active Jobs</div><div id=\"m-jobs\" class=\"value\">-</div><div id=\"m-jobs-sub\" class=\"subvalue\">pull/update/delete</div><svg id=\"spark-jobs\" class=\"spark\" viewBox=\"0 0 160 28\" preserveAspectRatio=\"none\"></svg></div>
        <div class=\"card\"><div class=\"label\">Last Action</div><div id=\"m-last\" class=\"value\" style=\"font-size:16px\">-</div><div id=\"m-last-sub\" class=\"subvalue\"></div></div>
      </div>
    """

    script = """
    <script>
      const MAX_POINTS = 80;
      const history = { models: [], store: [], loaded: [], latency: [], ram: [], vram: [], jobs: [] };
      function bytes(n) { if (!n && n !== 0) return '-'; const u=['B','KB','MB','GB','TB']; let i=0,v=Number(n); while(v>=1024&&i<u.length-1){v/=1024;i+=1;} return `${v.toFixed(2)} ${u[i]}`; }
      function percent(part, whole) { const p=Number(part), w=Number(whole); if (!Number.isFinite(p)||!Number.isFinite(w)||w<=0) return '-'; return `${Math.max(0,(p/w)*100).toFixed(1)}%`; }
      function setText(id, value) { const el=document.getElementById(id); if (el) el.textContent=value; }
      function pushMetric(key, value) { const arr=history[key]; if(!arr) return; const v=Number(value); if(!Number.isFinite(v)) return; arr.push(v); if(arr.length>MAX_POINTS) arr.shift(); }
      function renderSpark(id, values) {
        const svg=document.getElementById(id); if(!svg) return;
        if(!values||values.length<2){ svg.innerHTML='<path class="grid" d="M0 27 H160" />'; return; }
        const w=160,h=28; let min=Math.min(...values), max=Math.max(...values); if(max===min){min-=1;max+=1;}
        const pts=values.map((v,i)=>{ const x=(i/(values.length-1))*(w-2)+1; const y=h-2-((v-min)/(max-min))*(h-6); return `${x.toFixed(1)},${y.toFixed(1)}`; }).join(' ');
        svg.innerHTML=`<path class="grid" d="M0 27 H160" /><polyline class="line" points="${pts}" />`;
      }
      function renderSparks(){ renderSpark('spark-models',history.models); renderSpark('spark-store',history.store); renderSpark('spark-loaded',history.loaded); renderSpark('spark-latency',history.latency); renderSpark('spark-ram',history.ram); renderSpark('spark-vram',history.vram); renderSpark('spark-jobs',history.jobs); }

      async function refreshMetrics() {
        const r=await fetch('/api/metrics'); const m=await r.json();
        const hostRamGb = m.host_ram_gb || 0;
        const hostRamBytes = hostRamGb * 1024 * 1024 * 1024;
        const loadedNames = (m.loaded_names || []).join(', ') || 'none';
        setText('m-models', String(m.models_total ?? '-'));
        setText('m-store', bytes(m.models_store_bytes));
        setText('m-loaded', String(m.loaded_models ?? '-'));
        setText('m-loaded-sub', `${m.loaded_models ?? 0} / ${m.max_loaded_models ?? '-'} max · ${loadedNames}`);
        setText('m-latency', m.tags_latency_ms >= 0 ? `${m.tags_latency_ms} ms` : '-');
        setText('m-latency-sub', `/api/tags · parallel ${m.num_parallel ?? '-'}`);
        setText('m-ram', bytes(m.loaded_ram_bytes));
        setText('m-ram-sub', `model resident set · ${percent(m.loaded_ram_bytes, hostRamBytes)} of host`);
        setText('m-vram', bytes(m.loaded_vram_bytes));
        setText('m-vram-sub', `accelerator memory · ${percent(m.loaded_vram_bytes, hostRamBytes)} of host (est)`);
        setText('m-jobs', `${m.actions?.active_jobs ?? 0} active / ${m.actions?.completed_jobs ?? 0} done`);
        setText('m-jobs-sub', `queue cap ${m.max_queue ?? '-'}`);
        setText('m-last-sub', `keep_alive ${m.keep_alive ?? '-'} · boost ${m.boost_active ? 'on' : 'off'}`);
        setText('hw-cpu', `${m.cpu_logical ?? '-'} logical / ${m.cpu_physical ?? '-'} physical (${m.cpu_perf ?? '-'} perf) · ${m.machine || 'arm64'}`);
        setText('hw-gpu', `${m.gpu_name || 'Apple Silicon GPU'} (${m.gpu_backend || 'Metal'}${m.gpu_cores && m.gpu_cores !== 'unknown' ? `, ${m.gpu_cores} cores` : ''})`);
        setText('hw-ram', `${hostRamGb || '-'} GB · ${m.hw_model || 'model unknown'}`);
        setText('hw-runtime', `parallel ${m.num_parallel ?? '-'} · max loaded ${m.max_loaded_models ?? '-'} · queue ${m.max_queue ?? '-'} · keep_alive ${m.keep_alive ?? '-'}${m.boost_active ? ' · boost ON' : ''}`);

        const q = m.qdrant || {}; const qEnabled = q.enabled !== false; const qUp = qEnabled && q.up;
        setText('rag-preset', (m.rag_preset || '-').toLowerCase());
        setText('rag-qdrant', !qEnabled ? 'disabled' : (qUp ? 'online' : 'unavailable'));
        setText('rag-collections', qEnabled ? String(q.collections ?? '-') : '-');
        setText('rag-vectors', qEnabled ? `${q.indexed_vectors_total ?? 0} indexed / ${q.points_total ?? 0} total` : '-');
        setText('rag-latency', qEnabled ? (q.latency_ms >= 0 ? `${q.latency_ms} ms` : '-') : '-');

        const w = m.web || {}; const webEnabled = w.enabled === true;
        setText('web-enabled', webEnabled ? 'enabled' : 'disabled');
        setText('web-searxng', webEnabled ? (w.searxng_up ? 'online' : 'unavailable') : '-');
        setText('web-latency', webEnabled ? (w.searxng_latency_ms >= 0 ? `${w.searxng_latency_ms} ms` : '-') : '-');
        setText('web-results', webEnabled ? (w.last_results_count >= 0 ? String(w.last_results_count) : '-') : '-');
        setText('web-errors', webEnabled ? `${w.search_error_rate ?? 0}% (${w.search_err ?? 0}/${(w.search_ok ?? 0) + (w.search_err ?? 0)})` : '-');
        const redisMem=(w.redis_memory_pct>=0 && w.redis_used_memory_mb>=0)?`${w.redis_memory_pct}% (${w.redis_used_memory_mb}MB/${w.redis_maxmemory_mb}MB)`:'-';
        setText('web-redis', webEnabled ? ((w.redis_up ? 'online' : 'unavailable') + ` · ${redisMem}`) : '-');

        pushMetric('models', m.models_total ?? 0); pushMetric('store', m.models_store_bytes ?? 0); pushMetric('loaded', m.loaded_models ?? 0);
        pushMetric('latency', m.tags_latency_ms >= 0 ? m.tags_latency_ms : 0); pushMetric('ram', m.loaded_ram_bytes ?? 0); pushMetric('vram', m.loaded_vram_bytes ?? 0); pushMetric('jobs', m.actions?.active_jobs ?? 0);
        renderSparks();

        if (m.actions?.last_action && m.actions?.last_model) setText('m-last', `${m.actions.last_action} ${m.actions.last_model} (${m.actions.last_duration_ms} ms)`);
        else setText('m-last', '-');
      }
      refreshMetrics();
      setInterval(refreshMetrics, 1500);
    </script>
    """
    return _render_layout("LocalAI Runtime", "Hardware and live runtime statistics.", "runtime", content, script)


@app.get("/models", response_class=HTMLResponse)
async def models_page() -> str:
    content = """
      <div class=\"section-title\">Local Model Actions and Inventory</div>
      <div class=\"controls\">
        <input id=\"model\" placeholder=\"e.g. llama3.2:3b\" />
        <button class=\"ok\" onclick=\"run('pull')\">Pull</button>
        <button class=\"warn\" onclick=\"run('update')\">Update</button>
        <button class=\"danger\" onclick=\"run('delete')\">Delete Local</button>
        <button onclick=\"refreshAll()\">Refresh</button>
      </div>
      <pre id=\"log\">Ready.</pre>

      <details class=\"advanced\">
        <summary>Advanced Ollama Console (Debug)</summary>
        <div class=\"advanced-body\">
          <div class=\"advanced-controls\">
            <select id=\"console-op\">
              <option value=\"tags\">tags</option>
              <option value=\"ps\">ps</option>
              <option value=\"show\">show</option>
              <option value=\"generate\" selected>generate</option>
              <option value=\"pull\">pull</option>
            </select>
            <input id=\"console-model\" placeholder=\"model (required for show/generate/pull)\" />
            <button id=\"console-run\" onclick=\"runConsole()\">Run</button>
          </div>
          <textarea id=\"console-prompt\" rows=\"4\" placeholder=\"prompt (required for generate)\"></textarea>
          <pre id=\"console-log\">Console idle.</pre>
        </div>
      </details>

      <table>
        <thead><tr><th>Model</th><th>Size</th><th>Modified</th></tr></thead>
        <tbody id=\"models\"></tbody>
      </table>

      <div class=\"catalog\">
        <div class=\"catalog-head\">
          <h2>Browse Catalog</h2>
          <div class=\"catalog-tools\">
            <input id=\"catalog-q\" placeholder=\"Search models (llama, qwen, mistral...)\" />
            <select id=\"catalog-class\">
              <option value=\"all\">All Classes</option><option value=\"chat\">Chat</option><option value=\"reasoning\">Reasoning</option>
              <option value=\"code\">Code</option><option value=\"embed\">Embed</option><option value=\"vision\">Vision</option>
            </select>
            <select id=\"catalog-fit\"><option value=\"all\">All Fit</option><option value=\"great\">Great</option><option value=\"good\">Good</option><option value=\"tight\">Tight</option><option value=\"heavy\">Heavy</option></select>
            <button onclick=\"refreshCatalog()\">Search</button>
          </div>
        </div>
        <table class=\"catalog-table\">
          <thead><tr><th>Model</th><th>Recommended Pull</th><th>Class</th><th>Fit For This Mac</th><th>Status</th><th>Source</th><th></th></tr></thead>
          <tbody id=\"catalog\"></tbody>
        </table>
      </div>
    """

    script = """
    <script>
      const modelInput = document.getElementById('model');
      const logBox = document.getElementById('log');
      const modelsTable = document.getElementById('models');
      const catalogTable = document.getElementById('catalog');
      const catalogInput = document.getElementById('catalog-q');
      const catalogClass = document.getElementById('catalog-class');
      const catalogFit = document.getElementById('catalog-fit');
      const consoleOp = document.getElementById('console-op');
      const consoleModel = document.getElementById('console-model');
      const consolePrompt = document.getElementById('console-prompt');
      const consoleLog = document.getElementById('console-log');
      let catalogCache = [];
      let catalogTimer = null;

      function bytes(n){ if(!n&&n!==0) return '-'; const u=['B','KB','MB','GB','TB']; let i=0,v=Number(n); while(v>=1024&&i<u.length-1){v/=1024;i+=1;} return `${v.toFixed(2)} ${u[i]}`; }
      function fmtDate(ts){ if(!ts) return '-'; const d=new Date(ts); if(Number.isNaN(d.getTime())) return '-'; return `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,'0')}-${String(d.getDate()).padStart(2,'0')}`; }
      function esc(s){ return String(s ?? '').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;').replaceAll("'",'&#39;'); }
      function fitLabel(fit, est){ const name=fit||'unknown'; return `${name} (${(est || 0).toFixed(1)} GB est)`; }

      function renderCatalogRows() {
        const q = catalogInput.value.trim().toLowerCase();
        const cls = catalogClass.value;
        const fit = catalogFit.value;
        const filtered = catalogCache.filter((item) => {
          if (q && !(item.name || '').includes(q)) return false;
          if (cls !== 'all' && item.model_class !== cls) return false;
          if (fit !== 'all' && item.fit !== fit) return false;
          return true;
        });
        const rows = filtered.map((item) => {
          const installedChip = item.installed ? `<span class="chip installed">installed</span>` : '<span class="chip">not installed</span>';
          const sourceChip = item.source === 'curated' ? '<span class="chip">curated</span>' : '<span class="chip">library</span>';
          const classChip = `<span class="chip class">${esc(item.model_class)}</span>`;
          const fitChip = `<span class="chip fit-${esc(item.fit)}">${esc(fitLabel(item.fit, item.est_ram_gb || 0))}</span>`;
          const desc = item.description ? `<div class="muted">${esc(item.description)}</div>` : '';
          return `<tr><td><div>${esc(item.name)}</div>${desc}</td><td><code>${esc(item.pull_ref)}</code></td><td>${classChip}</td><td>${fitChip}</td><td>${installedChip}</td><td>${sourceChip}</td><td><button class="ok" onclick="run('pull', '${esc(item.pull_ref)}')">Pull</button></td></tr>`;
        });
        catalogTable.innerHTML = rows.join('') || '<tr><td colspan="7" class="muted">No matching catalog results.</td></tr>';
      }

      async function refreshModels() {
        const r = await fetch('/api/models');
        const data = await r.json();
        const rows = (data.models || []).map((m) => `<tr><td>${esc(m.name || '-')}</td><td>${bytes(m.size)}</td><td title="${esc(m.modified_at || '')}">${fmtDate(m.modified_at)}</td></tr>`);
        modelsTable.innerHTML = rows.join('') || '<tr><td colspan="3" class="muted">No models found.</td></tr>';
      }

      async function refreshCatalog() {
        const q = encodeURIComponent(catalogInput.value.trim());
        const r = await fetch(`/api/catalog?q=${q}&limit=80`);
        const data = await r.json();
        catalogCache = data.items || [];
        renderCatalogRows();
      }

      async function runConsole() {
        const op = (consoleOp.value || '').trim();
        const model = (consoleModel.value || '').trim();
        const prompt = consolePrompt.value || '';
        consoleLog.textContent = `Running ${op}...`;
        try {
          const r = await fetch('/api/console', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ op, model, prompt }) });
          const data = await r.json();
          if (!r.ok) throw new Error(JSON.stringify(data));
          consoleLog.textContent = JSON.stringify(data, null, 2);
          if (model && !modelInput.value.trim()) modelInput.value = model;
        } catch (e) {
          consoleLog.textContent = `console failed: ${e}`;
        }
      }

      async function run(action, overrideModel = null) {
        const model = (overrideModel || modelInput.value).trim();
        if (!model) { logBox.textContent = 'Enter a model name first.'; return; }
        modelInput.value = model;
        logBox.textContent = `${action} ${model}...`;
        try {
          const r = await fetch(`/api/models/${action}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ model }) });
          const data = await r.json();
          if (!r.ok) throw new Error(JSON.stringify(data));
          logBox.textContent = JSON.stringify(data, null, 2);
          await refreshAll();
        } catch (e) {
          logBox.textContent = `${action} failed: ${e}`;
        }
      }

      async function refreshAll(){ try { await Promise.all([refreshModels(), refreshCatalog()]); } catch (e) { logBox.textContent = `Refresh failed: ${e}`; } }
      function updateConsoleForm(){ const op = consoleOp.value; const needsPrompt = op === 'generate'; const needsModel = op === 'show' || op === 'generate' || op === 'pull'; consolePrompt.style.display = needsPrompt ? 'block' : 'none'; consoleModel.placeholder = needsModel ? 'model (required for this operation)' : 'model (optional)'; }
      catalogInput.addEventListener('input', () => { if (catalogTimer) clearTimeout(catalogTimer); catalogTimer = setTimeout(() => refreshCatalog().catch((e) => { logBox.textContent = `Catalog refresh failed: ${e}`; }), 250); });
      catalogClass.addEventListener('change', renderCatalogRows);
      catalogFit.addEventListener('change', renderCatalogRows);
      consoleOp.addEventListener('change', updateConsoleForm);
      updateConsoleForm();
      refreshAll();
      setInterval(refreshModels, 10000);
      setInterval(refreshCatalog, 30000);
    </script>
    """
    return _render_layout("LocalAI Model Admin", "Model lifecycle operations, inventory, and catalog browse.", "models", content, script)


@app.get("/tests", response_class=HTMLResponse)
async def tests_page() -> str:
    content = """
      <div class=\"section-title\">Unit Tests / Smoke Checks</div>
      <div class=\"controls\">
        <input id=\"smoke-model\" placeholder=\"Optional model override (e.g. llama3.2:3b)\" />
        <button class=\"ok\" onclick=\"runSmoke()\">Run Smoke Tests</button>
        <span id=\"smoke-overall\" class=\"chip\">idle</span>
      </div>

      <div class=\"grid\">
        <div class=\"card\"><div class=\"label\">Passed</div><div id=\"s-passed\" class=\"value\">-</div></div>
        <div class=\"card\"><div class=\"label\">Failed</div><div id=\"s-failed\" class=\"value\">-</div></div>
        <div class=\"card\"><div class=\"label\">Skipped</div><div id=\"s-skipped\" class=\"value\">-</div></div>
        <div class=\"card\"><div class=\"label\">Duration</div><div id=\"s-duration\" class=\"value\">-</div></div>
      </div>

      <table>
        <thead><tr><th>Check</th><th>Status</th><th>Duration</th><th>Model</th><th>Detail</th></tr></thead>
        <tbody id=\"smoke-results\"><tr><td colspan=\"5\" class=\"muted\">No smoke run yet.</td></tr></tbody>
      </table>

      <details class=\"advanced\">
        <summary>Raw Smoke JSON</summary>
        <pre id=\"smoke-log\">Smoke tests idle.</pre>
      </details>

      <div class=\"section-title\">Benchmarking</div>
      <div class=\"controls\">
        <input id=\"bench-model\" placeholder=\"Optional model override (e.g. llama3.2:3b)\" />
        <input id=\"bench-iters\" type=\"number\" min=\"1\" max=\"40\" value=\"5\" placeholder=\"iterations\" />
      </div>
      <div class=\"controls\">
        <textarea id=\"bench-prompt\" rows=\"3\">Summarize why local-first AI stacks are useful in one sentence.</textarea>
      </div>
      <div class=\"controls\">
        <button class=\"warn\" onclick=\"runBenchmark()\">Run Benchmark</button>
        <span id=\"bench-overall\" class=\"chip\">idle</span>
      </div>

      <div class=\"grid\">
        <div class=\"card\"><div class=\"label\">Iterations</div><div id=\"b-iters\" class=\"value\">-</div></div>
        <div class=\"card\"><div class=\"label\">Latency Avg</div><div id=\"b-lat-avg\" class=\"value\">-</div></div>
        <div class=\"card\"><div class=\"label\">Latency P95</div><div id=\"b-lat-p95\" class=\"value\">-</div></div>
        <div class=\"card\"><div class=\"label\">Tokens/sec Avg</div><div id=\"b-tps\" class=\"value\">-</div></div>
      </div>

      <table>
        <thead><tr><th>Run</th><th>Status</th><th>Latency</th><th>Tokens/s</th><th>Eval Tokens</th><th>Detail</th></tr></thead>
        <tbody id=\"bench-results\"><tr><td colspan=\"6\" class=\"muted\">No benchmark run yet.</td></tr></tbody>
      </table>

      <details class=\"advanced\">
        <summary>Raw Benchmark JSON</summary>
        <pre id=\"bench-log\">Benchmark idle.</pre>
      </details>
    """

    script = """
    <script>
      function esc(s){
        return String(s ?? '')
          .replaceAll('&', '&amp;')
          .replaceAll('<', '&lt;')
          .replaceAll('>', '&gt;')
          .replaceAll('"', '&quot;')
          .replaceAll("'", '&#39;');
      }
      function setText(id, value){ const el=document.getElementById(id); if(el) el.textContent=value; }
      function setBadge(id, state){
        const el=document.getElementById(id);
        if(!el) return;
        if(state === 'pass'){ el.className='chip fit-great'; el.textContent='pass'; return; }
        if(state === 'fail'){ el.className='chip fit-heavy'; el.textContent='fail'; return; }
        if(state === 'running'){ el.className='chip fit-tight'; el.textContent='running'; return; }
        el.className='chip'; el.textContent='idle';
      }

      function renderSmokeChecks(checks){
        const body=document.getElementById('smoke-results');
        if(!body) return;
        if(!checks || checks.length === 0){
          body.innerHTML='<tr><td colspan=\"5\" class=\"muted\">No checks returned.</td></tr>';
          return;
        }
        body.innerHTML = checks.map((c) => {
          const state = c.skipped ? '<span class=\"chip\">skipped</span>' : (c.ok ? '<span class=\"chip fit-great\">pass</span>' : '<span class=\"chip fit-heavy\">fail</span>');
          const detail = c.detail ? esc(c.detail) : '';
          return `<tr>
            <td>${esc(c.name || '-')}</td>
            <td>${state}</td>
            <td>${Number(c.duration_ms ?? 0)} ms</td>
            <td>${esc(c.model || '-')}</td>
            <td class=\"muted\">${detail}</td>
          </tr>`;
        }).join('');
      }

      function renderBenchmarkRuns(samples){
        const body=document.getElementById('bench-results');
        if(!body) return;
        if(!samples || samples.length === 0){
          body.innerHTML='<tr><td colspan=\"6\" class=\"muted\">No benchmark samples returned.</td></tr>';
          return;
        }
        body.innerHTML = samples.map((s) => {
          const status = s.ok ? '<span class=\"chip fit-great\">pass</span>' : '<span class=\"chip fit-heavy\">fail</span>';
          const latency = Number(s.latency_ms ?? 0).toFixed(2) + ' ms';
          const tps = s.tokens_per_second > 0 ? Number(s.tokens_per_second).toFixed(2) : '-';
          const evalCount = s.eval_count ?? '-';
          const detail = s.ok ? `${Number(s.total_duration_ms ?? 0).toFixed(2)} ms total` : esc(s.error || '');
          return `<tr>
            <td>${s.run ?? '-'}</td>
            <td>${status}</td>
            <td>${latency}</td>
            <td>${tps}</td>
            <td>${evalCount}</td>
            <td class=\"muted\">${detail}</td>
          </tr>`;
        }).join('');
      }

      async function runSmoke(){
        const model=(document.getElementById('smoke-model').value || '').trim();
        const box=document.getElementById('smoke-log');
        setBadge('smoke-overall', 'running');
        box.textContent='Running smoke tests...';
        try {
          const r=await fetch('/api/tests/smoke', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ model }) });
          const data=await r.json();
          if(!r.ok) throw new Error(JSON.stringify(data));
          const s=data.summary || {};
          setText('s-passed', String(s.passed ?? '-'));
          setText('s-failed', String(s.failed ?? '-'));
          setText('s-skipped', String(s.skipped ?? '-'));
          setText('s-duration', s.duration_ms >= 0 ? `${s.duration_ms} ms` : '-');
          renderSmokeChecks(data.checks || []);
          setBadge('smoke-overall', data.ok ? 'pass' : 'fail');
          box.textContent=JSON.stringify(data, null, 2);
        } catch(e) {
          setBadge('smoke-overall', 'fail');
          renderSmokeChecks([]);
          box.textContent=`Smoke tests failed: ${e}`;
        }
      }

      async function runBenchmark(){
        const model=(document.getElementById('bench-model').value || '').trim();
        const prompt=(document.getElementById('bench-prompt').value || '').trim();
        const iterations=Number(document.getElementById('bench-iters').value || 5);
        const box=document.getElementById('bench-log');
        setBadge('bench-overall', 'running');
        box.textContent='Running benchmark...';
        try {
          const r=await fetch('/api/benchmarks/run', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ model, prompt, iterations }) });
          const data=await r.json();
          if(!r.ok) throw new Error(JSON.stringify(data));
          const s=data.summary || {};
          setText('b-iters', String(s.iterations ?? '-'));
          setText('b-lat-avg', s.latency_ms_avg >= 0 ? `${s.latency_ms_avg} ms` : '-');
          setText('b-lat-p95', s.latency_ms_p95 >= 0 ? `${s.latency_ms_p95} ms` : '-');
          setText('b-tps', s.tokens_per_second_avg > 0 ? String(s.tokens_per_second_avg) : '-');
          renderBenchmarkRuns(data.samples || []);
          setBadge('bench-overall', (s.failed_runs ?? 0) === 0 ? 'pass' : 'fail');
          box.textContent=JSON.stringify(data, null, 2);
        } catch(e) {
          setBadge('bench-overall', 'fail');
          renderBenchmarkRuns([]);
          box.textContent=`Benchmark failed: ${e}`;
        }
      }
    </script>
    """
    return _render_layout("LocalAI Validation", "Run smoke checks and benchmark inference paths.", "tests", content, script)
