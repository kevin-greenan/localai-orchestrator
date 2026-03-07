from __future__ import annotations

import argparse
from datetime import datetime, timezone
import base64
import json
import math
import platform
import re
import secrets
import statistics
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from .config import DEFAULT_STACK, StackConfig, load_stack
from .docker import compose_down, compose_ps, compose_up, has_docker
from .health import http_ok, ollama_generate, ollama_generate_json
from .macos import (
    is_apple_silicon,
    is_macos,
    launch_agent_status,
    ollama_bin,
    start_ollama_launch_agent,
    stop_ollama_launch_agent,
)
from .shell import run
from .tuning import TuningResult, apply_autotune, tuning_to_dict


def _load_cfg_with_tuning(stack: str) -> tuple[StackConfig, TuningResult]:
    cfg = load_stack(stack)
    tuning = apply_autotune(cfg)
    return cfg, tuning


def _models_sync(cfg: StackConfig) -> int:
    host = f"{cfg.ollama.host}:{cfg.ollama.port}"
    for model in cfg.ollama.models:
        print(f"pulling {model}...")
        res = run(["ollama", "pull", model], env={"OLLAMA_HOST": host}, check=False)
        if res.code != 0:
            raise RuntimeError(res.stderr or res.stdout or f"failed pulling {model}")
    print("models synced")
    return 0


def _warmup(cfg: StackConfig) -> int:
    if not cfg.ollama.warmup_model:
        print("no warmup_model configured")
        return 0

    host = f"{cfg.ollama.host}:{cfg.ollama.port}"
    deadline = time.monotonic() + 60.0
    last_msg = "unknown error"

    while time.monotonic() < deadline:
        ok, msg = ollama_generate(host, cfg.ollama.warmup_model, cfg.ollama.warmup_prompt)
        if ok:
            print("warmup complete")
            return 0
        last_msg = msg
        # Missing model will not self-recover; fail fast with actionable guidance.
        if "not found" in msg.lower() and cfg.ollama.warmup_model.lower() in msg.lower():
            raise RuntimeError(
                f"warmup failed: {msg}. Pull the model first or run `localai up --sync-models --warmup`."
            )
        time.sleep(1.0)

    raise RuntimeError(f"warmup failed after retries: {last_msg}")


def _wait_for_ollama(cfg: StackConfig, timeout_s: float = 45.0, interval_s: float = 1.0) -> None:
    url = f"http://{cfg.ollama.host}:{cfg.ollama.port}/api/tags"
    deadline = time.monotonic() + timeout_s
    last_msg = "unknown error"

    while time.monotonic() < deadline:
        ok, msg = http_ok(url, timeout=2.0)
        if ok:
            return
        last_msg = msg
        time.sleep(interval_s)

    raise RuntimeError(
        f"ollama did not become ready within {int(timeout_s)}s at {url}: {last_msg}"
    )


def _keep_alive_minutes(value: str) -> int:
    m = re.fullmatch(r"\s*(\d+)\s*([mhdMHD])\s*", value)
    if not m:
        return 30
    n = int(m.group(1))
    unit = m.group(2).lower()
    if unit == "m":
        return n
    if unit == "h":
        return n * 60
    if unit == "d":
        return n * 24 * 60
    return 30


def _queue_cap_for_mem(mem_gb: int) -> int:
    if mem_gb >= 128:
        return 1024
    if mem_gb >= 64:
        return 768
    if mem_gb >= 32:
        return 512
    return 256


def _qdrant_recommended(mem_gb: int, perf_cores: int) -> dict[str, int]:
    if mem_gb >= 128:
        return {
            "default_segment_number": max(4, min(8, perf_cores)),
            "memmap_threshold_kb": 300000,
            "indexing_threshold_kb": 200000,
            "hnsw_m": 48,
            "hnsw_ef_construct": 200,
        }
    if mem_gb >= 64:
        return {
            "default_segment_number": max(3, min(6, perf_cores)),
            "memmap_threshold_kb": 200000,
            "indexing_threshold_kb": 100000,
            "hnsw_m": 32,
            "hnsw_ef_construct": 128,
        }
    if mem_gb >= 32:
        return {
            "default_segment_number": max(2, min(4, perf_cores)),
            "memmap_threshold_kb": 100000,
            "indexing_threshold_kb": 50000,
            "hnsw_m": 24,
            "hnsw_ef_construct": 100,
        }
    return {
        "default_segment_number": 1,
        "memmap_threshold_kb": 50000,
        "indexing_threshold_kb": 20000,
        "hnsw_m": 16,
        "hnsw_ef_construct": 64,
    }


def _qdrant_boosted(base: dict[str, int], mem_gb: int, perf_cores: int) -> dict[str, int]:
    seg_cap = max(1, min(12, perf_cores if perf_cores > 0 else 4))
    boosted = dict(base)
    boosted["default_segment_number"] = min(seg_cap, base["default_segment_number"] + 1)
    boosted["memmap_threshold_kb"] = min(600000 if mem_gb >= 64 else 300000, base["memmap_threshold_kb"] * 2)
    boosted["indexing_threshold_kb"] = min(500000 if mem_gb >= 64 else 200000, base["indexing_threshold_kb"] * 2)
    boosted["hnsw_m"] = min(64, base["hnsw_m"] + 8)
    boosted["hnsw_ef_construct"] = min(256, base["hnsw_ef_construct"] + 64)
    return boosted


def _qdrant_tuned_values(cfg: StackConfig, tuning: TuningResult, *, boost_enabled: bool) -> tuple[dict[str, int], dict[str, str]]:
    q = cfg.rag.qdrant
    base = _qdrant_recommended(tuning.specs.mem_gb, tuning.specs.perf_cores)
    target = _qdrant_boosted(base, tuning.specs.mem_gb, tuning.specs.perf_cores) if boost_enabled else base

    values = {
        "default_segment_number": q.default_segment_number,
        "memmap_threshold_kb": q.memmap_threshold_kb,
        "indexing_threshold_kb": q.indexing_threshold_kb,
        "hnsw_m": q.hnsw_m,
        "hnsw_ef_construct": q.hnsw_ef_construct,
    }
    user_set = {
        "default_segment_number": q.user_set_default_segment_number,
        "memmap_threshold_kb": q.user_set_memmap_threshold_kb,
        "indexing_threshold_kb": q.user_set_indexing_threshold_kb,
        "hnsw_m": q.user_set_hnsw_m,
        "hnsw_ef_construct": q.user_set_hnsw_ef_construct,
    }
    applied: dict[str, str] = {}
    for k, v in target.items():
        if cfg.tuning.respect_user_values and user_set[k]:
            continue
        if values[k] != v:
            values[k] = v
            applied[f"qdrant.{k}"] = str(v)
    return values, applied


def _web_recommended(mem_gb: int, perf_cores: int) -> dict[str, int]:
    perf_hint = perf_cores if perf_cores > 0 else 4
    if mem_gb >= 128:
        return {
            "result_count": 8,
            "search_concurrent_requests": min(16, perf_hint * 2),
            "loader_concurrent_requests": min(10, perf_hint),
            "request_timeout_seconds": 12,
            "redis_maxmemory_mb": 2048,
        }
    if mem_gb >= 64:
        return {
            "result_count": 6,
            "search_concurrent_requests": min(12, perf_hint * 2),
            "loader_concurrent_requests": min(8, perf_hint),
            "request_timeout_seconds": 10,
            "redis_maxmemory_mb": 1024,
        }
    if mem_gb >= 32:
        return {
            "result_count": 5,
            "search_concurrent_requests": min(8, perf_hint + 2),
            "loader_concurrent_requests": min(6, perf_hint),
            "request_timeout_seconds": 8,
            "redis_maxmemory_mb": 512,
        }
    return {
        "result_count": 4,
        "search_concurrent_requests": min(4, max(2, perf_hint)),
        "loader_concurrent_requests": 2,
        "request_timeout_seconds": 6,
        "redis_maxmemory_mb": 256,
    }


def _web_boosted(base: dict[str, int], mem_gb: int, perf_cores: int) -> dict[str, int]:
    perf_hint = perf_cores if perf_cores > 0 else 4
    boosted = dict(base)
    boosted["result_count"] = min(10, base["result_count"] + 2)
    boosted["search_concurrent_requests"] = min(20, max(base["search_concurrent_requests"] + 2, perf_hint * 2))
    boosted["loader_concurrent_requests"] = min(12, base["loader_concurrent_requests"] + 2)
    boosted["request_timeout_seconds"] = min(20, base["request_timeout_seconds"] + 2)
    redis_cap = 3072 if mem_gb >= 64 else 1536
    boosted["redis_maxmemory_mb"] = min(redis_cap, max(base["redis_maxmemory_mb"] * 2, base["redis_maxmemory_mb"] + 256))
    return boosted


def _web_tuned_values(cfg: StackConfig, tuning: TuningResult, *, boost_enabled: bool) -> tuple[dict[str, int], dict[str, str]]:
    web = cfg.web
    redis = cfg.web.redis
    base = _web_recommended(tuning.specs.mem_gb, tuning.specs.perf_cores)
    target = _web_boosted(base, tuning.specs.mem_gb, tuning.specs.perf_cores) if boost_enabled else base

    values = {
        "result_count": web.result_count,
        "search_concurrent_requests": web.search_concurrent_requests,
        "loader_concurrent_requests": web.loader_concurrent_requests,
        "request_timeout_seconds": web.request_timeout_seconds,
        "redis_maxmemory_mb": redis.maxmemory_mb,
    }
    user_set = {
        "result_count": web.user_set_result_count,
        "search_concurrent_requests": web.user_set_search_concurrent_requests,
        "loader_concurrent_requests": web.user_set_loader_concurrent_requests,
        "request_timeout_seconds": web.user_set_request_timeout_seconds,
        "redis_maxmemory_mb": redis.user_set_maxmemory_mb,
    }
    applied: dict[str, str] = {}
    for k, v in target.items():
        if cfg.tuning.respect_user_values and user_set[k]:
            continue
        if values[k] != v:
            values[k] = v
            applied[f"web.{k}"] = str(v)
    return values, applied


def _rag_preset_values(preset: str) -> dict[str, str]:
    mode = preset if preset in {"fast", "deep"} else "fast"
    if mode == "deep":
        return {
            "top_k": "5",
            "chunk_size": "1200",
            "chunk_overlap": "160",
            "hybrid_search": "true",
            "embedding_batch_size": "2",
            "embedding_concurrent_requests": "2",
        }
    return {
        "top_k": "2",
        "chunk_size": "800",
        "chunk_overlap": "100",
        "hybrid_search": "false",
        "embedding_batch_size": "1",
        "embedding_concurrent_requests": "1",
    }


def _read_runtime_env(root: Path) -> dict[str, str]:
    env_path = root / ".localai.env"
    if not env_path.exists():
        return {}
    out: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _effective_openwebui_url(cfg: StackConfig) -> str:
    runtime_env = _read_runtime_env(Path(cfg.root))
    port = runtime_env.get("LOCALAI_OPENWEBUI_PORT")
    if port and port.isdigit():
        return f"http://127.0.0.1:{port}/health"
    return cfg.health.openwebui_url


def _effective_qdrant_url(cfg: StackConfig) -> str:
    runtime_env = _read_runtime_env(Path(cfg.root))
    port = runtime_env.get("LOCALAI_QDRANT_PORT")
    if port and port.isdigit():
        return f"http://127.0.0.1:{port}/healthz"
    return cfg.health.qdrant_url


def _effective_searxng_url(cfg: StackConfig) -> str:
    runtime_env = _read_runtime_env(Path(cfg.root))
    port = runtime_env.get("LOCALAI_SEARXNG_PORT")
    if port and port.isdigit():
        return f"http://127.0.0.1:{port}/healthz"
    return "http://127.0.0.1:8082/healthz"


def _effective_services(cfg: StackConfig) -> list[str]:
    runtime_env = _read_runtime_env(Path(cfg.root))
    raw = runtime_env.get("LOCALAI_DOCKER_SERVICES", "")
    if raw:
        return [s.strip() for s in raw.split(",") if s.strip()]
    return list(cfg.docker.services)


def _effective_model_admin_url(cfg: StackConfig) -> str:
    runtime_env = _read_runtime_env(Path(cfg.root))
    port = runtime_env.get("LOCALAI_MODEL_ADMIN_PORT")
    if port and port.isdigit():
        return f"http://127.0.0.1:{port}"
    return "http://127.0.0.1:3010"


def _post_json(url: str, payload: dict[str, object], timeout: float) -> dict[str, object]:
    data = json.dumps(payload).encode("utf-8")
    req = Request(url, method="POST", data=data, headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}
    except HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"request failed ({e.code}): {detail}") from e
    except URLError as e:
        raise RuntimeError(f"request failed: {e}") from e


def _encode_image_file(path: str) -> str:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    if not p.exists():
        raise RuntimeError(f"image file not found: {p}")
    return base64.b64encode(p.read_bytes()).decode("ascii")


def _redis_ping(cfg: StackConfig) -> tuple[bool, str]:
    cmd = ["docker", "compose"]
    env_file = Path(cfg.root) / ".localai.env"
    if env_file.exists():
        cmd.extend(["--env-file", str(env_file)])
    cmd.extend(["-f", str(cfg.docker.compose_file), "exec", "-T", "redis", "redis-cli", "ping"])
    res = run(cmd, cwd=str(cfg.root), check=False)
    out = (res.stdout or res.stderr).strip()
    if res.code == 0 and "PONG" in out:
        return True, "PONG"
    if not out:
        out = f"exit {res.code}"
    return False, out


def _apply_boost_profile(cfg: StackConfig, tuning: TuningResult) -> dict[str, str]:
    specs = tuning.specs
    applied: dict[str, str] = {}

    perf_hint = specs.perf_cores if specs.perf_cores > 0 else max(2, specs.logical_cpu // 2)
    parallel_cap = 24
    if specs.mem_gb < 24:
        parallel_cap = 6
    elif specs.mem_gb < 36:
        parallel_cap = 10
    target_parallel = max(cfg.ollama.num_parallel + 1, math.ceil(cfg.ollama.num_parallel * 1.5))
    target_parallel = min(target_parallel, perf_hint * 3, parallel_cap)
    if target_parallel != cfg.ollama.num_parallel:
        cfg.ollama.num_parallel = target_parallel
        applied["num_parallel"] = str(target_parallel)

    max_loaded = cfg.ollama.max_loaded_models
    if specs.mem_gb >= 64:
        max_loaded = min(max_loaded + 2, 6)
    elif specs.mem_gb >= 32:
        max_loaded = min(max_loaded + 1, 4)
    elif specs.mem_gb >= 24:
        max_loaded = min(max_loaded + 1, 3)
    if max_loaded != cfg.ollama.max_loaded_models:
        cfg.ollama.max_loaded_models = max_loaded
        applied["max_loaded_models"] = str(max_loaded)

    current_keep_alive_min = _keep_alive_minutes(cfg.ollama.keep_alive)
    target_keep_alive_min = current_keep_alive_min
    if specs.mem_gb >= 64:
        target_keep_alive_min = max(target_keep_alive_min, 120)
    elif specs.mem_gb >= 24:
        target_keep_alive_min = max(target_keep_alive_min, 90)
    elif specs.mem_gb >= 16:
        target_keep_alive_min = max(target_keep_alive_min, 60)
    if target_keep_alive_min != current_keep_alive_min:
        cfg.ollama.keep_alive = f"{target_keep_alive_min}m"
        applied["keep_alive"] = cfg.ollama.keep_alive

    queue_key = "OLLAMA_MAX_QUEUE"
    try:
        current_queue = int(cfg.ollama.env.get(queue_key, "160"))
    except ValueError:
        current_queue = 160
    target_queue = min(max(current_queue * 2, current_queue + 64), _queue_cap_for_mem(specs.mem_gb))
    if target_queue != current_queue:
        cfg.ollama.env[queue_key] = str(target_queue)
        applied[f"env.{queue_key}"] = str(target_queue)

    return applied


def _write_runtime_env(
    cfg: StackConfig,
    tuning: TuningResult,
    *,
    rag_preset: str,
    boost_enabled: bool = False,
    expose_enabled: bool = False,
    expose_port: int = 80,
    docker_services: list[str] | None = None,
    openwebui_enabled: bool = True,
) -> None:
    env_path = Path(cfg.root) / ".localai.env"
    runtime_env = _read_runtime_env(Path(cfg.root))
    host_ram = tuning.specs.mem_gb if tuning and tuning.specs.mem_gb else 32
    max_queue = cfg.ollama.env.get("OLLAMA_MAX_QUEUE", "160")
    bind_ip = "0.0.0.0" if expose_enabled else "127.0.0.1"
    openwebui_port = str(expose_port if (expose_enabled and openwebui_enabled) else 3000)
    model_admin_port = "3010"
    qdrant_port = "6333"
    qdrant_values, _ = _qdrant_tuned_values(cfg, tuning, boost_enabled=boost_enabled)
    web_values, _ = _web_tuned_values(cfg, tuning, boost_enabled=boost_enabled)
    rag_values = _rag_preset_values(rag_preset)
    active_services = docker_services or cfg.docker.services
    redis_url = "redis://redis:6379/0" if cfg.web.enabled else ""
    searxng_secret = runtime_env.get("LOCALAI_SEARXNG_SECRET") or secrets.token_hex(24)
    lines = [
        "# Generated by localai CLI.",
        f"LOCALAI_BIND_IP={bind_ip}",
        f"LOCALAI_OPENWEBUI_PORT={openwebui_port}",
        f"LOCALAI_MODEL_ADMIN_PORT={model_admin_port}",
        f"LOCALAI_QDRANT_PORT={qdrant_port}",
        f"LOCALAI_HOST_RAM_GB={host_ram}",
        f"LOCALAI_CPU_LOGICAL={tuning.specs.logical_cpu}",
        f"LOCALAI_CPU_PHYSICAL={tuning.specs.physical_cpu}",
        f"LOCALAI_CPU_PERF={tuning.specs.perf_cores}",
        f"LOCALAI_MACHINE={tuning.specs.machine}",
        f"LOCALAI_HW_MODEL={tuning.specs.model}",
        "LOCALAI_GPU_BACKEND=Metal",
        "LOCALAI_GPU_NAME=Apple Silicon Integrated GPU",
        "LOCALAI_GPU_CORES=unknown",
        f"LOCALAI_OLLAMA_NUM_PARALLEL={cfg.ollama.num_parallel}",
        f"LOCALAI_OLLAMA_MAX_LOADED_MODELS={cfg.ollama.max_loaded_models}",
        f"LOCALAI_OLLAMA_KEEP_ALIVE={cfg.ollama.keep_alive}",
        f"LOCALAI_OLLAMA_MAX_QUEUE={max_queue}",
        f"LOCALAI_BOOST_ACTIVE={'1' if boost_enabled else '0'}",
        f"LOCALAI_EXPOSE_ACTIVE={'1' if expose_enabled else '0'}",
        f"LOCALAI_RAG_PRESET={rag_preset}",
        f"LOCALAI_RAG_TOP_K={rag_values['top_k']}",
        f"LOCALAI_RAG_CHUNK_SIZE={rag_values['chunk_size']}",
        f"LOCALAI_RAG_CHUNK_OVERLAP={rag_values['chunk_overlap']}",
        f"LOCALAI_RAG_HYBRID_SEARCH={rag_values['hybrid_search']}",
        f"LOCALAI_RAG_EMBED_BATCH_SIZE={rag_values['embedding_batch_size']}",
        f"LOCALAI_RAG_EMBED_CONCURRENT_REQUESTS={rag_values['embedding_concurrent_requests']}",
        f"LOCALAI_WEB_ENABLED={'true' if cfg.web.enabled else 'false'}",
        f"LOCALAI_WEB_SEARCH_ENGINE={cfg.web.engine}",
        f"LOCALAI_SEARXNG_QUERY_URL={cfg.web.searxng_query_url}",
        f"LOCALAI_WEB_SEARCH_RESULT_COUNT={web_values['result_count']}",
        f"LOCALAI_WEB_SEARCH_CONCURRENT_REQUESTS={web_values['search_concurrent_requests']}",
        f"LOCALAI_WEB_LOADER_CONCURRENT_REQUESTS={web_values['loader_concurrent_requests']}",
        f"LOCALAI_WEB_SEARCH_TIMEOUT_SECONDS={web_values['request_timeout_seconds']}",
        "LOCALAI_SEARXNG_PORT=8082",
        f"LOCALAI_SEARXNG_SECRET={searxng_secret}",
        f"LOCALAI_WEB_REDIS_MAXMEMORY_MB={web_values['redis_maxmemory_mb']}",
        f"LOCALAI_WEB_REDIS_MAXMEMORY_POLICY={cfg.web.redis.maxmemory_policy}",
        f"LOCALAI_SEARXNG_REDIS_URL={redis_url}",
        f"LOCALAI_QDRANT_ENABLED={'1' if cfg.rag.qdrant.enabled else '0'}",
        f"LOCALAI_QDRANT_DEFAULT_SEGMENT_NUMBER={qdrant_values['default_segment_number']}",
        f"LOCALAI_QDRANT_MEMMAP_THRESHOLD_KB={qdrant_values['memmap_threshold_kb']}",
        f"LOCALAI_QDRANT_INDEXING_THRESHOLD_KB={qdrant_values['indexing_threshold_kb']}",
        f"LOCALAI_QDRANT_HNSW_M={qdrant_values['hnsw_m']}",
        f"LOCALAI_QDRANT_HNSW_EF_CONSTRUCT={qdrant_values['hnsw_ef_construct']}",
        f"LOCALAI_VISION_ENABLED={'1' if cfg.vision.enabled else '0'}",
        f"LOCALAI_VISION_DEFAULT_MODEL={cfg.vision.default_model}",
        f"LOCALAI_VISION_MAX_IMAGE_MB={cfg.vision.max_image_mb}",
        f"LOCALAI_VISION_BENCHMARK_DATASET={cfg.vision.benchmark_dataset}",
        f"LOCALAI_IMAGE_GEN_ENABLED={'1' if cfg.image_gen.enabled else '0'}",
        f"LOCALAI_IMAGE_GEN_PROVIDER={cfg.image_gen.provider}",
        f"LOCALAI_IMAGE_GEN_CONCURRENCY={cfg.image_gen.concurrency}",
        f"LOCALAI_IMAGE_GEN_QUEUE_TIMEOUT_SECONDS={cfg.image_gen.queue_timeout_seconds}",
        f"LOCALAI_IMAGE_GEN_ARTIFACT_STORE={cfg.image_gen.artifact_store}",
        f"LOCALAI_IMAGE_GEN_BACKEND_URL={cfg.image_gen.backend_url}",
        f"LOCALAI_DOCKER_SERVICES={','.join(active_services)}",
    ]
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _cmd_up(args: argparse.Namespace) -> int:
    cfg, tuning = _load_cfg_with_tuning(args.stack)
    if args.web_search:
        cfg.web.enabled = True
    expose_active = args.expose is not None
    expose_port = int(args.expose) if expose_active else 80
    if expose_active and not (1 <= expose_port <= 65535):
        raise RuntimeError("--expose port must be between 1 and 65535")
    rag_preset = (args.rag_preset or cfg.rag.preset).strip().lower()
    if rag_preset not in {"fast", "deep"}:
        raise RuntimeError("--rag-preset must be one of: fast, deep")
    if args.no_webui and args.web_search:
        raise RuntimeError("--web-search cannot be combined with --no-webui")
    if args.no_webui and args.rag_preset:
        raise RuntimeError("--rag-preset cannot be combined with --no-webui")
    if cfg.web.enabled and cfg.web.engine != "searxng":
        raise RuntimeError("web.engine must be 'searxng' when web search is enabled")
    if cfg.web.enabled and "<query>" not in cfg.web.searxng_query_url:
        raise RuntimeError("web.searxng_query_url must contain the <query> placeholder")
    boost_active = bool(args.boost and cfg.ollama.enabled)
    if args.boost and cfg.ollama.enabled:
        boost_applied = _apply_boost_profile(cfg, tuning)
        if boost_applied:
            print(f"boost profile applied: {json.dumps(boost_applied)}")
    _, web_applied = _web_tuned_values(cfg, tuning, boost_enabled=boost_active)
    if cfg.web.enabled and web_applied:
        print(f"web tuning applied: {json.dumps(web_applied)}")
    _, qdrant_applied = _qdrant_tuned_values(cfg, tuning, boost_enabled=boost_active)
    if cfg.rag.qdrant.enabled and qdrant_applied:
        print(f"qdrant tuning applied: {json.dumps(qdrant_applied)}")
    if args.no_webui:
        cfg.web.enabled = False
        cfg.docker.services = ["model-admin", "qdrant"]
    elif cfg.web.enabled:
        if "searxng" not in cfg.docker.services:
            cfg.docker.services.append("searxng")
        if "redis" not in cfg.docker.services:
            cfg.docker.services.append("redis")
    else:
        cfg.docker.services = [s for s in cfg.docker.services if s not in {"searxng", "redis"}]
    if not cfg.rag.qdrant.enabled:
        cfg.docker.services = [s for s in cfg.docker.services if s != "qdrant"]
    _write_runtime_env(
        cfg,
        tuning,
        rag_preset=rag_preset,
        boost_enabled=boost_active,
        expose_enabled=expose_active,
        expose_port=expose_port,
        docker_services=cfg.docker.services,
        openwebui_enabled=(not args.no_webui),
    )
    print(f"rag preset: {rag_preset}")
    if cfg.ollama.enabled:
        if tuning.enabled and tuning.applied:
            print(f"autotune applied: {json.dumps(tuning.applied)}")
        start_ollama_launch_agent(cfg)
        if args.sync_models or args.warmup:
            _wait_for_ollama(cfg)
        if args.sync_models:
            _models_sync(cfg)
        if args.warmup:
            _warmup(cfg)
    compose_up(cfg)
    print("stack is up")
    return 0


def _cmd_down(args: argparse.Namespace) -> int:
    cfg, _ = _load_cfg_with_tuning(args.stack)
    compose_down(cfg)
    if args.stop_native and cfg.ollama.enabled:
        stop_ollama_launch_agent(cfg)
    print("stack is down")
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    cfg, tuning = _load_cfg_with_tuning(args.stack)
    status = {
        "platform": platform.platform(),
        "macos": is_macos(),
        "apple_silicon": is_apple_silicon(),
        "autotune": tuning_to_dict(tuning),
        "ollama": {
            "enabled": cfg.ollama.enabled,
            "binary": ollama_bin(),
            "service": launch_agent_status(cfg.ollama.label) if cfg.ollama.enabled and is_macos() else "disabled",
            "endpoint": f"http://{cfg.ollama.host}:{cfg.ollama.port}",
        },
        "docker_compose_ps": compose_ps(cfg) if has_docker() else "docker not found",
    }
    print(json.dumps(status, indent=2))
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    cfg, tuning = _load_cfg_with_tuning(args.stack)
    checks: list[tuple[str, bool, str]] = []

    checks.append(("macOS", is_macos(), platform.system()))
    checks.append(("Apple Silicon", is_apple_silicon(), platform.machine()))
    checks.append(("ollama binary", ollama_bin() is not None, str(ollama_bin())))
    checks.append(("docker binary", has_docker(), "docker" if has_docker() else "missing"))
    checks.append(("autotune enabled", tuning.enabled, json.dumps(tuning.applied or tuning.recommendations)))

    if has_docker():
        d = run(["docker", "info"])
        checks.append(("docker daemon", d.code == 0, d.stderr or "ok"))

    ollama_ok, ollama_msg = http_ok(f"http://{cfg.ollama.host}:{cfg.ollama.port}/api/tags")
    checks.append(("ollama http", ollama_ok, ollama_msg[:120]))

    active_services = set(_effective_services(cfg))
    if "openwebui" in active_services:
        ui_ok, ui_msg = http_ok(_effective_openwebui_url(cfg))
        checks.append(("openwebui http", ui_ok, ui_msg[:120]))
    else:
        checks.append(("openwebui http", True, "skipped (service not enabled)"))
    if "searxng" in active_services:
        searxng_ok, searxng_msg = http_ok(_effective_searxng_url(cfg))
        checks.append(("searxng http", searxng_ok, searxng_msg[:120]))
    elif cfg.web.enabled:
        checks.append(("searxng http", True, "skipped (service not enabled)"))
    if "redis" in active_services:
        redis_ok, redis_msg = _redis_ping(cfg)
        checks.append(("redis ping", redis_ok, redis_msg[:120]))
    elif cfg.web.enabled:
        checks.append(("redis ping", True, "skipped (service not enabled)"))
    if cfg.rag.qdrant.enabled and "qdrant" in active_services:
        qdrant_ok, qdrant_msg = http_ok(_effective_qdrant_url(cfg))
        checks.append(("qdrant http", qdrant_ok, qdrant_msg[:120]))
    elif cfg.rag.qdrant.enabled:
        checks.append(("qdrant http", True, "skipped (service not enabled)"))

    failed = False
    for name, ok, msg in checks:
        icon = "OK" if ok else "FAIL"
        print(f"[{icon}] {name}: {msg}")
        if not ok:
            failed = True
    return 1 if failed else 0


def _cmd_models_sync(args: argparse.Namespace) -> int:
    cfg, _ = _load_cfg_with_tuning(args.stack)
    return _models_sync(cfg)


def _cmd_warmup(args: argparse.Namespace) -> int:
    cfg, _ = _load_cfg_with_tuning(args.stack)
    return _warmup(cfg)


def _cmd_logs(args: argparse.Namespace) -> int:
    cfg, _ = _load_cfg_with_tuning(args.stack)

    if cfg.ollama.enabled:
        log_path = Path.home() / ".localai" / "logs" / "ollama.out.log"
        if log_path.exists():
            out = run(["tail", "-n", str(args.tail), str(log_path)])
            if out.stdout:
                print("# ollama.out.log")
                print(out.stdout)

    cmd = ["docker", "compose"]
    env_file = Path(cfg.root) / ".localai.env"
    if env_file.exists():
        cmd.extend(["--env-file", str(env_file)])
    cmd.extend(["-f", str(cfg.docker.compose_file), "logs", f"--tail={args.tail}"])
    if args.follow:
        cmd.append("-f")
    res = run(cmd, cwd=str(cfg.root))
    if res.stdout:
        print(res.stdout)
    if res.stderr:
        print(res.stderr, file=sys.stderr)
    return res.code


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(math.ceil((q / 100.0) * len(ordered)) - 1)))
    return ordered[idx]


def _probe_http(url: str, timeout: float) -> dict[str, object]:
    started = time.perf_counter()
    ok, msg = http_ok(url, timeout=timeout)
    latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
    return {"ok": ok, "latency_ms": latency_ms, "detail": msg[:200]}


def _cmd_benchmark(args: argparse.Namespace) -> int:
    if getattr(args, "benchmark_kind", "text") == "vision":
        return _cmd_benchmark_vision(args)

    cfg, _ = _load_cfg_with_tuning(args.stack)
    if args.iterations < 1:
        raise RuntimeError("--iterations must be at least 1")

    model = args.model or cfg.ollama.warmup_model or (cfg.ollama.models[0] if cfg.ollama.models else "")
    if not model:
        raise RuntimeError("No benchmark model configured; set warmup_model or pass --model")

    host = f"{cfg.ollama.host}:{cfg.ollama.port}"
    active_services = set(_effective_services(cfg))

    health: dict[str, dict[str, object]] = {
        "ollama": _probe_http(f"http://{host}/api/tags", timeout=args.timeout),
    }
    if "openwebui" in active_services:
        health["openwebui"] = _probe_http(_effective_openwebui_url(cfg), timeout=args.timeout)
    if "qdrant" in active_services:
        health["qdrant"] = _probe_http(_effective_qdrant_url(cfg), timeout=args.timeout)
    if "searxng" in active_services:
        health["searxng"] = _probe_http(_effective_searxng_url(cfg), timeout=args.timeout)

    samples: list[dict[str, object]] = []
    latencies: list[float] = []
    token_rates: list[float] = []
    failures = 0

    for idx in range(args.iterations):
        started = time.perf_counter()
        ok, payload, err = ollama_generate_json(host, model, args.prompt, timeout=args.timeout)
        wall_ms = round((time.perf_counter() - started) * 1000.0, 2)
        sample: dict[str, object] = {"run": idx + 1, "ok": ok, "latency_ms": wall_ms}
        if ok:
            eval_count = int(payload.get("eval_count", 0) or 0)
            eval_duration_ns = int(payload.get("eval_duration", 0) or 0)
            total_duration_ns = int(payload.get("total_duration", 0) or 0)
            prompt_eval_count = int(payload.get("prompt_eval_count", 0) or 0)
            prompt_eval_duration_ns = int(payload.get("prompt_eval_duration", 0) or 0)
            load_duration_ns = int(payload.get("load_duration", 0) or 0)
            token_s = round((eval_count / (eval_duration_ns / 1_000_000_000.0)), 2) if eval_duration_ns > 0 else 0.0
            latencies.append(wall_ms)
            if token_s > 0:
                token_rates.append(token_s)
            sample.update(
                {
                    "eval_count": eval_count,
                    "eval_duration_ms": round(eval_duration_ns / 1_000_000.0, 2),
                    "total_duration_ms": round(total_duration_ns / 1_000_000.0, 2),
                    "load_duration_ms": round(load_duration_ns / 1_000_000.0, 2),
                    "prompt_eval_count": prompt_eval_count,
                    "prompt_eval_duration_ms": round(prompt_eval_duration_ns / 1_000_000.0, 2),
                    "tokens_per_second": token_s,
                }
            )
        else:
            failures += 1
            sample["error"] = err
        samples.append(sample)

    summary = {
        "iterations": args.iterations,
        "successful_runs": len(latencies),
        "failed_runs": failures,
        "latency_ms_avg": round(statistics.fmean(latencies), 2) if latencies else 0.0,
        "latency_ms_min": round(min(latencies), 2) if latencies else 0.0,
        "latency_ms_p50": round(_percentile(latencies, 50), 2) if latencies else 0.0,
        "latency_ms_p95": round(_percentile(latencies, 95), 2) if latencies else 0.0,
        "latency_ms_max": round(max(latencies), 2) if latencies else 0.0,
        "tokens_per_second_avg": round(statistics.fmean(token_rates), 2) if token_rates else 0.0,
    }
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "prompt_chars": len(args.prompt),
        "health": health,
        "summary": summary,
        "samples": samples,
    }
    print(json.dumps(result, indent=2))
    return 1 if failures == args.iterations else 0


def _cmd_test_vision_smoke(args: argparse.Namespace) -> int:
    cfg, _ = _load_cfg_with_tuning(args.stack)
    if not cfg.vision.enabled:
        raise RuntimeError("vision lane is disabled in stack.toml ([vision].enabled=false)")
    if not args.image and not args.image_url:
        raise RuntimeError("vision smoke requires one of --image or --image-url")

    model_admin = _effective_model_admin_url(cfg)
    payload: dict[str, object] = {
        "model": args.model or cfg.vision.default_model,
        "prompt": args.prompt,
        "timeout_seconds": args.timeout,
    }
    if args.image:
        payload["image_base64"] = _encode_image_file(args.image)
    if args.image_url:
        payload["image_url"] = args.image_url

    out = _post_json(f"{model_admin}/api/tests/vision-smoke", payload, timeout=max(5.0, args.timeout + 5.0))
    print(json.dumps(out, indent=2))
    return 0 if bool(out.get("ok")) else 1


def _cmd_benchmark_vision(args: argparse.Namespace) -> int:
    cfg, _ = _load_cfg_with_tuning(args.stack)
    if not cfg.vision.enabled:
        raise RuntimeError("vision lane is disabled in stack.toml ([vision].enabled=false)")

    model_admin = _effective_model_admin_url(cfg)
    payload: dict[str, object] = {
        "model": args.model or cfg.vision.default_model,
        "dataset_path": args.dataset or cfg.vision.benchmark_dataset,
        "iterations": args.iterations,
        "timeout_seconds": args.timeout,
    }
    out = _post_json(f"{model_admin}/api/tests/vision-benchmark", payload, timeout=max(5.0, args.timeout + 10.0))
    print(json.dumps(out, indent=2))
    summary = out.get("summary", {})
    if isinstance(summary, dict):
        return 0 if int(summary.get("failed_runs", 1) or 0) == 0 else 1
    return 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="localai", description="macOS-first orchestrator for native Ollama + Docker stack")
    p.add_argument("--stack", default=DEFAULT_STACK, help="Path to stack.toml")

    sub = p.add_subparsers(dest="command", required=True)

    sub_up = sub.add_parser("up", help="Start native Ollama and docker services")
    sub_up.add_argument("--sync-models", action="store_true", help="Pull configured models after starting native Ollama")
    sub_up.add_argument("--warmup", action="store_true", help="Run warmup inference after model sync/start")
    sub_up.add_argument(
        "--expose",
        nargs="?",
        const=80,
        default=None,
        type=int,
        metavar="PORT",
        help="Expose services on all interfaces. Optional OpenWebUI port (default: 80).",
    )
    sub_up.add_argument(
        "--no-webui",
        action="store_true",
        dest="no_webui",
        help="Start Model Admin + Qdrant only (skip OpenWebUI).",
    )
    sub_up.add_argument(
        "--admin-only",
        action="store_true",
        dest="no_webui",
        help=argparse.SUPPRESS,
    )
    sub_up.add_argument(
        "--web-search",
        action="store_true",
        help="Enable web search for this run (starts SearxNG + Redis with OpenWebUI). Incompatible with --no-webui.",
    )
    sub_up.add_argument(
        "--boost",
        action="store_true",
        help="Apply a higher-utilization runtime profile (more parallelism/queue/keep-alive).",
    )
    sub_up.add_argument(
        "--rag-preset",
        choices=["fast", "deep"],
        help="RAG retrieval profile for OpenWebUI defaults (fast=lower latency, deep=more context/quality). Incompatible with --no-webui.",
    )
    sub_up.set_defaults(func=_cmd_up)

    sub_down = sub.add_parser("down", help="Stop docker services")
    sub_down.add_argument("--stop-native", action="store_true", help="Also stop native Ollama launch agent")
    sub_down.set_defaults(func=_cmd_down)

    sub_status = sub.add_parser("status", help="Show stack status")
    sub_status.set_defaults(func=_cmd_status)

    sub_doctor = sub.add_parser("doctor", help="Run system checks")
    sub_doctor.set_defaults(func=_cmd_doctor)

    sub_models = sub.add_parser("models-sync", help="Pull configured Ollama models")
    sub_models.set_defaults(func=_cmd_models_sync)

    sub_warmup = sub.add_parser("warmup", help="Run warmup inference on configured model")
    sub_warmup.set_defaults(func=_cmd_warmup)

    sub_logs = sub.add_parser("logs", help="Show stack logs")
    sub_logs.add_argument("--follow", action="store_true", help="Stream logs")
    sub_logs.add_argument("--tail", type=int, default=100)
    sub_logs.set_defaults(func=_cmd_logs)

    sub_test = sub.add_parser("test", help="Run test workflows")
    sub_test_sub = sub_test.add_subparsers(dest="test_kind", required=True)
    sub_test_vision = sub_test_sub.add_parser("vision-smoke", help="Run one vision smoke check via Model Admin")
    sub_test_vision.add_argument("--model", help="Vision model tag (defaults to [vision].default_model)")
    sub_test_vision.add_argument("--prompt", default="Return exactly: ok", help="Vision prompt for smoke test")
    sub_test_vision.add_argument("--image", help="Local image path to send as base64")
    sub_test_vision.add_argument("--image-url", help="HTTP(S) image URL")
    sub_test_vision.add_argument("--timeout", type=float, default=30.0, help="Request timeout in seconds")
    sub_test_vision.set_defaults(func=_cmd_test_vision_smoke)

    sub_bench = sub.add_parser("benchmark", help="Run quick local performance probes and Ollama inference benchmark")
    sub_bench.add_argument(
        "benchmark_kind",
        nargs="?",
        choices=["text", "vision"],
        default="text",
        help="Benchmark lane: text (default) or vision.",
    )
    sub_bench.add_argument("--model", help="Model to benchmark (defaults: warmup_model, then first configured model)")
    sub_bench.add_argument("--dataset", help="Vision benchmark dataset JSONL path (for benchmark vision)")
    sub_bench.add_argument("--prompt", default="Summarize why local-first AI stacks are useful in one sentence.")
    sub_bench.add_argument("--iterations", type=int, default=5, help="Number of benchmark inference runs")
    sub_bench.add_argument("--timeout", type=float, default=60.0, help="Per-request timeout in seconds")
    sub_bench.set_defaults(func=_cmd_benchmark)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        rc = args.func(args)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        raise SystemExit(1) from e

    raise SystemExit(rc)


if __name__ == "__main__":
    main()
