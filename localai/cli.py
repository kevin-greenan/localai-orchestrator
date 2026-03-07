from __future__ import annotations

import argparse
import json
import math
import platform
import re
import sys
import time
from pathlib import Path

from .config import DEFAULT_STACK, StackConfig, load_stack
from .docker import compose_down, compose_ps, compose_up, has_docker
from .health import http_ok, ollama_generate
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


def _effective_services(cfg: StackConfig) -> list[str]:
    runtime_env = _read_runtime_env(Path(cfg.root))
    raw = runtime_env.get("LOCALAI_DOCKER_SERVICES", "")
    if raw:
        return [s.strip() for s in raw.split(",") if s.strip()]
    return list(cfg.docker.services)


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
) -> None:
    env_path = Path(cfg.root) / ".localai.env"
    host_ram = tuning.specs.mem_gb if tuning and tuning.specs.mem_gb else 32
    max_queue = cfg.ollama.env.get("OLLAMA_MAX_QUEUE", "160")
    bind_ip = "0.0.0.0" if expose_enabled else "127.0.0.1"
    openwebui_port = str(expose_port if expose_enabled else 3000)
    model_admin_port = "3010"
    qdrant_port = "6333"
    qdrant_values, _ = _qdrant_tuned_values(cfg, tuning, boost_enabled=boost_enabled)
    rag_values = _rag_preset_values(rag_preset)
    active_services = docker_services or cfg.docker.services
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
        f"LOCALAI_QDRANT_ENABLED={'1' if cfg.rag.qdrant.enabled else '0'}",
        f"LOCALAI_QDRANT_DEFAULT_SEGMENT_NUMBER={qdrant_values['default_segment_number']}",
        f"LOCALAI_QDRANT_MEMMAP_THRESHOLD_KB={qdrant_values['memmap_threshold_kb']}",
        f"LOCALAI_QDRANT_INDEXING_THRESHOLD_KB={qdrant_values['indexing_threshold_kb']}",
        f"LOCALAI_QDRANT_HNSW_M={qdrant_values['hnsw_m']}",
        f"LOCALAI_QDRANT_HNSW_EF_CONSTRUCT={qdrant_values['hnsw_ef_construct']}",
        f"LOCALAI_DOCKER_SERVICES={','.join(active_services)}",
    ]
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _cmd_up(args: argparse.Namespace) -> int:
    cfg, tuning = _load_cfg_with_tuning(args.stack)
    expose_active = args.expose is not None
    expose_port = int(args.expose) if expose_active else 80
    if expose_active and not (1 <= expose_port <= 65535):
        raise RuntimeError("--expose port must be between 1 and 65535")
    rag_preset = (args.rag_preset or cfg.rag.preset).strip().lower()
    if rag_preset not in {"fast", "deep"}:
        raise RuntimeError("--rag-preset must be one of: fast, deep")
    boost_active = bool(args.boost and cfg.ollama.enabled)
    if args.boost and cfg.ollama.enabled:
        boost_applied = _apply_boost_profile(cfg, tuning)
        if boost_applied:
            print(f"boost profile applied: {json.dumps(boost_applied)}")
    _, qdrant_applied = _qdrant_tuned_values(cfg, tuning, boost_enabled=boost_active)
    if cfg.rag.qdrant.enabled and qdrant_applied:
        print(f"qdrant tuning applied: {json.dumps(qdrant_applied)}")
    if args.no_webui:
        cfg.docker.services = ["model-admin", "qdrant"]
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
        "--boost",
        action="store_true",
        help="Apply a higher-utilization runtime profile (more parallelism/queue/keep-alive).",
    )
    sub_up.add_argument(
        "--rag-preset",
        choices=["fast", "deep"],
        help="RAG retrieval profile for OpenWebUI defaults (fast=lower latency, deep=more context/quality).",
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
