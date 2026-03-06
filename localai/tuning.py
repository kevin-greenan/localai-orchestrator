from __future__ import annotations

from dataclasses import asdict, dataclass
import os
import platform

from .config import StackConfig
from .shell import run


@dataclass(slots=True)
class SystemSpecs:
    mem_gb: int
    logical_cpu: int
    physical_cpu: int
    perf_cores: int
    model: str
    machine: str


@dataclass(slots=True)
class TuningResult:
    enabled: bool
    specs: SystemSpecs
    applied: dict[str, str]
    recommendations: dict[str, str]


def _sysctl_int(key: str, default: int) -> int:
    res = run(["sysctl", "-n", key])
    if res.code != 0 or not res.stdout:
        return default
    try:
        return int(res.stdout.strip())
    except ValueError:
        return default


def _sysctl_str(key: str, default: str) -> str:
    res = run(["sysctl", "-n", key])
    if res.code != 0 or not res.stdout:
        return default
    return res.stdout.strip()


def detect_system_specs() -> SystemSpecs:
    mem_bytes = _sysctl_int("hw.memsize", 0)
    mem_gb = max(8, round(mem_bytes / (1024**3)))
    logical_cpu = _sysctl_int("hw.logicalcpu", os.cpu_count() or 4)
    physical_cpu = _sysctl_int("hw.physicalcpu", logical_cpu)
    perf_cores = _sysctl_int("hw.perflevel0.physicalcpu", physical_cpu)
    model = _sysctl_str("hw.model", "unknown")
    machine = _sysctl_str("hw.machine", platform.machine())

    return SystemSpecs(
        mem_gb=mem_gb,
        logical_cpu=logical_cpu,
        physical_cpu=physical_cpu,
        perf_cores=perf_cores,
        model=model,
        machine=machine,
    )


def _recommended(specs: SystemSpecs) -> dict[str, str]:
    if specs.mem_gb >= 128:
        max_loaded_models = 5
        keep_alive = "3h"
        max_queue = "1024"
    elif specs.mem_gb >= 64:
        max_loaded_models = 4
        keep_alive = "2h"
        max_queue = "768"
    elif specs.mem_gb >= 36:
        max_loaded_models = 3
        keep_alive = "90m"
        max_queue = "512"
    elif specs.mem_gb >= 24:
        max_loaded_models = 2
        keep_alive = "60m"
        max_queue = "320"
    else:
        max_loaded_models = 1
        keep_alive = "30m"
        max_queue = "160"

    perf_parallel = specs.perf_cores * 2 if specs.perf_cores > 0 else specs.logical_cpu // 2
    num_parallel = max(2, min(16, perf_parallel))
    if specs.mem_gb < 24:
        num_parallel = min(num_parallel, 4)

    return {
        "num_parallel": str(num_parallel),
        "max_loaded_models": str(max_loaded_models),
        "keep_alive": keep_alive,
        "env.OLLAMA_MAX_QUEUE": max_queue,
    }


def apply_autotune(cfg: StackConfig) -> TuningResult:
    specs = detect_system_specs()
    rec = _recommended(specs)

    if not cfg.tuning.enabled:
        return TuningResult(enabled=False, specs=specs, applied={}, recommendations=rec)

    applied: dict[str, str] = {}

    if not (cfg.tuning.respect_user_values and cfg.ollama.user_set_num_parallel):
        cfg.ollama.num_parallel = int(rec["num_parallel"])
        applied["num_parallel"] = rec["num_parallel"]

    if not (cfg.tuning.respect_user_values and cfg.ollama.user_set_max_loaded_models):
        cfg.ollama.max_loaded_models = int(rec["max_loaded_models"])
        applied["max_loaded_models"] = rec["max_loaded_models"]

    if not (cfg.tuning.respect_user_values and cfg.ollama.user_set_keep_alive):
        cfg.ollama.keep_alive = rec["keep_alive"]
        applied["keep_alive"] = rec["keep_alive"]

    queue_key = "OLLAMA_MAX_QUEUE"
    if not (cfg.tuning.respect_user_values and queue_key in cfg.ollama.user_set_env_keys):
        cfg.ollama.env[queue_key] = rec["env.OLLAMA_MAX_QUEUE"]
        applied[f"env.{queue_key}"] = rec["env.OLLAMA_MAX_QUEUE"]

    return TuningResult(enabled=True, specs=specs, applied=applied, recommendations=rec)


def tuning_to_dict(t: TuningResult) -> dict[str, object]:
    return {
        "enabled": t.enabled,
        "specs": asdict(t.specs),
        "applied": t.applied,
        "recommendations": t.recommendations,
    }
