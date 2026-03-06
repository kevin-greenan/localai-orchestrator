from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(slots=True)
class OllamaConfig:
    enabled: bool
    label: str
    host: str
    port: int
    keep_alive: str
    num_parallel: int
    max_loaded_models: int
    models: list[str]
    warmup_model: str
    warmup_prompt: str
    env: dict[str, str]
    user_set_keep_alive: bool
    user_set_num_parallel: bool
    user_set_max_loaded_models: bool
    user_set_env_keys: set[str]


@dataclass(slots=True)
class DockerConfig:
    compose_file: Path
    services: list[str]


@dataclass(slots=True)
class HealthConfig:
    openwebui_url: str


@dataclass(slots=True)
class TuningConfig:
    enabled: bool
    respect_user_values: bool


@dataclass(slots=True)
class StackConfig:
    name: str
    root: Path
    ollama: OllamaConfig
    docker: DockerConfig
    health: HealthConfig
    tuning: TuningConfig


DEFAULT_STACK = "stack.toml"


def load_stack(path: str | Path = DEFAULT_STACK) -> StackConfig:
    stack_path = Path(path).expanduser().resolve()
    with stack_path.open("rb") as f:
        raw = tomllib.load(f)

    root = stack_path.parent
    project = raw.get("project", {})
    ollama_raw = raw.get("native", {}).get("ollama", {})
    docker_raw = raw.get("docker", {})
    health_raw = raw.get("health", {})
    tuning_raw = raw.get("tuning", {})

    ollama = OllamaConfig(
        enabled=bool(ollama_raw.get("enabled", True)),
        label=str(ollama_raw.get("label", "com.localai.ollama")),
        host=str(ollama_raw.get("host", "127.0.0.1")),
        port=int(ollama_raw.get("port", 11434)),
        keep_alive=str(ollama_raw.get("keep_alive", "30m")),
        num_parallel=int(ollama_raw.get("num_parallel", 4)),
        max_loaded_models=int(ollama_raw.get("max_loaded_models", 2)),
        models=list(ollama_raw.get("models", [])),
        warmup_model=str(ollama_raw.get("warmup_model", "")),
        warmup_prompt=str(ollama_raw.get("warmup_prompt", "Respond with: ok")),
        env={str(k): str(v) for k, v in dict(ollama_raw.get("env", {})).items()},
        user_set_keep_alive="keep_alive" in ollama_raw,
        user_set_num_parallel="num_parallel" in ollama_raw,
        user_set_max_loaded_models="max_loaded_models" in ollama_raw,
        user_set_env_keys={str(k) for k in dict(ollama_raw.get("env", {})).keys()},
    )

    compose_file = Path(str(docker_raw.get("compose_file", "docker-compose.yml")))
    if not compose_file.is_absolute():
        compose_file = (root / compose_file).resolve()

    docker = DockerConfig(
        compose_file=compose_file,
        services=list(docker_raw.get("services", [])),
    )

    health = HealthConfig(
        openwebui_url=str(health_raw.get("openwebui_url", "http://127.0.0.1:3000/health")),
    )
    tuning = TuningConfig(
        enabled=bool(tuning_raw.get("enabled", True)),
        respect_user_values=bool(tuning_raw.get("respect_user_values", True)),
    )

    return StackConfig(
        name=str(project.get("name", "localai")),
        root=root,
        ollama=ollama,
        docker=docker,
        health=health,
        tuning=tuning,
    )
