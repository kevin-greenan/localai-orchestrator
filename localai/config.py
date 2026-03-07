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
    qdrant_url: str


@dataclass(slots=True)
class TuningConfig:
    enabled: bool
    respect_user_values: bool


@dataclass(slots=True)
class QdrantConfig:
    enabled: bool
    default_segment_number: int
    memmap_threshold_kb: int
    indexing_threshold_kb: int
    hnsw_m: int
    hnsw_ef_construct: int
    user_set_default_segment_number: bool
    user_set_memmap_threshold_kb: bool
    user_set_indexing_threshold_kb: bool
    user_set_hnsw_m: bool
    user_set_hnsw_ef_construct: bool


@dataclass(slots=True)
class RagConfig:
    preset: str
    qdrant: QdrantConfig


@dataclass(slots=True)
class WebRedisConfig:
    maxmemory_mb: int
    maxmemory_policy: str
    user_set_maxmemory_mb: bool
    user_set_maxmemory_policy: bool


@dataclass(slots=True)
class WebConfig:
    enabled: bool
    engine: str
    searxng_query_url: str
    result_count: int
    search_concurrent_requests: int
    loader_concurrent_requests: int
    request_timeout_seconds: int
    redis: WebRedisConfig
    user_set_result_count: bool
    user_set_search_concurrent_requests: bool
    user_set_loader_concurrent_requests: bool
    user_set_request_timeout_seconds: bool


@dataclass(slots=True)
class VisionConfig:
    enabled: bool
    default_model: str
    max_image_mb: int
    benchmark_dataset: str
    user_set_max_image_mb: bool


@dataclass(slots=True)
class ImageGenConfig:
    enabled: bool
    provider: str
    concurrency: int
    queue_timeout_seconds: int
    artifact_store: str
    backend_url: str
    a1111_url: str
    openwebui_model: str
    openwebui_image_size: str
    user_set_concurrency: bool
    user_set_queue_timeout_seconds: bool
    user_set_openwebui_image_size: bool


@dataclass(slots=True)
class StackConfig:
    name: str
    root: Path
    ollama: OllamaConfig
    docker: DockerConfig
    health: HealthConfig
    tuning: TuningConfig
    rag: RagConfig
    web: WebConfig
    vision: VisionConfig
    image_gen: ImageGenConfig


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
    rag_raw = raw.get("rag", {})
    qdrant_raw = rag_raw.get("qdrant", {})
    web_raw = raw.get("web", {})
    web_redis_raw = web_raw.get("redis", {})
    vision_raw = raw.get("vision", {})
    image_gen_raw = raw.get("image_gen", {})
    preset_raw = str(rag_raw.get("preset", "fast")).strip().lower()
    preset = preset_raw if preset_raw in {"fast", "deep"} else "fast"

    def _is_user_override(section: dict[str, object], key: str, default: object) -> bool:
        if key not in section:
            return False
        return section.get(key) != default

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
        qdrant_url=str(health_raw.get("qdrant_url", "http://127.0.0.1:6333/healthz")),
    )
    tuning = TuningConfig(
        enabled=bool(tuning_raw.get("enabled", True)),
        respect_user_values=bool(tuning_raw.get("respect_user_values", True)),
    )
    qdrant = QdrantConfig(
        enabled=bool(qdrant_raw.get("enabled", True)),
        default_segment_number=int(qdrant_raw.get("default_segment_number", 2)),
        memmap_threshold_kb=int(qdrant_raw.get("memmap_threshold_kb", 100000)),
        indexing_threshold_kb=int(qdrant_raw.get("indexing_threshold_kb", 50000)),
        hnsw_m=int(qdrant_raw.get("hnsw_m", 24)),
        hnsw_ef_construct=int(qdrant_raw.get("hnsw_ef_construct", 100)),
        user_set_default_segment_number="default_segment_number" in qdrant_raw,
        user_set_memmap_threshold_kb="memmap_threshold_kb" in qdrant_raw,
        user_set_indexing_threshold_kb="indexing_threshold_kb" in qdrant_raw,
        user_set_hnsw_m="hnsw_m" in qdrant_raw,
        user_set_hnsw_ef_construct="hnsw_ef_construct" in qdrant_raw,
    )
    rag = RagConfig(
        preset=preset,
        qdrant=qdrant,
    )
    web_redis = WebRedisConfig(
        maxmemory_mb=int(web_redis_raw.get("maxmemory_mb", 256)),
        maxmemory_policy=str(web_redis_raw.get("maxmemory_policy", "allkeys-lru")),
        user_set_maxmemory_mb="maxmemory_mb" in web_redis_raw,
        user_set_maxmemory_policy="maxmemory_policy" in web_redis_raw,
    )
    web = WebConfig(
        enabled=bool(web_raw.get("enabled", False)),
        engine=str(web_raw.get("engine", "searxng")).strip().lower(),
        searxng_query_url=str(
            web_raw.get("searxng_query_url", "http://searxng:8080/search?q=<query>&format=json")
        ).strip(),
        result_count=int(web_raw.get("result_count", 3)),
        search_concurrent_requests=int(web_raw.get("search_concurrent_requests", 4)),
        loader_concurrent_requests=int(web_raw.get("loader_concurrent_requests", 2)),
        request_timeout_seconds=int(web_raw.get("request_timeout_seconds", 8)),
        redis=web_redis,
        user_set_result_count="result_count" in web_raw,
        user_set_search_concurrent_requests="search_concurrent_requests" in web_raw,
        user_set_loader_concurrent_requests="loader_concurrent_requests" in web_raw,
        user_set_request_timeout_seconds="request_timeout_seconds" in web_raw,
    )
    vision = VisionConfig(
        enabled=bool(vision_raw.get("enabled", False)),
        default_model=str(vision_raw.get("default_model", "llava:latest")).strip(),
        max_image_mb=int(vision_raw.get("max_image_mb", 10)),
        benchmark_dataset=str(vision_raw.get("benchmark_dataset", "tests/fixtures/vision/smoke.jsonl")).strip(),
        user_set_max_image_mb=_is_user_override(vision_raw, "max_image_mb", 10),
    )
    image_gen = ImageGenConfig(
        enabled=bool(image_gen_raw.get("enabled", False)),
        provider=str(image_gen_raw.get("provider", "mock")).strip(),
        concurrency=int(image_gen_raw.get("concurrency", 1)),
        queue_timeout_seconds=int(image_gen_raw.get("queue_timeout_seconds", 300)),
        artifact_store=str(image_gen_raw.get("artifact_store", "minio")).strip(),
        backend_url=str(image_gen_raw.get("backend_url", "http://image-gen:8090")).strip(),
        a1111_url=str(image_gen_raw.get("a1111_url", "")).strip(),
        openwebui_model=str(image_gen_raw.get("openwebui_model", "localai-imagegen")).strip(),
        openwebui_image_size=str(image_gen_raw.get("openwebui_image_size", "1024x1024")).strip(),
        user_set_concurrency=_is_user_override(image_gen_raw, "concurrency", 1),
        user_set_queue_timeout_seconds=_is_user_override(image_gen_raw, "queue_timeout_seconds", 300),
        user_set_openwebui_image_size=_is_user_override(image_gen_raw, "openwebui_image_size", "1024x1024"),
    )

    return StackConfig(
        name=str(project.get("name", "localai")),
        root=root,
        ollama=ollama,
        docker=docker,
        health=health,
        tuning=tuning,
        rag=rag,
        web=web,
        vision=vision,
        image_gen=image_gen,
    )
