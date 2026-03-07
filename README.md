# localai-orchestrator

macOS-first orchestration for Apple Silicon: run Ollama natively (Metal path) and run UI/services in Docker.

## What it does

- Starts `ollama serve` as a macOS `launchd` service
- Runs OpenWebUI + Model Admin + Qdrant via Docker Compose
- Optional web-search add-ons: SearxNG (+ Redis cache)
- Auto-tunes Ollama + Qdrant + web-search defaults from host hardware on startup
- Provides one CLI for lifecycle, checks, model sync, and warmup
- Exposes a Model Admin web UI with:
  - pull/update/delete model actions
  - live utilization cards + sparklines
  - RAG health strip (preset, qdrant status, collections, indexed vectors, qdrant latency)
  - searchable catalog with one-click pull
  - class and fit filters (`chat`, `reasoning`, `code`, `embed`, `vision`)

## Why this architecture

Docker on macOS does not give Linux containers native Apple Metal acceleration. This project keeps Ollama on the macOS host for performance, while still using Compose for everything else.

## Requirements

- macOS on Apple Silicon (`arm64`)
- Python `3.11+`
- [Ollama](https://ollama.com/) installed and available in `PATH`
- Docker Desktop (or compatible Docker engine) running

## Quick Start

```bash
cd localai-orchestrator
python3 -m venv .venv
source .venv/bin/activate
pip install setuptools wheel
pip install .

localai doctor
localai up --sync-models --warmup
```

When you change local source code and want `localai` to pick it up:

```bash
source .venv/bin/activate
pip install --no-build-isolation .
```

First run note:

- `--sync-models` pulls every model listed in `stack.toml` (`[native.ollama].models`).
- If a model is not local yet, the first run will download it and may take a while.
- If a model is already local, pull is usually quick and acts like an update check.

Python 3.14 note:

- Prefer `pip install .` or `pip install --no-build-isolation .` for this project.
- Avoid `pip install -e .` here; it can produce a CLI entrypoint that does not resolve the `localai` module reliably.

## Public-Safe Defaults

By default, Docker ports are bound to loopback only:

- `127.0.0.1:3000` (OpenWebUI)
- `127.0.0.1:3010` (Model Admin)
- `127.0.0.1:6333` (Qdrant)
- `127.0.0.1:8082` (SearxNG, only when `[web].enabled=true`)

If you run `localai up --expose` (or `localai up --expose <port>`), services bind to all interfaces:

- `0.0.0.0:80` (OpenWebUI)
- `0.0.0.0:3010` (Model Admin)
- `0.0.0.0:6333` (Qdrant)
- `0.0.0.0:8082` (SearxNG, only when `[web].enabled=true`)

Optional hardening config lives in `.env` (template: `.env.example`):

- `LOCALAI_BIND_IP=127.0.0.1` (recommended)
- `LOCALAI_QDRANT_PORT=6333`
- `MODEL_ADMIN_USERNAME=...`
- `MODEL_ADMIN_PASSWORD=...`
- `QDRANT_API_KEY=...` (optional)
- `LOCALAI_RAG_EMBED_MODEL=nomic-embed-text` (optional OpenWebUI embedding model override)

If both Model Admin credentials are set, the Model Admin UI/API requires HTTP Basic Auth.

## Usage

### Start Modes

```bash
# Full startup (recommended first run)
localai up --sync-models --warmup

# Full startup with web search enabled for this run
localai up --web-search --sync-models --warmup

# Higher-utilization mode (good for larger Apple Silicon machines)
localai up --boost --warmup

# Higher-quality retrieval profile (more context, slower)
localai up --rag-preset deep --warmup

# Expose services on all interfaces (OpenWebUI defaults to :80)
localai up --expose --sync-models --warmup

# Expose services on all interfaces with custom OpenWebUI port
localai up --expose 8080 --sync-models --warmup

# Start only Model Admin (skip OpenWebUI)
localai up --no-webui --warmup

# Start stack without pulling models
localai up --warmup

# Start only services (no sync, no warmup)
localai up
```

What each flag does:

- `--sync-models`: runs `ollama pull` for all configured models in `stack.toml`
- `--warmup`: runs one test inference on `warmup_model` after Ollama is reachable
- `--expose [PORT]`: binds services to `0.0.0.0` (OpenWebUI on `:PORT`, default `80`; Model Admin on `:3010`)
- `--no-webui`: starts `model-admin` + `qdrant` (OpenWebUI is not started)
- `--web-search`: enables web search for this run and starts `searxng` + `redis`
- `--boost`: applies a higher-utilization runtime profile (parallelism/queue/keep-alive, and model residency when RAM allows)
  - Also applies more aggressive Qdrant and web-search profiles unless manually overridden
- `--rag-preset {fast,deep}`: sets OpenWebUI RAG defaults
  - `fast` (default): lower latency (`top_k=2`, `chunk_size=800`, no hybrid search)
  - `deep`: higher recall/context (`top_k=5`, `chunk_size=1200`, hybrid search on)

### Stop Modes

```bash
# Stop Docker services only (OpenWebUI + Model Admin + Qdrant)
localai down

# Stop Docker services and native Ollama launch agent
localai down --stop-native
```

Use `localai down` when you want to keep native Ollama running for direct CLI/API use.
Use `localai down --stop-native` when you want everything fully stopped.

### Status, Logs, and Health

```bash
localai status
localai logs --tail 200
localai doctor
```

## Web UI Endpoints

- Default (`localai up`):
  - OpenWebUI: <http://127.0.0.1:3000>
  - Model Admin: <http://127.0.0.1:3010>
  - Qdrant API: <http://127.0.0.1:6333>
  - SearxNG: <http://127.0.0.1:8082> (only when web search is enabled)
- Exposed mode (`localai up --expose [PORT]`):
  - OpenWebUI: `http://<host-ip>:<PORT>` (defaults to `80` when omitted)
  - Model Admin: `http://<host-ip>:3010`
  - Qdrant API: `http://<host-ip>:6333`
  - SearxNG: `http://<host-ip>:8082` (only when web search is enabled)

## RAG Storage (Qdrant)

Qdrant is included as a default service for local RAG/vector storage.

- Container: `localai-qdrant`
- API port: `6333` (or `LOCALAI_QDRANT_PORT`)
- Data persistence: Docker named volume `qdrant-data`
- Optional auth: set `QDRANT_API_KEY` in `.env`
- Auto-tuned at startup from host specs, with boost-aware profile on `localai up --boost`
- Manual overrides available in `stack.toml` under `[rag.qdrant]`

OpenWebUI is pre-wired for RAG out of the box:

- `VECTOR_DB=qdrant`
- `QDRANT_URI=http://qdrant:6333`
- `RAG_EMBEDDING_ENGINE=ollama`
- `RAG_EMBEDDING_MODEL=nomic-embed-text` (override with `LOCALAI_RAG_EMBED_MODEL`)
- `RAG_TOP_K`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, and `ENABLE_RAG_HYBRID_SEARCH` are set from the selected `rag` preset (`fast` by default)

Default model sync includes both chat and embedding models:

- `llama3.2:3b`
- `nomic-embed-text`

Persistence note:

- Qdrant collections/vectors persist across container rebuilds/restarts.
- Data is removed only if you explicitly remove volumes (for example `docker compose down -v`).

## Web Search (SearxNG + Redis)

Web search is optional and disabled by default.

Enable in `stack.toml`:

```toml
[web]
enabled = true
engine = "searxng"
searxng_query_url = "http://searxng:8080/search?q=<query>&format=json"

[web.redis]
maxmemory_mb = 512
```

Behavior:

- When `[web].enabled = true`, `localai up` includes `searxng` automatically.
- When web search is enabled, `localai up` includes `redis` automatically.
- `localai up --no-webui` skips OpenWebUI and web-search add-ons.
- OpenWebUI env wiring is auto-generated in `.localai.env`.

## Configuration

Main config: `stack.toml`

Key sections:

- `[native.ollama]`: host/port, models, warmup defaults, optional manual runtime overrides
- `[docker]`: compose file and service list
- `[rag]`: RAG preset (`fast` or `deep`) used for OpenWebUI retrieval defaults
- `[rag.qdrant]`: qdrant enable/disable and optional manual tuning overrides
- `[web]`: web-search enablement and OpenWebUI/SearxNG defaults
- `[web.redis]`: Redis cache sizing policy for web-search workloads
- `[health]`: health-check URLs
- `[tuning]`: auto-tuning behavior

Health URLs include:

- `openwebui_url`
- `qdrant_url`

### Auto-Tuning

On `localai up`, host hardware is detected and these are auto-derived:

- `num_parallel`
- `max_loaded_models`
- `keep_alive`
- `OLLAMA_MAX_QUEUE`
- Qdrant `default_segment_number`
- Qdrant `memmap_threshold_kb`
- Qdrant `indexing_threshold_kb`
- Qdrant `hnsw_m`
- Qdrant `hnsw_ef_construct`
- Web search `result_count`
- Web search request/loader concurrency
- Web search request timeout
- Redis `maxmemory_mb` (when web search is enabled)

Default policy:

```toml
[tuning]
enabled = true
respect_user_values = true
```

If you set manual values in `[native.ollama]`, `[rag.qdrant]`, `[web]`, or `[web.redis]` and keep `respect_user_values = true`, your values win.

RAG preset persistence note:

- OpenWebUI stores retrieval settings in its own persistent DB.
- `--rag-preset` (or `[rag].preset`) applies startup defaults, mainly for fresh setups.
- If you already changed retrieval settings in the OpenWebUI UI, those saved values may override env defaults until you change/reset them in UI.

## Runtime Files

Generated at runtime:

- Launch agent plist: `~/Library/LaunchAgents/com.localai.ollama.plist`
- Ollama logs:
  - `~/.localai/logs/ollama.out.log`
  - `~/.localai/logs/ollama.err.log`
- Compose env generated by CLI: `.localai.env`
  - includes `LOCALAI_HOST_RAM_GB` for Model Admin fit badges
  - includes generated OpenWebUI + SearxNG + Redis runtime tuning values

## Model Admin Notes

- Delete is most reliable with a full tag (`model:tag`), for example `mistral:latest`
- Catalog merges curated recommendations with best-effort discovery from Ollama Library
- Fit badges are heuristic estimates based on host RAM profile and model/tag size patterns

### Advanced Ollama Console (Debug)

The Model Admin UI includes a collapsible **Advanced Ollama Console (Debug)** panel with safe, allowlisted operations:

- `tags`: Calls Ollama `/api/tags` to list local models and metadata.
- `ps`: Calls Ollama `/api/ps` to show currently loaded/running models.
- `show`: Calls Ollama `/api/show` for details about one model.
  - Requires `model`.
- `generate`: Calls Ollama `/api/generate` for a single non-streamed inference.
  - Requires `model` and `prompt`.
- `pull`: Calls Ollama `/api/pull` to pull/update one model (streamed server-side, summarized in response).
  - Requires `model`.

The console returns raw JSON plus execution time (`duration_ms`) for debugging.

Security note:

- The console is intentionally allowlisted to Ollama operations only (not a host shell).
- If you expose Model Admin beyond localhost, set Basic Auth credentials and use a trusted reverse proxy/TLS.

## Troubleshooting

- `docker daemon permission denied`:
  - Ensure Docker Desktop is running and your user can access Docker socket
- `ollama binary not found`:
  - Confirm `ollama` is in `PATH` (`which ollama`)
- OpenWebUI asks for admin setup repeatedly:
  - Do not remove volumes (`docker compose down -v` clears persisted data)
- OpenWebUI web search returns no results:
  - Confirm `[web].enabled = true` and `localai up` started the `searxng` service.
  - Verify `SEARXNG_QUERY_URL` contains `<query>` and includes `format=json`.
- Qdrant collections missing after restart:
  - Ensure you did not run `docker compose down -v` (removes `qdrant-data` volume)
- Model Admin changes not reflected after code updates:
  - Rebuild only that service:
  - `docker compose up -d --build model-admin`
- `localai` missing new flags after pulling latest code:
  - Reinstall package into the active venv:
  - `pip install --no-build-isolation .`
- Model Admin returns `401 Unauthorized`:
  - You likely enabled `MODEL_ADMIN_USERNAME`/`MODEL_ADMIN_PASSWORD` in `.env`.
  - Open `http://127.0.0.1:3010` and authenticate with those credentials.
- `Bootstrap failed: 125: Domain does not support specified action`:
  - Update to the latest release; launchctl domain fallback (`gui/<uid>` -> `user/<uid>`) is included.
  - If it still occurs, run from a normal logged-in desktop session (not a restricted/background shell context).

## Project Structure

- `localai/`: Python CLI + orchestration logic
- `model_admin/`: FastAPI app for model management UI
- `docker-compose.yml`: OpenWebUI + model-admin + qdrant services
- `stack.toml`: user-tunable stack configuration
