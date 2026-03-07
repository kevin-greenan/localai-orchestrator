# CLI Reference

## Start Modes

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

# Start only Model Admin + Qdrant (skip OpenWebUI)
localai up --no-webui --warmup

# Start stack without pulling models
localai up --warmup

# Start only services (no sync, no warmup)
localai up
```

## Flag Behavior (`localai up`)

- `--sync-models`: runs `ollama pull` for all configured models in `stack.toml`
- `--warmup`: runs one test inference on `warmup_model` after Ollama is reachable
- `--expose [PORT]`: binds services to `0.0.0.0` (OpenWebUI on `:PORT`, default `80`; Model Admin on `:3010`)
- `--no-webui`: starts `model-admin` + `qdrant` only (OpenWebUI and web-search add-ons are not started)
- `--web-search`: enables web search for this run and starts `searxng` + `redis`
- `--boost`: applies a higher-utilization runtime profile (parallelism/queue/keep-alive and model residency when RAM allows)
- `--rag-preset {fast,deep}`: sets OpenWebUI RAG defaults
  - `fast` (default): lower latency (`top_k=2`, `chunk_size=800`, no hybrid search)
  - `deep`: higher recall/context (`top_k=5`, `chunk_size=1200`, hybrid search on)

## Compatibility Notes

- `--web-search` cannot be combined with `--no-webui`.
- `--rag-preset` cannot be combined with `--no-webui`.
- `--expose` can be combined with `--no-webui`; when OpenWebUI is disabled, the optional `PORT` value is ignored.

## Stop Modes

```bash
# Stop Docker services only (OpenWebUI + Model Admin + Qdrant)
localai down

# Stop Docker services and native Ollama launch agent
localai down --stop-native
```

Use `localai down` when you want to keep native Ollama running for direct CLI/API use.
Use `localai down --stop-native` when you want everything fully stopped.

## Status, Logs, and Health

```bash
localai status
localai logs --tail 200
localai doctor
```
