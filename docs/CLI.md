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

## Benchmarking and Metrics

```bash
# Quick baseline (5 runs, JSON output)
localai benchmark

# Vision benchmark lane (calls Model Admin vision benchmark API)
localai benchmark vision --model llava:latest --dataset tests/fixtures/vision/smoke.jsonl --iterations 5

# Vision smoke check with a local image
localai test vision-smoke --model llava:latest --image ./sample.png

# Vision smoke check with an HTTP image URL
localai test vision-smoke --model llava:latest --image-url https://example.com/sample.png

# Compare model variants with more samples
localai benchmark --model llama3.2:3b --iterations 10
localai benchmark --model qwen2.5:7b --iterations 10

# Keep prompt fixed when comparing profile or config changes
localai benchmark --prompt "Summarize this repo in one sentence." --iterations 8
```

`localai benchmark` prints JSON including:

- service health probe latencies (`ollama`, and active docker services)
- per-run inference wall time
- token throughput from Ollama (`tokens_per_second`)
- summary stats (`avg`, `p50`, `p95`, `min`, `max`)

`localai benchmark vision` prints JSON including:

- dataset path and iteration counts
- per-sample latency and pass/fail scoring
- aggregate latency and pass-rate metrics

Vision command notes:

- `localai test vision-smoke` requires one of `--image` or `--image-url`
- Vision commands require `[vision].enabled = true` in `stack.toml`
