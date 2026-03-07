# Model Admin

`model_admin/` is a FastAPI service that provides a browser UI for Ollama model management and stack observability.

## What It Provides

- Pull, update, and delete model actions
- Live utilization cards + sparklines
- RAG health strip (preset, qdrant status, collections, indexed vectors, qdrant latency)
- Searchable catalog with one-click pull
- Class and fit filters (`chat`, `reasoning`, `code`, `embed`, `vision`)

## Operational Notes

- Delete is most reliable with a full tag (`model:tag`), for example `mistral:latest`.
- Catalog merges curated recommendations with best-effort discovery from Ollama Library.
- Fit badges are heuristic estimates based on host RAM profile and model/tag size patterns.

## Security

- If both `MODEL_ADMIN_USERNAME` and `MODEL_ADMIN_PASSWORD` are set, UI/API routes require HTTP Basic Auth.
- If you expose Model Admin beyond localhost, use a trusted reverse proxy with TLS.

## Advanced Ollama Console (Debug)

The UI includes an allowlisted debug console for safe Ollama API operations (not a host shell):

- `tags`: call `/api/tags` to list local models and metadata
- `ps`: call `/api/ps` to show currently loaded/running models
- `show`: call `/api/show` for one model (requires `model`)
- `generate`: call `/api/generate` for one non-streamed inference (requires `model` and `prompt`)
- `pull`: call `/api/pull` for one model update/pull (requires `model`)

Responses include raw JSON and `duration_ms`.
