# Networking and Exposure

## Public-Safe Defaults

By default, Docker ports bind to loopback only:

- `127.0.0.1:3000` (OpenWebUI)
- `127.0.0.1:3010` (Model Admin)
- `127.0.0.1:6333` (Qdrant)
- `127.0.0.1:8082` (SearxNG, only when web search is enabled)

If you run `localai up --expose` (or `localai up --expose <port>`), services bind to all interfaces:

- `0.0.0.0:80` (OpenWebUI)
- `0.0.0.0:3010` (Model Admin)
- `0.0.0.0:6333` (Qdrant)
- `0.0.0.0:8082` (SearxNG, only when web search is enabled)

## Endpoint Summary

- Default mode (`localai up`):
  - OpenWebUI: <http://127.0.0.1:3000>
  - Model Admin: <http://127.0.0.1:3010>
  - Qdrant API: <http://127.0.0.1:6333>
  - SearxNG: <http://127.0.0.1:8082> (only when web search is enabled)
- Exposed mode (`localai up --expose [PORT]`):
  - OpenWebUI: `http://<host-ip>:<PORT>` (defaults to `80` when omitted)
  - Model Admin: `http://<host-ip>:3010`
  - Qdrant API: `http://<host-ip>:6333`
  - SearxNG: `http://<host-ip>:8082` (only when web search is enabled)

## Optional Hardening (`.env`)

Template: `.env.example`

- `LOCALAI_BIND_IP=127.0.0.1` (recommended)
- `LOCALAI_QDRANT_PORT=6333`
- `MODEL_ADMIN_USERNAME=...`
- `MODEL_ADMIN_PASSWORD=...`
- `QDRANT_API_KEY=...` (optional)
- `LOCALAI_RAG_EMBED_MODEL=nomic-embed-text` (optional OpenWebUI embedding model override)

If both Model Admin credentials are set, Model Admin UI/API requires HTTP Basic Auth.
