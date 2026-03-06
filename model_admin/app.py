from __future__ import annotations

import json
import os
import re
from time import perf_counter
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434").rstrip("/")
OLLAMA_LIBRARY_URL = os.getenv("OLLAMA_LIBRARY_URL", "https://ollama.com/library")
LOCALAI_HOST_RAM_GB = max(8, int(os.getenv("LOCALAI_HOST_RAM_GB", "32")))

POPULAR_MODELS: list[dict[str, str]] = [
    {"name": "llama3.2", "tag": "3b", "description": "Fast general-purpose model"},
    {"name": "llama3.3", "tag": "latest", "description": "Large-capability Llama model"},
    {"name": "qwen2.5", "tag": "7b", "description": "Strong coding + reasoning balance"},
    {"name": "deepseek-r1", "tag": "latest", "description": "Reasoning-focused model family"},
    {"name": "mistral", "tag": "latest", "description": "Compact high-quality instruction model"},
    {"name": "mistral-small", "tag": "latest", "description": "Lower-latency Mistral variant"},
    {"name": "gemma3", "tag": "latest", "description": "Google Gemma open model line"},
    {"name": "phi4", "tag": "latest", "description": "Small but capable model"},
    {"name": "codestral", "tag": "latest", "description": "Code-first model"},
    {"name": "qwen2.5-coder", "tag": "latest", "description": "Coding-specialized Qwen"},
    {"name": "nomic-embed-text", "tag": "latest", "description": "Embedding model for RAG"},
    {"name": "bge-m3", "tag": "latest", "description": "Multilingual embedding model"},
    {"name": "llava", "tag": "latest", "description": "Vision-language model"},
]

app = FastAPI(title="LocalAI Model Admin", version="0.4.0")

ACTION_STATE: dict[str, Any] = {
    "active_jobs": 0,
    "completed_jobs": 0,
    "failed_jobs": 0,
    "last_action": "",
    "last_model": "",
    "last_duration_ms": 0,
    "last_error": "",
}


class ModelAction(BaseModel):
    model: str = Field(min_length=1)


def _sum_int(items: list[dict[str, Any]], key: str) -> int:
    total = 0
    for item in items:
        value = item.get(key, 0)
        if isinstance(value, int):
            total += value
    return total


def _by_name(items: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for item in items:
        name = item.get("name", "").strip().lower()
        if name:
            out[name] = item
    return out


def _classify_model(name: str) -> str:
    n = name.lower()
    if any(x in n for x in ["embed", "bge", "e5", "nomic"]):
        return "embed"
    if any(x in n for x in ["coder", "codestral", "codegemma", "code"]):
        return "code"
    if any(x in n for x in ["llava", "vision", "moondream", "vl"]):
        return "vision"
    if any(x in n for x in ["r1", "reason"]):
        return "reasoning"
    return "chat"


def _estimate_ram_gb(model_name: str, tag: str, model_class: str) -> float:
    if model_class == "embed":
        return 2.0

    match = re.search(r"(\d+)(?:\.\d+)?b", tag.lower())
    if match:
        params_b = float(match.group(1))
        return max(2.0, round(params_b * 0.8, 1))

    if "llama3.3" in model_name:
        return 32.0
    if "phi" in model_name or "mistral-small" in model_name:
        return 4.0
    return 8.0


def _fit_tier(required_ram_gb: float, host_ram_gb: int) -> str:
    if required_ram_gb <= host_ram_gb * 0.35:
        return "great"
    if required_ram_gb <= host_ram_gb * 0.60:
        return "good"
    if required_ram_gb <= host_ram_gb * 0.85:
        return "tight"
    return "heavy"


async def _ollama_get(path: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(f"{OLLAMA_BASE_URL}{path}")
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return resp.json()


async def _ollama_stream_post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    last: dict[str, Any] = {}
    raw_lines: list[str] = []
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", f"{OLLAMA_BASE_URL}{path}", json=payload) as resp:
            if resp.status_code >= 400:
                body = await resp.aread()
                raise HTTPException(status_code=resp.status_code, detail=body.decode("utf-8", errors="ignore"))

            async for line in resp.aiter_lines():
                if not line:
                    continue
                raw_lines.append(line)
                try:
                    item = json.loads(line)
                    if isinstance(item, dict):
                        last = item
                except Exception:
                    continue

    return {
        "ok": True,
        "last": last,
        "events": raw_lines[-15:],
    }


async def _ollama_request(method: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.request(method, f"{OLLAMA_BASE_URL}{path}", json=payload)
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    if resp.content:
        try:
            return resp.json()
        except Exception:
            return {"ok": True, "raw": resp.text}
    return {"ok": True}


async def _tracked_action(
    action: str,
    model: str,
    path: str,
    payload: dict[str, Any],
    *,
    stream: bool,
    method: str = "POST",
) -> dict[str, Any]:
    ACTION_STATE["active_jobs"] += 1
    ACTION_STATE["last_action"] = action
    ACTION_STATE["last_model"] = model
    ACTION_STATE["last_error"] = ""

    start = perf_counter()
    try:
        if stream:
            result = await _ollama_stream_post(path, payload)
        else:
            try:
                result = await _ollama_request(method, path, payload)
            except HTTPException as e:
                # Compatibility path for older/newer Ollama API variants.
                if method == "DELETE" and e.status_code == 405:
                    result = await _ollama_request("POST", path, payload)
                else:
                    raise
        ACTION_STATE["completed_jobs"] += 1
        return result
    except Exception as e:  # noqa: BLE001
        ACTION_STATE["failed_jobs"] += 1
        ACTION_STATE["last_error"] = str(e)
        raise
    finally:
        ACTION_STATE["active_jobs"] -= 1
        ACTION_STATE["last_duration_ms"] = int((perf_counter() - start) * 1000)


async def _discover_library_names(limit: int = 120) -> list[str]:
    async with httpx.AsyncClient(timeout=8.0) as client:
        resp = await client.get(OLLAMA_LIBRARY_URL)
    if resp.status_code >= 400:
        return []

    names = re.findall(r'href="/library/([a-zA-Z0-9._-]+)"', resp.text)
    dedup: list[str] = []
    seen: set[str] = set()
    for name in names:
        n = name.strip().lower()
        if not n or n in seen:
            continue
        seen.add(n)
        dedup.append(n)
        if len(dedup) >= limit:
            break
    return dedup


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/models")
async def list_models() -> dict[str, Any]:
    return await _ollama_get("/api/tags")


@app.get("/api/catalog")
async def catalog(q: str = Query("", max_length=80), limit: int = Query(60, ge=5, le=200)) -> dict[str, Any]:
    q_norm = q.strip().lower()

    tags_data = await _ollama_get("/api/tags")
    local_models = tags_data.get("models", []) if isinstance(tags_data, dict) else []

    local_by_base: dict[str, list[str]] = {}
    for model in local_models:
        raw = str(model.get("name", "")).strip()
        if not raw:
            continue
        base = raw.split(":", 1)[0].lower()
        local_by_base.setdefault(base, []).append(raw)

    curated = _by_name(POPULAR_MODELS)
    discovered = await _discover_library_names(limit=limit * 2)

    merged_names: list[str] = []
    seen: set[str] = set()
    for name in list(curated.keys()) + discovered:
        n = name.strip().lower()
        if not n or n in seen:
            continue
        seen.add(n)
        merged_names.append(n)

    items: list[dict[str, Any]] = []
    for name in merged_names:
        if q_norm and q_norm not in name:
            continue

        curated_item = curated.get(name)
        tag = (curated_item or {}).get("tag", "latest")
        desc = (curated_item or {}).get("description", "")
        pull_ref = f"{name}:{tag}"
        installed_tags = sorted(local_by_base.get(name, []))

        model_class = _classify_model(name)
        est_ram_gb = _estimate_ram_gb(name, tag, model_class)
        fit = _fit_tier(est_ram_gb, LOCALAI_HOST_RAM_GB)

        items.append(
            {
                "name": name,
                "tag": tag,
                "pull_ref": pull_ref,
                "description": desc,
                "installed": len(installed_tags) > 0,
                "installed_tags": installed_tags,
                "source": "curated" if curated_item else "library",
                "model_class": model_class,
                "est_ram_gb": est_ram_gb,
                "fit": fit,
            }
        )

        if len(items) >= limit:
            break

    return {
        "items": items,
        "query": q,
        "total": len(items),
        "host_ram_gb": LOCALAI_HOST_RAM_GB,
        "library_url": OLLAMA_LIBRARY_URL,
    }


@app.get("/api/metrics")
async def metrics() -> dict[str, Any]:
    tags_latency_start = perf_counter()
    ollama_up = True

    try:
        tags_data = await _ollama_get("/api/tags")
        tags_latency_ms = int((perf_counter() - tags_latency_start) * 1000)
    except Exception as e:  # noqa: BLE001
        ollama_up = False
        tags_latency_ms = -1
        return {
            "ollama_up": ollama_up,
            "tags_latency_ms": tags_latency_ms,
            "models_total": 0,
            "models_store_bytes": 0,
            "loaded_models": 0,
            "loaded_ram_bytes": 0,
            "loaded_vram_bytes": 0,
            "loaded_names": [],
            "actions": ACTION_STATE,
            "error": str(e),
        }

    try:
        ps_data = await _ollama_get("/api/ps")
    except Exception:
        ps_data = {"models": []}

    models = tags_data.get("models", []) if isinstance(tags_data, dict) else []
    loaded = ps_data.get("models", []) if isinstance(ps_data, dict) else []

    return {
        "ollama_up": ollama_up,
        "tags_latency_ms": tags_latency_ms,
        "models_total": len(models),
        "models_store_bytes": _sum_int(models, "size"),
        "loaded_models": len(loaded),
        "loaded_ram_bytes": _sum_int(loaded, "size"),
        "loaded_vram_bytes": _sum_int(loaded, "size_vram"),
        "loaded_names": [m.get("name", "") for m in loaded if isinstance(m, dict)],
        "actions": ACTION_STATE,
    }


@app.post("/api/models/pull")
async def pull_model(action: ModelAction) -> dict[str, Any]:
    return await _tracked_action(
        "pull",
        action.model,
        "/api/pull",
        {"model": action.model, "stream": True},
        stream=True,
    )


@app.post("/api/models/update")
async def update_model(action: ModelAction) -> dict[str, Any]:
    return await _tracked_action(
        "update",
        action.model,
        "/api/pull",
        {"model": action.model, "stream": True},
        stream=True,
    )


@app.post("/api/models/delete")
async def delete_model(action: ModelAction) -> dict[str, Any]:
    return await _tracked_action(
        "delete",
        action.model,
        "/api/delete",
        {"model": action.model},
        stream=False,
        method="DELETE",
    )


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return """<!doctype html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>LocalAI Model Admin</title>
    <style>
      :root {
        --bg: #0a0a0a;
        --surface: #111111;
        --surface-2: #151515;
        --border: #262626;
        --text: #f3f4f6;
        --muted: #a3a3a3;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        color: var(--text);
        font-family: Inter, ui-sans-serif, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
        background: #090909;
      }
      .page { max-width: 1160px; margin: 20px auto; padding: 0 14px; }
      .shell {
        border: 1px solid var(--border);
        border-radius: 12px;
        background: var(--surface);
        padding: 16px;
      }
      .top { display: flex; align-items: center; justify-content: space-between; gap: 16px; }
      .title { margin: 0; font-size: 36px; font-weight: 700; }
      .subtitle { margin: 5px 0 0; color: var(--muted); }
      .status {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        border: 1px solid var(--border);
        border-radius: 999px;
        padding: 7px 12px;
        background: var(--surface-2);
        color: var(--muted);
      }
      .dot { width: 8px; height: 8px; border-radius: 50%; background: #ef4444; }
      .dot.up { background: #22c55e; }
      .grid { margin-top: 12px; display: grid; gap: 8px; grid-template-columns: repeat(4, minmax(140px, 1fr)); }
      .card { border: 1px solid var(--border); border-radius: 10px; background: #121212; padding: 10px; }
      .label { font-size: 11px; color: var(--muted); margin-bottom: 4px; }
      .value { font-size: 29px; font-weight: 650; line-height: 1.15; }
      .subvalue { margin-top: 2px; font-size: 11px; color: var(--muted); min-height: 14px; }
      .spark { margin-top: 7px; width: 100%; height: 28px; display: block; }
      .spark path.grid { stroke: #2b2b2b; stroke-width: 1; }
      .spark polyline.line { fill: none; stroke: #d4d4d8; stroke-width: 1.8; stroke-linecap: round; stroke-linejoin: round; }
      .controls { margin-top: 12px; display: flex; gap: 8px; flex-wrap: wrap; }
      input, select, button {
        border-radius: 8px;
        border: 1px solid var(--border);
        color: var(--text);
        background: #0f0f10;
        padding: 9px 11px;
      }
      input { width: min(360px, 100%); }
      button { cursor: pointer; }
      button:hover { filter: brightness(1.08); }
      button.ok { border-color: #14532d; background: #0d1f12; }
      button.warn { border-color: #78350f; background: #251605; }
      button.danger { border-color: #7f1d1d; background: #2a0a0a; }
      .catalog {
        margin-top: 12px;
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 10px;
        background: #101010;
      }
      .catalog-head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 8px;
        margin-bottom: 8px;
      }
      .catalog-head h2 { margin: 0; font-size: 15px; }
      .catalog-tools { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
      .catalog-table { width: 100%; border-collapse: collapse; }
      .catalog-table th, .catalog-table td { text-align: left; border-bottom: 1px solid var(--border); padding: 7px; vertical-align: top; }
      .catalog-table th { color: var(--muted); font-weight: 500; font-size: 12px; }
      .chip { display: inline-block; border: 1px solid var(--border); color: var(--muted); padding: 1px 7px; border-radius: 999px; font-size: 11px; }
      .chip.installed { color: #86efac; border-color: #14532d; }
      .chip.remote { color: #d4d4d8; }
      .chip.class { color: #c4b5fd; border-color: #4c1d95; }
      .chip.fit-great { color: #86efac; border-color: #166534; }
      .chip.fit-good { color: #bef264; border-color: #4d7c0f; }
      .chip.fit-tight { color: #fcd34d; border-color: #92400e; }
      .chip.fit-heavy { color: #fca5a5; border-color: #991b1b; }
      pre {
        margin-top: 12px;
        margin-bottom: 0;
        min-height: 72px;
        border: 1px solid var(--border);
        border-radius: 10px;
        background: #0c0c0d;
        color: #d4d4d8;
        padding: 11px;
        overflow: auto;
        white-space: pre-wrap;
      }
      table { width: 100%; border-collapse: collapse; margin-top: 12px; }
      th, td { text-align: left; border-bottom: 1px solid var(--border); padding: 8px 7px; }
      th { color: var(--muted); font-weight: 500; font-size: 12px; }
      .muted { color: var(--muted); }
      @media (max-width: 960px) { .grid { grid-template-columns: repeat(2, minmax(140px, 1fr)); } }
      @media (max-width: 560px) {
        .top { flex-direction: column; align-items: flex-start; }
        .grid { grid-template-columns: 1fr; }
      }
    </style>
  </head>
  <body>
    <div class=\"page\">
      <div class=\"shell\">
        <div class=\"top\">
          <div>
            <h1 class=\"title\">LocalAI Model Admin</h1>
            <p class=\"subtitle\">Manage Ollama models and watch live runtime utilization.</p>
          </div>
          <div class=\"status\"><span id=\"status-dot\" class=\"dot\"></span><span id=\"status-text\">Ollama unavailable</span></div>
        </div>

        <div class=\"grid\">
          <div class=\"card\"><div class=\"label\">Stored Models</div><div id=\"m-models\" class=\"value\">-</div><div class=\"subvalue\">count</div><svg id=\"spark-models\" class=\"spark\" viewBox=\"0 0 160 28\" preserveAspectRatio=\"none\"></svg></div>
          <div class=\"card\"><div class=\"label\">Model Store Size</div><div id=\"m-store\" class=\"value\">-</div><div class=\"subvalue\">disk footprint</div><svg id=\"spark-store\" class=\"spark\" viewBox=\"0 0 160 28\" preserveAspectRatio=\"none\"></svg></div>
          <div class=\"card\"><div class=\"label\">Loaded Models</div><div id=\"m-loaded\" class=\"value\">-</div><div id=\"m-loaded-names\" class=\"subvalue\">none</div><svg id=\"spark-loaded\" class=\"spark\" viewBox=\"0 0 160 28\" preserveAspectRatio=\"none\"></svg></div>
          <div class=\"card\"><div class=\"label\">API Latency</div><div id=\"m-latency\" class=\"value\">-</div><div class=\"subvalue\">/api/tags</div><svg id=\"spark-latency\" class=\"spark\" viewBox=\"0 0 160 28\" preserveAspectRatio=\"none\"></svg></div>
          <div class=\"card\"><div class=\"label\">Loaded RAM</div><div id=\"m-ram\" class=\"value\">-</div><div class=\"subvalue\">model resident set</div><svg id=\"spark-ram\" class=\"spark\" viewBox=\"0 0 160 28\" preserveAspectRatio=\"none\"></svg></div>
          <div class=\"card\"><div class=\"label\">Loaded VRAM</div><div id=\"m-vram\" class=\"value\">-</div><div class=\"subvalue\">accelerator memory</div><svg id=\"spark-vram\" class=\"spark\" viewBox=\"0 0 160 28\" preserveAspectRatio=\"none\"></svg></div>
          <div class=\"card\"><div class=\"label\">Active Jobs</div><div id=\"m-jobs\" class=\"value\">-</div><div class=\"subvalue\">pull/update/delete</div><svg id=\"spark-jobs\" class=\"spark\" viewBox=\"0 0 160 28\" preserveAspectRatio=\"none\"></svg></div>
          <div class=\"card\"><div class=\"label\">Last Action</div><div id=\"m-last\" class=\"value\" style=\"font-size:16px\">-</div><div id=\"m-host-ram\" class=\"subvalue\"></div></div>
        </div>

        <div class=\"controls\">
          <input id=\"model\" placeholder=\"e.g. llama3.2:3b\" />
          <button class=\"ok\" onclick=\"run('pull')\">Pull</button>
          <button class=\"warn\" onclick=\"run('update')\">Update</button>
          <button class=\"danger\" onclick=\"run('delete')\">Delete Local</button>
          <button onclick=\"refreshAll()\">Refresh</button>
        </div>

        <div class=\"catalog\">
          <div class=\"catalog-head\">
            <h2>Browse Catalog</h2>
            <div class=\"catalog-tools\">
              <input id=\"catalog-q\" placeholder=\"Search models (llama, qwen, mistral...)\" />
              <select id=\"catalog-class\">
                <option value=\"all\">All Classes</option>
                <option value=\"chat\">Chat</option>
                <option value=\"reasoning\">Reasoning</option>
                <option value=\"code\">Code</option>
                <option value=\"embed\">Embed</option>
                <option value=\"vision\">Vision</option>
              </select>
              <select id=\"catalog-fit\">
                <option value=\"all\">All Fit</option>
                <option value=\"great\">Great</option>
                <option value=\"good\">Good</option>
                <option value=\"tight\">Tight</option>
                <option value=\"heavy\">Heavy</option>
              </select>
              <button onclick=\"refreshCatalog()\">Search</button>
            </div>
          </div>
          <table class=\"catalog-table\">
            <thead>
              <tr><th>Model</th><th>Recommended Pull</th><th>Class</th><th>Fit For This Mac</th><th>Status</th><th>Source</th><th></th></tr>
            </thead>
            <tbody id=\"catalog\"></tbody>
          </table>
        </div>

        <pre id=\"log\">Ready.</pre>

        <table>
          <thead>
            <tr><th>Model</th><th>Size</th><th>Modified</th></tr>
          </thead>
          <tbody id=\"models\"></tbody>
        </table>
      </div>
    </div>
    <script>
      const modelInput = document.getElementById('model');
      const logBox = document.getElementById('log');
      const modelsTable = document.getElementById('models');
      const catalogTable = document.getElementById('catalog');
      const catalogInput = document.getElementById('catalog-q');
      const catalogClass = document.getElementById('catalog-class');
      const catalogFit = document.getElementById('catalog-fit');
      const statusDot = document.getElementById('status-dot');
      const statusText = document.getElementById('status-text');
      const MAX_POINTS = 80;
      let catalogCache = [];
      let hostRamGb = null;
      let catalogTimer = null;
      const history = {
        models: [],
        store: [],
        loaded: [],
        latency: [],
        ram: [],
        vram: [],
        jobs: []
      };

      function bytes(n) {
        if (!n && n !== 0) return '-';
        const units = ['B', 'KB', 'MB', 'GB', 'TB'];
        let i = 0;
        let v = Number(n);
        while (v >= 1024 && i < units.length - 1) {
          v /= 1024;
          i += 1;
        }
        return `${v.toFixed(2)} ${units[i]}`;
      }

      function fmtDate(ts) {
        if (!ts) return '-';
        const d = new Date(ts);
        if (Number.isNaN(d.getTime())) return '-';
        const y = d.getFullYear();
        const m = String(d.getMonth() + 1).padStart(2, '0');
        const day = String(d.getDate()).padStart(2, '0');
        return `${y}-${m}-${day}`;
      }

      function esc(s) {
        return String(s ?? '')
          .replaceAll('&', '&amp;')
          .replaceAll('<', '&lt;')
          .replaceAll('>', '&gt;')
          .replaceAll('"', '&quot;')
          .replaceAll("'", '&#39;');
      }

      function setText(id, value) {
        const el = document.getElementById(id);
        if (el) el.textContent = value;
      }

      function pushMetric(key, value) {
        if (!history[key]) return;
        const v = Number(value);
        if (!Number.isFinite(v)) return;
        const arr = history[key];
        arr.push(v);
        if (arr.length > MAX_POINTS) arr.shift();
      }

      function renderSpark(id, values) {
        const svg = document.getElementById(id);
        if (!svg) return;
        if (!values || values.length < 2) {
          svg.innerHTML = '<path class="grid" d="M0 27 H160" />';
          return;
        }
        const w = 160;
        const h = 28;
        let min = Math.min(...values);
        let max = Math.max(...values);
        if (max === min) {
          min -= 1;
          max += 1;
        }
        const points = values.map((v, i) => {
          const x = (i / (values.length - 1)) * (w - 2) + 1;
          const y = h - 2 - ((v - min) / (max - min)) * (h - 6);
          return `${x.toFixed(1)},${y.toFixed(1)}`;
        }).join(' ');
        svg.innerHTML =
          `<path class="grid" d="M0 27 H160" />` +
          `<polyline class="line" points="${points}" />`;
      }

      function renderSparks() {
        renderSpark('spark-models', history.models);
        renderSpark('spark-store', history.store);
        renderSpark('spark-loaded', history.loaded);
        renderSpark('spark-latency', history.latency);
        renderSpark('spark-ram', history.ram);
        renderSpark('spark-vram', history.vram);
        renderSpark('spark-jobs', history.jobs);
      }

      function fitLabel(fit, est) {
        const name = fit || 'unknown';
        return `${name} (${est.toFixed(1)} GB est)`;
      }

      function renderCatalogRows() {
        const q = catalogInput.value.trim().toLowerCase();
        const cls = catalogClass.value;
        const fit = catalogFit.value;

        const filtered = catalogCache.filter((item) => {
          if (q && !(item.name || '').includes(q)) return false;
          if (cls !== 'all' && item.model_class !== cls) return false;
          if (fit !== 'all' && item.fit !== fit) return false;
          return true;
        });

        const rows = filtered.map((item) => {
          const installedChip = item.installed
            ? `<span class="chip installed">installed</span>`
            : '<span class="chip">not installed</span>';
          const sourceChip = item.source === 'curated'
            ? '<span class="chip">curated</span>'
            : '<span class="chip remote">library</span>';
          const classChip = `<span class="chip class">${esc(item.model_class)}</span>`;
          const fitChip = `<span class="chip fit-${esc(item.fit)}">${esc(fitLabel(item.fit, item.est_ram_gb || 0))}</span>`;
          const desc = item.description ? `<div class="muted">${esc(item.description)}</div>` : '';

          return `<tr>
            <td><div>${esc(item.name)}</div>${desc}</td>
            <td><code>${esc(item.pull_ref)}</code></td>
            <td>${classChip}</td>
            <td>${fitChip}</td>
            <td>${installedChip}</td>
            <td>${sourceChip}</td>
            <td><button class="ok" onclick="run('pull', '${esc(item.pull_ref)}')">Pull</button></td>
          </tr>`;
        });
        catalogTable.innerHTML = rows.join('') || '<tr><td colspan="7" class="muted">No matching catalog results.</td></tr>';
      }

      async function refreshModels() {
        const r = await fetch('/api/models');
        const data = await r.json();
        const rows = (data.models || []).map((m) => {
          const modified = fmtDate(m.modified_at);
          const full = m.modified_at || '';
          return `<tr><td>${esc(m.name || '-')}</td><td>${bytes(m.size)}</td><td title="${esc(full)}">${modified}</td></tr>`;
        });
        modelsTable.innerHTML = rows.join('') || '<tr><td colspan="3" class="muted">No models found.</td></tr>';
      }

      async function refreshCatalog() {
        const q = encodeURIComponent(catalogInput.value.trim());
        const r = await fetch(`/api/catalog?q=${q}&limit=80`);
        const data = await r.json();
        catalogCache = data.items || [];
        hostRamGb = data.host_ram_gb || hostRamGb;
        if (hostRamGb) {
          setText('m-host-ram', `Host RAM profile: ${hostRamGb} GB`);
        }
        renderCatalogRows();
      }

      async function refreshMetrics() {
        const r = await fetch('/api/metrics');
        const m = await r.json();

        statusDot.className = m.ollama_up ? 'dot up' : 'dot';
        statusText.textContent = m.ollama_up ? 'Ollama online' : 'Ollama unavailable';

        setText('m-models', String(m.models_total ?? '-'));
        setText('m-store', bytes(m.models_store_bytes));
        setText('m-loaded', String(m.loaded_models ?? '-'));
        setText('m-loaded-names', (m.loaded_names || []).join(', ') || 'none');
        setText('m-latency', m.tags_latency_ms >= 0 ? `${m.tags_latency_ms} ms` : '-');
        setText('m-ram', bytes(m.loaded_ram_bytes));
        setText('m-vram', bytes(m.loaded_vram_bytes));
        setText('m-jobs', `${m.actions?.active_jobs ?? 0} active / ${m.actions?.completed_jobs ?? 0} done`);

        pushMetric('models', m.models_total ?? 0);
        pushMetric('store', m.models_store_bytes ?? 0);
        pushMetric('loaded', m.loaded_models ?? 0);
        pushMetric('latency', m.tags_latency_ms >= 0 ? m.tags_latency_ms : 0);
        pushMetric('ram', m.loaded_ram_bytes ?? 0);
        pushMetric('vram', m.loaded_vram_bytes ?? 0);
        pushMetric('jobs', m.actions?.active_jobs ?? 0);
        renderSparks();

        const lastAction = m.actions?.last_action;
        const lastModel = m.actions?.last_model;
        const lastMs = m.actions?.last_duration_ms;
        if (lastAction && lastModel) {
          setText('m-last', `${lastAction} ${lastModel} (${lastMs} ms)`);
        } else {
          setText('m-last', '-');
        }
      }

      async function run(action, overrideModel = null) {
        const model = (overrideModel || modelInput.value).trim();
        if (!model) {
          logBox.textContent = 'Enter a model name first.';
          return;
        }

        modelInput.value = model;
        logBox.textContent = `${action} ${model}...`;
        try {
          const r = await fetch(`/api/models/${action}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model })
          });
          const data = await r.json();
          if (!r.ok) {
            throw new Error(JSON.stringify(data));
          }
          logBox.textContent = JSON.stringify(data, null, 2);
          await refreshAll();
        } catch (e) {
          logBox.textContent = `${action} failed: ${e}`;
          await refreshMetrics();
        }
      }

      async function refreshAll() {
        try {
          await Promise.all([refreshModels(), refreshMetrics(), refreshCatalog()]);
        } catch (e) {
          logBox.textContent = `Refresh failed: ${e}`;
        }
      }

      catalogInput.addEventListener('input', () => {
        if (catalogTimer) clearTimeout(catalogTimer);
        catalogTimer = setTimeout(() => {
          refreshCatalog().catch((e) => {
            logBox.textContent = `Catalog refresh failed: ${e}`;
          });
        }, 250);
      });
      catalogClass.addEventListener('change', renderCatalogRows);
      catalogFit.addEventListener('change', renderCatalogRows);

      refreshAll();
      setInterval(refreshMetrics, 1500);
      setInterval(refreshModels, 10000);
      setInterval(refreshCatalog, 30000);
    </script>
  </body>
</html>
"""
