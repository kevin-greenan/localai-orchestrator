from __future__ import annotations

from urllib.request import Request, urlopen
import json


def http_ok(url: str, timeout: float = 3.0) -> tuple[bool, str]:
    req = Request(url, method="GET")
    try:
        with urlopen(req, timeout=timeout) as resp:
            status = resp.status
            body = resp.read().decode("utf-8", errors="ignore")
            if 200 <= status < 400:
                return True, body
            return False, f"status={status} body={body[:200]}"
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def ollama_generate(host: str, model: str, prompt: str, timeout: float = 20.0) -> tuple[bool, str]:
    url = f"http://{host}/api/generate"
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode("utf-8")
    req = Request(url, method="POST", data=payload, headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
            if 200 <= resp.status < 300:
                return True, body[:500]
            return False, f"status={resp.status} body={body[:200]}"
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def ollama_generate_json(
    host: str, model: str, prompt: str, timeout: float = 60.0
) -> tuple[bool, dict[str, object], str]:
    url = f"http://{host}/api/generate"
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode("utf-8")
    req = Request(url, method="POST", data=payload, headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
            if not (200 <= resp.status < 300):
                return False, {}, f"status={resp.status} body={body[:200]}"
            parsed = json.loads(body)
            if not isinstance(parsed, dict):
                return False, {}, "ollama response was not a JSON object"
            return True, parsed, ""
    except Exception as e:  # noqa: BLE001
        return False, {}, str(e)
