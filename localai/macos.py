from __future__ import annotations

from pathlib import Path
import os
import platform
import shutil

from .config import StackConfig
from .shell import run


def is_macos() -> bool:
    return platform.system() == "Darwin"


def is_apple_silicon() -> bool:
    return platform.machine() == "arm64"


def ollama_bin() -> str | None:
    return shutil.which("ollama")


def launch_agent_path(label: str) -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"


def _plist_text(label: str, bin_path: str, env: dict[str, str], stdout_path: Path, stderr_path: Path) -> str:
    env_items = "\n".join(
        f"    <key>{k}</key>\n    <string>{v}</string>" for k, v in sorted(env.items())
    )
    return f"""<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">
<plist version=\"1.0\">
<dict>
  <key>Label</key>
  <string>{label}</string>
  <key>ProgramArguments</key>
  <array>
    <string>{bin_path}</string>
    <string>serve</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
{env_items}
  </dict>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>{stdout_path}</string>
  <key>StandardErrorPath</key>
  <string>{stderr_path}</string>
</dict>
</plist>
"""


def build_ollama_env(cfg: StackConfig) -> dict[str, str]:
    env = {
        "OLLAMA_HOST": f"{cfg.ollama.host}:{cfg.ollama.port}",
        "OLLAMA_KEEP_ALIVE": cfg.ollama.keep_alive,
        "OLLAMA_NUM_PARALLEL": str(cfg.ollama.num_parallel),
        "OLLAMA_MAX_LOADED_MODELS": str(cfg.ollama.max_loaded_models),
        # Conservative default avoids disk eviction churn and keeps repeated workloads hot.
        "OLLAMA_NOPRUNE": "1",
    }
    env.update(cfg.ollama.env)
    return env


def _launchctl_domains(uid: str) -> list[str]:
    # `gui/<uid>` is common in desktop sessions, while some newer/session-specific
    # environments only accept `user/<uid>`.
    return [f"gui/{uid}", f"user/{uid}"]


def start_ollama_launch_agent(cfg: StackConfig) -> None:
    if not is_macos():
        raise RuntimeError("This orchestrator is macOS-only for native Ollama acceleration.")

    bin_path = ollama_bin()
    if not bin_path:
        raise RuntimeError("Ollama binary not found in PATH. Install Ollama first.")

    label = cfg.ollama.label
    plist_path = launch_agent_path(label)
    plist_path.parent.mkdir(parents=True, exist_ok=True)

    log_dir = Path.home() / ".localai" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / "ollama.out.log"
    stderr_path = log_dir / "ollama.err.log"

    env = build_ollama_env(cfg)
    plist_path.write_text(_plist_text(label, bin_path, env, stdout_path, stderr_path), encoding="utf-8")

    uid = str(os.getuid())
    errors: list[str] = []
    for domain in _launchctl_domains(uid):
        run(["launchctl", "bootout", domain, str(plist_path)])
        b = run(["launchctl", "bootstrap", domain, str(plist_path)])
        if b.code != 0:
            errors.append(f"{domain} bootstrap failed: {b.stderr or b.stdout}")
            continue
        run(["launchctl", "enable", f"{domain}/{label}"])
        k = run(["launchctl", "kickstart", "-k", f"{domain}/{label}"])
        if k.code == 0:
            return
        errors.append(f"{domain} kickstart failed: {k.stderr or k.stdout}")

    detail = " | ".join(errors) if errors else "unknown launchctl error"
    raise RuntimeError(f"could not start launch agent in gui/user domains: {detail}")


def stop_ollama_launch_agent(cfg: StackConfig) -> None:
    label = cfg.ollama.label
    plist_path = launch_agent_path(label)
    uid = str(os.getuid())
    for domain in _launchctl_domains(uid):
        run(["launchctl", "bootout", domain, str(plist_path)])


def launch_agent_status(label: str) -> str:
    uid = str(os.getuid())
    for domain in _launchctl_domains(uid):
        res = run(["launchctl", "print", f"{domain}/{label}"])
        if res.code == 0:
            return "running"
    return "stopped"
