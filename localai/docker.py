from __future__ import annotations

import shutil
from pathlib import Path

from .config import StackConfig
from .shell import run


def has_docker() -> bool:
    return shutil.which("docker") is not None


def _compose_base_cmd(cfg: StackConfig) -> list[str]:
    cmd = ["docker", "compose"]
    env_file = Path(cfg.root) / ".localai.env"
    if env_file.exists():
        cmd.extend(["--env-file", str(env_file)])
    cmd.extend(["-f", str(cfg.docker.compose_file)])
    return cmd


def compose_up(cfg: StackConfig) -> None:
    cmd = _compose_base_cmd(cfg) + ["up", "-d"]
    if cfg.docker.services:
        cmd.extend(cfg.docker.services)
    run(cmd, check=True, cwd=str(cfg.root))


def compose_down(cfg: StackConfig) -> None:
    run(_compose_base_cmd(cfg) + ["down"], check=True, cwd=str(cfg.root))


def compose_ps(cfg: StackConfig) -> str:
    res = run(_compose_base_cmd(cfg) + ["ps"], cwd=str(cfg.root))
    return res.stdout or res.stderr


def compose_logs(cfg: StackConfig, follow: bool = False, tail: int = 200) -> None:
    cmd = _compose_base_cmd(cfg) + ["logs", f"--tail={tail}"]
    if follow:
        cmd.append("-f")
    run(cmd, check=True, cwd=str(cfg.root))
