from __future__ import annotations

from dataclasses import dataclass
import os
import subprocess


@dataclass(slots=True)
class CmdResult:
    code: int
    stdout: str
    stderr: str


def run(cmd: list[str], *, check: bool = False, cwd: str | None = None, env: dict[str, str] | None = None) -> CmdResult:
    merged_env = None
    if env is not None:
        merged_env = os.environ.copy()
        merged_env.update(env)

    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=merged_env,
        capture_output=True,
        text=True,
    )
    res = CmdResult(proc.returncode, proc.stdout.strip(), proc.stderr.strip())
    if check and res.code != 0:
        err = res.stderr or res.stdout or f"Command failed: {' '.join(cmd)}"
        raise RuntimeError(err)
    return res
