from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from localai import cli
from localai.config import load_stack
from localai.tuning import SystemSpecs, TuningResult


def _fake_tuning() -> TuningResult:
    return TuningResult(
        enabled=True,
        specs=SystemSpecs(
            mem_gb=32,
            logical_cpu=8,
            physical_cpu=8,
            perf_cores=4,
            model="test-mac",
            machine="arm64",
        ),
        applied={},
        recommendations={},
    )


def _read_env(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line or line.startswith("#"):
            continue
        k, v = line.split("=", 1)
        out[k] = v
    return out


class UpFlagTests(unittest.TestCase):
    def _args(self, *argv: str):
        return cli.build_parser().parse_args(["up", *argv])

    def _cfg(self):
        cfg = load_stack(Path(__file__).resolve().parents[1] / "stack.toml")
        return cfg

    def test_web_search_cannot_be_combined_with_no_webui(self):
        args = self._args("--no-webui", "--web-search")
        cfg = self._cfg()

        with patch.object(cli, "_load_cfg_with_tuning", return_value=(cfg, _fake_tuning())):
            with self.assertRaisesRegex(RuntimeError, "--web-search cannot be combined with --no-webui"):
                cli._cmd_up(args)

    def test_rag_preset_cannot_be_combined_with_no_webui(self):
        args = self._args("--no-webui", "--rag-preset", "deep")
        cfg = self._cfg()

        with patch.object(cli, "_load_cfg_with_tuning", return_value=(cfg, _fake_tuning())):
            with self.assertRaisesRegex(RuntimeError, "--rag-preset cannot be combined with --no-webui"):
                cli._cmd_up(args)

    def test_no_webui_disables_web_mode_and_services(self):
        args = self._args("--no-webui")
        cfg = self._cfg()
        cfg.web.enabled = True

        with tempfile.TemporaryDirectory() as td:
            cfg.root = Path(td)
            with (
                patch.object(cli, "_load_cfg_with_tuning", return_value=(cfg, _fake_tuning())),
                patch.object(cli, "compose_up"),
                patch.object(cli, "start_ollama_launch_agent"),
            ):
                rc = cli._cmd_up(args)

            self.assertEqual(rc, 0)
            self.assertFalse(cfg.web.enabled)
            env = _read_env(Path(td) / ".localai.env")
            self.assertEqual(env["LOCALAI_WEB_ENABLED"], "false")
            self.assertEqual(env["LOCALAI_DOCKER_SERVICES"], "model-admin,qdrant")

    def test_web_search_enables_search_services(self):
        args = self._args("--web-search")
        cfg = self._cfg()
        cfg.web.enabled = False

        with tempfile.TemporaryDirectory() as td:
            cfg.root = Path(td)
            with (
                patch.object(cli, "_load_cfg_with_tuning", return_value=(cfg, _fake_tuning())),
                patch.object(cli, "compose_up"),
                patch.object(cli, "start_ollama_launch_agent"),
            ):
                rc = cli._cmd_up(args)

            self.assertEqual(rc, 0)
            env = _read_env(Path(td) / ".localai.env")
            services = set(env["LOCALAI_DOCKER_SERVICES"].split(","))
            self.assertIn("searxng", services)
            self.assertIn("redis", services)
            self.assertEqual(env["LOCALAI_WEB_ENABLED"], "true")


if __name__ == "__main__":
    unittest.main()
