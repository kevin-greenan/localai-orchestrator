from __future__ import annotations

import io
import json
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


class BenchmarkTests(unittest.TestCase):
    def _cfg(self):
        return load_stack(Path(__file__).resolve().parents[1] / "stack.toml")

    def _args(self, *argv: str):
        return cli.build_parser().parse_args(["benchmark", *argv])

    def test_benchmark_uses_default_model_and_reports_summary(self):
        args = self._args("--iterations", "2")
        cfg = self._cfg()

        with tempfile.TemporaryDirectory() as td:
            cfg.root = Path(td)
            fake_out = io.StringIO()
            with (
                patch.object(cli, "_load_cfg_with_tuning", return_value=(cfg, _fake_tuning())),
                patch.object(cli, "_effective_services", return_value=[]),
                patch.object(cli, "_probe_http", return_value={"ok": True, "latency_ms": 5.0, "detail": "ok"}),
                patch.object(
                    cli,
                    "ollama_generate_json",
                    side_effect=[
                        (
                            True,
                            {
                                "eval_count": 30,
                                "eval_duration": 2_000_000_000,
                                "total_duration": 2_500_000_000,
                                "load_duration": 200_000_000,
                                "prompt_eval_count": 10,
                                "prompt_eval_duration": 500_000_000,
                            },
                            "",
                        ),
                        (
                            True,
                            {
                                "eval_count": 40,
                                "eval_duration": 2_000_000_000,
                                "total_duration": 2_600_000_000,
                                "load_duration": 150_000_000,
                                "prompt_eval_count": 10,
                                "prompt_eval_duration": 400_000_000,
                            },
                            "",
                        ),
                    ],
                ),
                patch("sys.stdout", new=fake_out),
            ):
                rc = cli._cmd_benchmark(args)

        self.assertEqual(rc, 0)
        payload = json.loads(fake_out.getvalue())
        self.assertEqual(payload["model"], cfg.ollama.warmup_model)
        self.assertEqual(payload["summary"]["iterations"], 2)
        self.assertEqual(payload["summary"]["successful_runs"], 2)
        self.assertEqual(payload["summary"]["failed_runs"], 0)
        self.assertGreater(payload["summary"]["tokens_per_second_avg"], 0)
        self.assertEqual(len(payload["samples"]), 2)

    def test_benchmark_returns_nonzero_when_all_runs_fail(self):
        args = self._args("--iterations", "2", "--model", "llama3.2:3b")
        cfg = self._cfg()

        with tempfile.TemporaryDirectory() as td:
            cfg.root = Path(td)
            fake_out = io.StringIO()
            with (
                patch.object(cli, "_load_cfg_with_tuning", return_value=(cfg, _fake_tuning())),
                patch.object(cli, "_effective_services", return_value=[]),
                patch.object(cli, "_probe_http", return_value={"ok": True, "latency_ms": 3.0, "detail": "ok"}),
                patch.object(cli, "ollama_generate_json", return_value=(False, {}, "request timeout")),
                patch("sys.stdout", new=fake_out),
            ):
                rc = cli._cmd_benchmark(args)

        self.assertEqual(rc, 1)
        payload = json.loads(fake_out.getvalue())
        self.assertEqual(payload["summary"]["failed_runs"], 2)
        self.assertEqual(payload["summary"]["successful_runs"], 0)

    def test_benchmark_requires_positive_iterations(self):
        args = self._args("--iterations", "0")
        cfg = self._cfg()

        with patch.object(cli, "_load_cfg_with_tuning", return_value=(cfg, _fake_tuning())):
            with self.assertRaisesRegex(RuntimeError, "--iterations must be at least 1"):
                cli._cmd_benchmark(args)


if __name__ == "__main__":
    unittest.main()
