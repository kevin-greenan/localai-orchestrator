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


class VisionCliTests(unittest.TestCase):
    def _cfg(self):
        return load_stack(Path(__file__).resolve().parents[1] / "stack.toml")

    def test_vision_smoke_requires_image_argument(self):
        args = cli.build_parser().parse_args(["test", "vision-smoke"])
        cfg = self._cfg()
        cfg.vision.enabled = True

        with patch.object(cli, "_load_cfg_with_tuning", return_value=(cfg, _fake_tuning())):
            with self.assertRaisesRegex(RuntimeError, "one of --image or --image-url"):
                cli._cmd_test_vision_smoke(args)

    def test_vision_smoke_posts_to_model_admin(self):
        args = cli.build_parser().parse_args(["test", "vision-smoke", "--image", "sample.png"])
        cfg = self._cfg()
        cfg.vision.enabled = True

        with tempfile.TemporaryDirectory() as td:
            cfg.root = Path(td)
            img = Path(td) / "sample.png"
            img.write_bytes(b"fake-image")
            args.image = str(img)
            out_stream = io.StringIO()
            with (
                patch.object(cli, "_load_cfg_with_tuning", return_value=(cfg, _fake_tuning())),
                patch.object(cli, "_post_json", return_value={"ok": True, "summary": {"failed": 0}}) as mock_post,
                patch("sys.stdout", new=out_stream),
            ):
                rc = cli._cmd_test_vision_smoke(args)

        self.assertEqual(rc, 0)
        self.assertIn('"ok": true', out_stream.getvalue())
        self.assertTrue(mock_post.called)

    def test_benchmark_vision_dispatches_and_returns_failure_code(self):
        args = cli.build_parser().parse_args(["benchmark", "vision", "--iterations", "2"])
        cfg = self._cfg()
        cfg.vision.enabled = True
        out_stream = io.StringIO()
        with (
            patch.object(cli, "_load_cfg_with_tuning", return_value=(cfg, _fake_tuning())),
            patch.object(
                cli,
                "_post_json",
                return_value={"summary": {"failed_runs": 1}, "samples": []},
            ),
            patch("sys.stdout", new=out_stream),
        ):
            rc = cli._cmd_benchmark(args)

        self.assertEqual(rc, 1)
        parsed = json.loads(out_stream.getvalue())
        self.assertEqual(parsed["summary"]["failed_runs"], 1)


if __name__ == "__main__":
    unittest.main()
