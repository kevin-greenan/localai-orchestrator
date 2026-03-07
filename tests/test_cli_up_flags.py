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

    def test_image_gen_flag_enables_image_services(self):
        args = self._args("--image-gen")
        cfg = self._cfg()
        cfg.image_gen.enabled = False
        cfg.image_gen.provider = "automatic1111"
        cfg.image_gen.a1111_url = "http://host.docker.internal:7860"

        with tempfile.TemporaryDirectory() as td:
            cfg.root = Path(td)
            with (
                patch.object(cli, "_load_cfg_with_tuning", return_value=(cfg, _fake_tuning())),
                patch.object(cli, "_validate_image_gen_provider"),
                patch.object(cli, "compose_up"),
                patch.object(cli, "start_ollama_launch_agent"),
            ):
                rc = cli._cmd_up(args)

            self.assertEqual(rc, 0)
            env = _read_env(Path(td) / ".localai.env")
            services = set(env["LOCALAI_DOCKER_SERVICES"].split(","))
            self.assertIn("image-gen", services)
            self.assertIn("minio", services)
            self.assertIn("image-redis", services)
            self.assertEqual(env["LOCALAI_IMAGE_GEN_ENABLED"], "1")

    def test_expose_port_is_ignored_when_no_webui(self):
        args = self._args("--no-webui", "--expose", "8080")
        cfg = self._cfg()

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
            self.assertEqual(env["LOCALAI_OPENWEBUI_PORT"], "3000")
            self.assertEqual(env["LOCALAI_BIND_IP"], "0.0.0.0")

    def test_image_gen_requires_a1111_url_when_enabled(self):
        args = self._args("--image-gen")
        cfg = self._cfg()
        cfg.image_gen.enabled = True
        cfg.image_gen.provider = "automatic1111"
        cfg.image_gen.a1111_url = ""

        with patch.object(cli, "_load_cfg_with_tuning", return_value=(cfg, _fake_tuning())):
            with self.assertRaisesRegex(RuntimeError, "image_gen.a1111_url is required"):
                cli._cmd_up(args)

    def test_runtime_env_includes_vision_and_image_gen_flags(self):
        args = self._args()
        cfg = self._cfg()
        cfg.vision.enabled = True
        cfg.vision.default_model = "llava:latest"
        cfg.vision.max_image_mb = 12
        cfg.vision.benchmark_dataset = "tests/fixtures/vision/smoke.jsonl"
        cfg.image_gen.enabled = True
        cfg.image_gen.provider = "automatic1111"
        cfg.image_gen.concurrency = 2
        cfg.image_gen.queue_timeout_seconds = 450
        cfg.image_gen.artifact_store = "minio"
        cfg.image_gen.backend_url = "http://image-gen:8090"
        cfg.image_gen.a1111_url = "http://host.docker.internal:7860"
        cfg.image_gen.openwebui_model = "localai-imagegen"
        cfg.image_gen.openwebui_image_size = "1024x1024"

        with tempfile.TemporaryDirectory() as td:
            cfg.root = Path(td)
            with (
                patch.object(cli, "_load_cfg_with_tuning", return_value=(cfg, _fake_tuning())),
                patch.object(cli, "_validate_image_gen_provider"),
                patch.object(cli, "compose_up"),
                patch.object(cli, "start_ollama_launch_agent"),
            ):
                rc = cli._cmd_up(args)

            self.assertEqual(rc, 0)
            env = _read_env(Path(td) / ".localai.env")
            self.assertEqual(env["LOCALAI_VISION_ENABLED"], "1")
            self.assertEqual(env["LOCALAI_VISION_DEFAULT_MODEL"], "llava:latest")
            self.assertEqual(env["LOCALAI_VISION_MAX_IMAGE_MB"], "16")
            self.assertEqual(env["LOCALAI_VISION_BENCHMARK_DATASET"], "tests/fixtures/vision/smoke.jsonl")
            self.assertEqual(env["LOCALAI_IMAGE_GEN_ENABLED"], "1")
            self.assertEqual(env["LOCALAI_IMAGE_GEN_PROVIDER"], "automatic1111")
            self.assertEqual(env["LOCALAI_IMAGE_GEN_CONCURRENCY"], "2")
            self.assertEqual(env["LOCALAI_IMAGE_GEN_QUEUE_TIMEOUT_SECONDS"], "450")
            self.assertEqual(env["LOCALAI_IMAGE_GEN_ARTIFACT_STORE"], "minio")
            self.assertEqual(env["LOCALAI_IMAGE_GEN_BACKEND_URL"], "http://image-gen:8090")
            self.assertEqual(env["LOCALAI_IMAGE_GEN_A1111_URL"], "http://host.docker.internal:7860")
            self.assertEqual(env["LOCALAI_OPENWEBUI_ENABLE_IMAGE_GENERATION"], "True")
            self.assertEqual(env["LOCALAI_OPENWEBUI_IMAGE_GENERATION_ENGINE"], "openai")
            self.assertEqual(env["LOCALAI_OPENWEBUI_IMAGE_GENERATION_MODEL"], "localai-imagegen")
            self.assertEqual(env["LOCALAI_OPENWEBUI_IMAGES_OPENAI_API_BASE_URL"], "http://image-gen:8090/v1")
            self.assertEqual(env["LOCALAI_OPENWEBUI_IMAGE_SIZE"], "1024x1024")
            self.assertEqual(env["LOCALAI_IMAGE_GEN_REDIS_MAXMEMORY_MB"], "512")
            services = set(env["LOCALAI_DOCKER_SERVICES"].split(","))
            self.assertIn("image-gen", services)
            self.assertIn("minio", services)
            self.assertIn("image-redis", services)


if __name__ == "__main__":
    unittest.main()
