from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from localai.config import load_stack


class VisionImageGenConfigTests(unittest.TestCase):
    def test_load_stack_parses_vision_and_image_gen_sections(self):
        content = """
[project]
name = "localai-mac"

[native.ollama]
enabled = true
host = "127.0.0.1"
port = 11434
models = ["llama3.2:3b"]
warmup_model = "llama3.2:3b"
warmup_prompt = "warm"

[docker]
compose_file = "docker-compose.yml"
services = ["openwebui", "model-admin", "qdrant"]

[vision]
enabled = true
default_model = "llava:latest"
max_image_mb = 16
benchmark_dataset = "tests/fixtures/vision/smoke.jsonl"

[image_gen]
enabled = true
provider = "automatic1111"
concurrency = 2
queue_timeout_seconds = 450
artifact_store = "minio"
backend_url = "http://image-gen:8090"
a1111_url = "http://host.docker.internal:7860"
openwebui_model = "localai-imagegen"
openwebui_image_size = "1024x1024"
"""
        with tempfile.TemporaryDirectory() as td:
            stack_path = Path(td) / "stack.toml"
            stack_path.write_text(content, encoding="utf-8")
            cfg = load_stack(stack_path)

        self.assertTrue(cfg.vision.enabled)
        self.assertEqual(cfg.vision.default_model, "llava:latest")
        self.assertEqual(cfg.vision.max_image_mb, 16)
        self.assertEqual(cfg.vision.benchmark_dataset, "tests/fixtures/vision/smoke.jsonl")

        self.assertTrue(cfg.image_gen.enabled)
        self.assertEqual(cfg.image_gen.provider, "automatic1111")
        self.assertEqual(cfg.image_gen.concurrency, 2)
        self.assertEqual(cfg.image_gen.queue_timeout_seconds, 450)
        self.assertEqual(cfg.image_gen.artifact_store, "minio")
        self.assertEqual(cfg.image_gen.backend_url, "http://image-gen:8090")
        self.assertEqual(cfg.image_gen.a1111_url, "http://host.docker.internal:7860")
        self.assertEqual(cfg.image_gen.openwebui_model, "localai-imagegen")
        self.assertEqual(cfg.image_gen.openwebui_image_size, "1024x1024")


if __name__ == "__main__":
    unittest.main()
