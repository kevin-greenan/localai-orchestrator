from __future__ import annotations

import importlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch


def _load_admin_module():
    try:
        return importlib.import_module("model_admin.app")
    except Exception:
        return None


admin = _load_admin_module()
if admin is not None:
    try:
        from fastapi.testclient import TestClient
    except Exception:
        TestClient = None
else:
    TestClient = None


@unittest.skipUnless(admin is not None and TestClient is not None, "model_admin dependencies not installed")
class ModelAdminEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(admin.app)

    def test_smoke_returns_failure_when_tags_unreachable(self):
        with patch.object(admin, "_ollama_get", new=AsyncMock(side_effect=RuntimeError("ollama down"))):
            resp = self.client.post("/api/tests/smoke", json={})

        self.assertEqual(resp.status_code, 200)
        payload = resp.json()
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["summary"]["failed"], 1)
        self.assertEqual(payload["checks"][0]["name"], "ollama_tags")
        self.assertFalse(payload["checks"][0]["ok"])

    def test_smoke_success_with_optional_checks_skipped(self):
        with (
            patch.object(admin, "LOCALAI_QDRANT_ENABLED", False),
            patch.object(admin, "LOCALAI_WEB_ENABLED", False),
            patch.object(
                admin,
                "_ollama_get",
                new=AsyncMock(side_effect=[{"models": []}, {"models": []}]),
            ),
        ):
            resp = self.client.post("/api/tests/smoke", json={})

        self.assertEqual(resp.status_code, 200)
        payload = resp.json()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["summary"]["failed"], 0)
        self.assertEqual(payload["summary"]["skipped"], 5)
        checks = {c["name"]: c for c in payload["checks"]}
        self.assertTrue(checks["ollama_tags"]["ok"])
        self.assertTrue(checks["ollama_ps"]["ok"])
        self.assertTrue(checks["ollama_generate"]["skipped"])
        self.assertTrue(checks["qdrant_health"]["skipped"])
        self.assertTrue(checks["web_stack_health"]["skipped"])
        self.assertTrue(checks["image_gen_health"]["skipped"])
        self.assertTrue(checks["image_gen_generate"]["skipped"])

    def test_benchmark_requires_any_available_model(self):
        with patch.object(admin, "_ollama_get", new=AsyncMock(return_value={"models": []})):
            resp = self.client.post("/api/benchmarks/run", json={"iterations": 2})

        self.assertEqual(resp.status_code, 400)
        self.assertIn("no generation-capable model", resp.json().get("detail", ""))

    def test_benchmark_auto_skips_embedding_model_for_default_selection(self):
        with (
            patch.object(
                admin,
                "_ollama_get",
                new=AsyncMock(return_value={"models": [{"name": "nomic-embed-text:latest"}, {"name": "llama3.2:3b"}]}),
            ),
            patch.object(
                admin,
                "_ollama_request",
                new=AsyncMock(return_value={"eval_count": 20, "eval_duration": 2_000_000_000, "total_duration": 2_100_000_000}),
            ),
        ):
            resp = self.client.post("/api/benchmarks/run", json={"iterations": 1})

        self.assertEqual(resp.status_code, 200)
        payload = resp.json()
        self.assertEqual(payload["model"], "llama3.2:3b")
        self.assertEqual(payload["summary"]["successful_runs"], 1)

    def test_benchmark_rejects_explicit_embedding_model(self):
        with patch.object(admin, "_ollama_get", new=AsyncMock(return_value={"models": [{"name": "llama3.2:3b"}]})):
            resp = self.client.post("/api/benchmarks/run", json={"model": "nomic-embed-text:latest", "iterations": 1})

        self.assertEqual(resp.status_code, 400)
        self.assertIn("embedding-only", resp.json().get("detail", ""))

    def test_benchmark_returns_aggregated_metrics(self):
        with (
            patch.object(
                admin,
                "_ollama_get",
                new=AsyncMock(return_value={"models": [{"name": "llama3.2:3b"}]}),
            ),
            patch.object(
                admin,
                "_ollama_request",
                new=AsyncMock(
                    side_effect=[
                        {"eval_count": 24, "eval_duration": 2_000_000_000, "total_duration": 2_200_000_000},
                        {"eval_count": 30, "eval_duration": 2_000_000_000, "total_duration": 2_100_000_000},
                        {"eval_count": 36, "eval_duration": 2_000_000_000, "total_duration": 2_300_000_000},
                    ]
                ),
            ),
        ):
            resp = self.client.post("/api/benchmarks/run", json={"iterations": 3})

        self.assertEqual(resp.status_code, 200)
        payload = resp.json()
        self.assertEqual(payload["model"], "llama3.2:3b")
        self.assertEqual(payload["summary"]["iterations"], 3)
        self.assertEqual(payload["summary"]["successful_runs"], 3)
        self.assertEqual(payload["summary"]["failed_runs"], 0)
        self.assertGreater(payload["summary"]["tokens_per_second_avg"], 0)
        self.assertEqual(len(payload["samples"]), 3)

    def test_vision_analyze_requires_feature_flag(self):
        with patch.object(admin, "LOCALAI_VISION_ENABLED", False):
            resp = self.client.post(
                "/api/vision/analyze",
                json={"model": "llava:latest", "prompt": "describe", "image_url": "https://example.com/a.png"},
            )

        self.assertEqual(resp.status_code, 503)
        self.assertIn("vision lane disabled", resp.json().get("detail", ""))

    def test_vision_analyze_returns_ready_payload_when_enabled(self):
        with (
            patch.object(admin, "LOCALAI_VISION_ENABLED", True),
            patch.object(
                admin,
                "_vision_infer",
                new=AsyncMock(
                    return_value={
                        "response": "scene description",
                        "raw": {"response": "scene description"},
                        "eval_count": 20,
                        "eval_duration_ms": 120.0,
                        "total_duration_ms": 150.0,
                        "tokens_per_second": 166.0,
                    }
                ),
            ),
        ):
            resp = self.client.post(
                "/api/vision/analyze",
                json={"model": "llava:latest", "prompt": "describe", "image_base64": "aGVsbG8="},
            )

        self.assertEqual(resp.status_code, 200)
        payload = resp.json()
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["model"], "llava:latest")
        self.assertEqual(payload["image_source"], "base64")
        self.assertEqual(payload["response"], "scene description")

    def test_vision_analyze_rejects_non_vision_model(self):
        with patch.object(admin, "LOCALAI_VISION_ENABLED", True):
            resp = self.client.post(
                "/api/vision/analyze",
                json={"model": "llama3.2:3b", "prompt": "describe", "image_url": "https://example.com/a.png"},
            )

        self.assertEqual(resp.status_code, 400)
        self.assertIn("does not appear vision-capable", resp.json().get("detail", ""))

    def test_vision_smoke_runs_inference_check(self):
        with (
            patch.object(admin, "LOCALAI_VISION_ENABLED", True),
            patch.object(
                admin,
                "_vision_infer",
                new=AsyncMock(
                    return_value={
                        "response": "ok",
                        "raw": {"response": "ok"},
                        "eval_count": 10,
                        "eval_duration_ms": 50.0,
                        "total_duration_ms": 70.0,
                        "tokens_per_second": 200.0,
                    }
                ),
            ),
        ):
            resp = self.client.post(
                "/api/tests/vision-smoke",
                json={"model": "llava:latest", "prompt": "ok", "image_base64": "aGVsbG8="},
            )

        self.assertEqual(resp.status_code, 200)
        payload = resp.json()
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["summary"]["skipped"], 0)
        self.assertTrue(payload["checks"][0]["ok"])

    def test_vision_benchmark_runs_dataset(self):
        dataset = (
            '{"prompt":"Return exactly: ok","image_base64":"aGVsbG8=","expected_contains":"ok"}\n'
            '{"prompt":"Return exactly: ok","image_base64":"aGVsbG8=","expected_contains":"ok"}\n'
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "vision.jsonl"
            path.write_text(dataset, encoding="utf-8")
            with (
                patch.object(admin, "LOCALAI_VISION_ENABLED", True),
                patch.object(
                    admin,
                    "_vision_infer",
                    new=AsyncMock(
                        side_effect=[
                            {
                                "response": "ok",
                                "raw": {"response": "ok"},
                                "eval_count": 10,
                                "eval_duration_ms": 50.0,
                                "total_duration_ms": 70.0,
                                "tokens_per_second": 200.0,
                            },
                            {
                                "response": "ok",
                                "raw": {"response": "ok"},
                                "eval_count": 12,
                                "eval_duration_ms": 60.0,
                                "total_duration_ms": 80.0,
                                "tokens_per_second": 180.0,
                            },
                        ]
                    ),
                ),
            ):
                resp = self.client.post(
                    "/api/tests/vision-benchmark",
                    json={"model": "llava:latest", "dataset_path": str(path), "iterations": 2, "timeout_seconds": 10.0},
                )

        self.assertEqual(resp.status_code, 200)
        payload = resp.json()
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["summary"]["iterations_executed"], 2)
        self.assertEqual(payload["summary"]["failed_runs"], 0)
        self.assertEqual(payload["summary"]["score_passed"], 2)
        self.assertEqual(len(payload["samples"]), 2)

    def test_image_gen_health_reports_stub_status(self):
        with (
            patch.object(admin, "LOCALAI_IMAGE_GEN_ENABLED", True),
            patch.object(admin, "LOCALAI_IMAGE_GEN_PROVIDER", "comfyui"),
            patch.object(admin, "LOCALAI_IMAGE_GEN_BACKEND_URL", "http://image-gen:8090"),
        ):
            resp = self.client.get("/api/image-gen/health")

        self.assertEqual(resp.status_code, 200)
        payload = resp.json()
        self.assertIn(payload["status"], {"ready", "down"})
        self.assertTrue(payload["enabled"])
        self.assertEqual(payload["provider"], "comfyui")
        self.assertEqual(payload["backend_url"], "http://image-gen:8090")

    def test_image_gen_benchmark_returns_summary(self):
        with (
            patch.object(admin, "LOCALAI_IMAGE_GEN_ENABLED", True),
            patch.object(
                admin,
                "_image_gen_generate",
                new=AsyncMock(side_effect=[{"data": [{"url": "http://x/a.png"}]}, {"data": [{"url": "http://x/b.png"}]}]),
            ),
        ):
            resp = self.client.post("/api/benchmarks/image-gen", json={"iterations": 2, "size": "1024x1024"})

        self.assertEqual(resp.status_code, 200)
        payload = resp.json()
        self.assertEqual(payload["summary"]["iterations"], 2)
        self.assertEqual(payload["summary"]["failed_runs"], 0)
        self.assertEqual(len(payload["samples"]), 2)


if __name__ == "__main__":
    unittest.main()
