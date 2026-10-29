"""
Integration tests for the FastAPI serving endpoints.

Tests:
- GET /health
- GET /config
- GET /metrics
- POST /generate (validation, response structure)
- POST /generate/stream (SSE streaming)
- Error handling on bad inputs
"""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from app.main import app
from app.engine.model_backend import GenerationOutput
from app.schemas import GenerateRequest


@pytest.fixture
def client():
    """Create a FastAPI test client with mocked engine."""
    with patch("app.main._engine") as mock_engine, patch("app.main._config") as mock_config:
        mock_config.server.max_input_tokens = 1024
        mock_config.logging.level = "INFO"
        mock_config.logging.format = "%(message)s"

        # Mock engine methods
        mock_engine.get_health.return_value = {
            "status": "ok",
            "model_loaded": True,
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "device": "cuda",
            "precision": "bfloat16",
            "gpu_memory_allocated_mb": 1200.0,
            "gpu_memory_reserved_mb": 1500.0,
        }

        mock_engine.get_config_info.return_value = {
            "model": {"name": "Qwen/Qwen2.5-0.5B-Instruct"},
            "server": {"port": 8000},
        }

        mock_engine.get_metrics.return_value = {
            "total_requests": 5,
            "active_requests": 0,
            "completed_requests": 5,
            "failed_requests": 0,
            "total_input_tokens": 100,
            "total_output_tokens": 250,
            "avg_tokens_per_second": 45.2,
            "avg_latency_ms": 110.5,
        }

        mock_engine.generate.return_value = (
            "test-req-id",
            GenerationOutput(
                text="The KV cache stores precomputed keys and values.",
                num_input_tokens=15,
                num_output_tokens=10,
                finish_reason="eos",
                total_time_s=0.25,
            ),
        )

        async def mock_stream_gen(*args, **kwargs):
            yield "test-stream-id", "Hello", None
            yield "test-stream-id", " world", None
            yield "test-stream-id", "!", None
            yield "test-stream-id", "", GenerationOutput(
                text="Hello world!",
                num_input_tokens=5,
                num_output_tokens=3,
                finish_reason="eos",
                total_time_s=0.1,
            )

        mock_engine.generate_stream = mock_stream_gen

        with TestClient(app) as test_client:
            yield test_client


def test_chat_ui_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "LLM Inference Engine" in response.text


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["model_name"] == "Qwen/Qwen2.5-0.5B-Instruct"


def test_config_endpoint(client):
    response = client.get("/config")
    assert response.status_code == 200
    data = response.json()
    assert "model" in data


def test_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["total_requests"] == 5
    assert data["avg_tokens_per_second"] == 45.2


def test_generate_endpoint_success(client):
    payload = {
        "prompt": "Explain continuous batching",
        "max_new_tokens": 64,
        "temperature": 0.7,
    }
    response = client.post("/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["request_id"] == "test-req-id"
    assert "KV cache" in data["generated_text"]
    assert data["num_input_tokens"] == 15
    assert data["num_output_tokens"] == 10
    assert data["finish_reason"] == "eos"
    assert data["tokens_per_second"] > 0


def test_generate_endpoint_validation_empty_prompt(client):
    response = client.post("/generate", json={"prompt": ""})
    assert response.status_code == 422  # Pydantic validation error


def test_generate_stream_endpoint(client):
    payload = {
        "prompt": "Stream this test",
        "max_new_tokens": 10,
    }
    response = client.post("/generate/stream", json=payload)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    # Verify SSE format contains data chunks
    content = response.text
    assert "data: " in content
    assert "Hello" in content
    assert "world" in content
