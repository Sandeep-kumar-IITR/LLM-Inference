"""Tests for model backend loading, text generation, and streaming."""

import pytest
import time
from unittest.mock import MagicMock, patch

from app.config import ModelConfig, GenerationConfig, EngineConfig
from app.engine.model_backend import GenerationOutput
from app.engine.generation import merge_generation_params


class TestParameterMerging:
    """Test that request parameters correctly override config defaults."""

    def test_all_defaults(self):
        """When no request params given, use all config defaults."""
        defaults = GenerationConfig(
            max_new_tokens=100,
            temperature=0.5,
            top_p=0.8,
            top_k=40,
            repetition_penalty=1.2,
            do_sample=True,
        )
        params = merge_generation_params(defaults)
        assert params["max_new_tokens"] == 100
        assert params["temperature"] == 0.5
        assert params["top_p"] == 0.8
        assert params["top_k"] == 40
        assert params["repetition_penalty"] == 1.2
        assert params["do_sample"] is True

    def test_partial_override(self):
        """Request params override only the specified fields."""
        defaults = GenerationConfig()
        params = merge_generation_params(
            defaults,
            max_new_tokens=50,
            temperature=0.0,
        )
        assert params["max_new_tokens"] == 50
        assert params["temperature"] == 0.0
        # These should still be defaults
        assert params["top_p"] == defaults.top_p
        assert params["top_k"] == defaults.top_k

    def test_full_override(self):
        """All params can be overridden."""
        defaults = GenerationConfig()
        params = merge_generation_params(
            defaults,
            max_new_tokens=10,
            temperature=1.5,
            top_p=0.5,
            top_k=10,
            repetition_penalty=2.0,
            do_sample=False,
        )
        assert params["max_new_tokens"] == 10
        assert params["temperature"] == 1.5
        assert params["top_p"] == 0.5
        assert params["top_k"] == 10
        assert params["repetition_penalty"] == 2.0
        assert params["do_sample"] is False


class TestGenerationOutput:
    """Test GenerationOutput data class."""

    def test_basic_output(self):
        output = GenerationOutput(
            text="Hello world",
            num_input_tokens=5,
            num_output_tokens=2,
            finish_reason="eos",
            total_time_s=0.5,
        )
        assert output.text == "Hello world"
        assert output.num_input_tokens == 5
        assert output.num_output_tokens == 2
        assert output.finish_reason == "eos"
        assert output.total_time_s == 0.5

    def test_error_output(self):
        output = GenerationOutput(
            text="",
            num_input_tokens=0,
            num_output_tokens=0,
            finish_reason="error",
        )
        assert output.finish_reason == "error"
        assert output.text == ""


class TestModelConfig:
    """Test configuration validation."""

    def test_default_config(self):
        config = ModelConfig()
        assert config.name == "Qwen/Qwen2.5-0.5B-Instruct"
        assert config.precision == "auto"
        assert config.device == "auto"

    def test_custom_config(self):
        config = ModelConfig(
            name="meta-llama/Llama-3.2-1B",
            precision="float16",
            device="cuda",
        )
        assert config.name == "meta-llama/Llama-3.2-1B"
        assert config.precision == "float16"

    def test_engine_config_defaults(self):
        config = EngineConfig()
        assert config.model.name == "Qwen/Qwen2.5-0.5B-Instruct"
        assert config.generation.max_new_tokens == 256
        assert config.server.port == 8000
        assert config.cache.block_size == 16
        assert config.scheduler.max_batch_size == 8
