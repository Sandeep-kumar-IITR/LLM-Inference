"""
Configuration management for the LLM Inference Engine.

Loads settings from YAML files and environment variables using Pydantic.
"""

import os
import logging
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


class ModelConfig(BaseModel):
    """Model loading configuration."""
    name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    precision: str = "auto"
    device: str = "auto"
    max_seq_len: int = 2048
    trust_remote_code: bool = True


class GenerationConfig(BaseModel):
    """Default generation parameters."""
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True


class ServerConfig(BaseModel):
    """HTTP server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    max_concurrent_requests: int = 16
    max_input_tokens: int = 1024
    request_timeout: int = 120


class CacheConfig(BaseModel):
    """KV cache and prefix cache configuration."""
    block_size: int = 16
    num_blocks: int = 256
    enable_prefix_cache: bool = True
    max_prefix_entries: int = 128


class SchedulerConfig(BaseModel):
    """Continuous batching scheduler configuration."""
    max_batch_size: int = 8
    policy: str = "fcfs"
    scheduling_interval: float = 0.01


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


class EngineConfig(BaseModel):
    """
    Top-level configuration for the entire inference engine.

    Aggregates model, generation, server, cache, scheduler, and logging configs.
    """
    model: ModelConfig = Field(default_factory=ModelConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def load_config(config_path: Optional[str] = None) -> EngineConfig:
    """
    Load configuration from a YAML file.

    Resolution order:
    1. Explicit config_path argument.
    2. LLM_ENGINE_CONFIG environment variable.
    3. configs/default.yaml in the project root.

    Args:
        config_path: Optional path to a YAML config file.

    Returns:
        Fully resolved EngineConfig instance.
    """
    path = config_path or os.environ.get("LLM_ENGINE_CONFIG")

    if path is None:
        path = str(DEFAULT_CONFIG_PATH)

    config_file = Path(path)
    if not config_file.is_absolute() and not config_file.exists():
        candidate = PROJECT_ROOT / path
        if candidate.exists():
            config_file = candidate

    if config_file.exists():
        logger.info(f"Loading config from {config_file}")
        with open(config_file, "r") as f:
            raw = yaml.safe_load(f) or {}
        config = EngineConfig(**raw)
    else:
        logger.warning(
            f"Config file not found at {config_file}, using defaults"
        )
        config = EngineConfig()

    logger.info(f"Model: {config.model.name}")
    logger.info(f"Precision: {config.model.precision}")
    logger.info(f"Device: {config.model.device}")
    return config
