"""
Generation utilities and helpers.

Provides convenience functions for running generation with proper
parameter merging between request-level and config-level defaults.
"""

import logging
from typing import Optional, Generator

from app.config import GenerationConfig
from app.engine.model_backend import ModelBackend, GenerationOutput

logger = logging.getLogger(__name__)


def merge_generation_params(
    defaults: GenerationConfig,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    repetition_penalty: Optional[float] = None,
    do_sample: Optional[bool] = None,
) -> dict:
    """
    Merge request-level generation parameters with config defaults.

    Request-level parameters take precedence when provided (not None).
    Otherwise, the server's default config values are used.
    """
    return {
        "max_new_tokens": max_new_tokens if max_new_tokens is not None else defaults.max_new_tokens,
        "temperature": temperature if temperature is not None else defaults.temperature,
        "top_p": top_p if top_p is not None else defaults.top_p,
        "top_k": top_k if top_k is not None else defaults.top_k,
        "repetition_penalty": (
            repetition_penalty if repetition_penalty is not None
            else defaults.repetition_penalty
        ),
        "do_sample": do_sample if do_sample is not None else defaults.do_sample,
    }


def run_generation(
    backend: ModelBackend,
    prompt: str,
    defaults: GenerationConfig,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    repetition_penalty: Optional[float] = None,
    do_sample: Optional[bool] = None,
) -> GenerationOutput:
    """
    Run a complete (non-streaming) generation with merged parameters.
    """
    params = merge_generation_params(
        defaults, max_new_tokens, temperature, top_p, top_k,
        repetition_penalty, do_sample,
    )
    return backend.generate(prompt, **params)


def run_generation_stream(
    backend: ModelBackend,
    prompt: str,
    defaults: GenerationConfig,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    repetition_penalty: Optional[float] = None,
    do_sample: Optional[bool] = None,
) -> Generator[str, None, GenerationOutput]:
    """
    Run streaming generation with merged parameters.
    """
    params = merge_generation_params(
        defaults, max_new_tokens, temperature, top_p, top_k,
        repetition_penalty, do_sample,
    )
    return backend.generate_stream(prompt, **params)
