"""
Model backend abstraction defining the interface for inference runtime implementations.
"""

import abc
import logging
import time
from dataclasses import dataclass
from typing import Any, Generator, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class GenerationOutput:
    """Result of a single generation call."""
    text: str
    num_input_tokens: int
    num_output_tokens: int
    finish_reason: str  # "eos", "length", "error"
    total_time_s: float = 0.0


class ModelBackend(abc.ABC):
    """Abstract interface decoupling engine serving logic from backend model execution."""

    @abc.abstractmethod
    def load(self) -> None:
        """Load the model and tokenizer. Called once at startup."""
        ...

    @abc.abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
    ) -> GenerationOutput:
        """
        Generate text for a single prompt.

        Returns the complete generated text and metadata.
        """
        ...

    @abc.abstractmethod
    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
    ) -> Generator[str, None, GenerationOutput]:
        """
        Generate text token-by-token, yielding each decoded chunk.

        The generator yields string chunks as they become available.
        The final return value (accessed via StopIteration.value) is
        a GenerationOutput with complete metadata.
        """
        ...

    @abc.abstractmethod
    def unload(self) -> None:
        """Release model resources (GPU memory, etc.)."""
        ...

    @abc.abstractmethod
    def is_loaded(self) -> bool:
        """Check if the model is currently loaded."""
        ...

    @abc.abstractmethod
    def get_info(self) -> dict[str, Any]:
        """Return model info: name, device, dtype, memory usage, etc."""
        ...
