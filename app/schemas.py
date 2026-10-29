"""
API request and response schemas.

Defines the Pydantic models for all HTTP API interactions.
"""

from typing import Optional
from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """Request body for text generation endpoints."""
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=32768,
        description="Input prompt text.",
    )
    max_new_tokens: Optional[int] = Field(
        None,
        ge=1,
        le=4096,
        description="Maximum number of tokens to generate. Uses server default if not set.",
    )
    temperature: Optional[float] = Field(
        None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. 0.0 = greedy decoding.",
    )
    top_p: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Top-p (nucleus) sampling threshold.",
    )
    top_k: Optional[int] = Field(
        None,
        ge=0,
        description="Top-k sampling. 0 = disabled.",
    )
    repetition_penalty: Optional[float] = Field(
        None,
        ge=1.0,
        le=3.0,
        description="Repetition penalty factor.",
    )
    stream: bool = Field(
        False,
        description="Enable SSE token streaming.",
    )

    model_config = {"json_schema_extra": {
        "examples": [
            {
                "prompt": "Explain what a KV cache is in one paragraph.",
                "max_new_tokens": 128,
                "temperature": 0.7,
            }
        ]
    }}


class GenerateResponse(BaseModel):
    """Response body for the standard /generate endpoint."""
    request_id: str = Field(..., description="Unique request identifier.")
    prompt: str = Field(..., description="Original input prompt.")
    generated_text: str = Field(..., description="Generated output text.")
    num_input_tokens: int = Field(..., description="Number of input tokens.")
    num_output_tokens: int = Field(..., description="Number of generated tokens.")
    total_time_ms: float = Field(..., description="Total generation time in milliseconds.")
    tokens_per_second: float = Field(..., description="Output tokens per second.")
    finish_reason: str = Field(
        ...,
        description="Why generation stopped: 'length', 'eos', 'error'.",
    )


class StreamChunk(BaseModel):
    """A single chunk in a streaming response."""
    request_id: str
    token: str = ""
    finish_reason: Optional[str] = None
    # Timing info on final chunk
    num_output_tokens: Optional[int] = None
    total_time_ms: Optional[float] = None


class HealthResponse(BaseModel):
    """Response for the /health endpoint."""
    status: str = "ok"
    model_loaded: bool = False
    model_name: str = ""
    device: str = ""
    precision: str = ""
    gpu_memory_allocated_mb: Optional[float] = None
    gpu_memory_reserved_mb: Optional[float] = None


class MetricsResponse(BaseModel):
    """Response for the /metrics endpoint."""
    total_requests: int = 0
    active_requests: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    avg_tokens_per_second: float = 0.0
    avg_latency_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    kv_blocks_allocated: int = 0
    kv_blocks_free: int = 0


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: str = ""
    request_id: Optional[str] = None
