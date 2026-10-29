"""
FastAPI application serving REST endpoints and Server-Sent Events token streaming.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.config import load_config, EngineConfig
from app.schemas import (
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    MetricsResponse,
    ErrorResponse,
    StreamChunk,
)
from app.engine.inference_engine import InferenceEngine
from app.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

# Global engine instance (initialized during lifespan)
_engine: InferenceEngine | None = None
_config: EngineConfig | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan: startup and shutdown hooks.

    The model is loaded once here at startup. It is NOT reloaded per request.
    """
    global _engine, _config

    _config = load_config()
    setup_logging(level=_config.logging.level, fmt=_config.logging.format)

    if _engine is None:
        _engine = InferenceEngine(_config)
        # Load model in a thread to avoid blocking the event loop during startup
        logger.info("Starting model loading (this may take a while)...")
        await asyncio.to_thread(_engine.startup)
        logger.info("Server ready!")
    else:
        logger.info("Using pre-configured / mock engine instance.")

    yield  # Server runs here

    # Shutdown
    logger.info("Shutting down...")
    if _engine is not None and hasattr(_engine, "shutdown"):
        _engine.shutdown()


app = FastAPI(
    title="LLM Inference Engine",
    description="High-performance LLM inference server with KV-cache management, "
                "continuous batching, and streaming support.",
    version="0.1.0",
    lifespan=lifespan,
)

# Enable CORS for local and web client access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_HTML = STATIC_DIR / "index.html"


def get_engine() -> InferenceEngine:
    """Get the global engine instance, raising 503 if not ready."""
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    return _engine


# ============================================================
# Endpoints
# ============================================================


@app.get("/", response_class=HTMLResponse)
async def chat_ui():
    """Serve the interactive chat web interface."""
    if INDEX_HTML.exists():
        return FileResponse(INDEX_HTML)
    return HTMLResponse("<h2>LLM Inference Engine is online. Chat UI not found.</h2>")


@app.post("/generate", response_model=None)
async def generate(request: GenerateRequest):
    """Execute text generation for an input prompt."""
    if request.stream:
        return await generate_stream(request)

    engine = get_engine()

    if _config and len(request.prompt) > _config.server.max_input_tokens * 4:
        raise HTTPException(
            status_code=400,
            detail=f"Prompt exceeds limit of ~{_config.server.max_input_tokens} tokens.",
        )

    request_id, output = await asyncio.to_thread(
        engine.generate,
        prompt=request.prompt,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        repetition_penalty=request.repetition_penalty,
    )

    if output.finish_reason == "error":
        raise HTTPException(status_code=500, detail="Generation failed")

    tps = (
        output.num_output_tokens / output.total_time_s
        if output.total_time_s > 0 else 0.0
    )

    return GenerateResponse(
        request_id=request_id,
        prompt=request.prompt,
        generated_text=output.text,
        num_input_tokens=output.num_input_tokens,
        num_output_tokens=output.num_output_tokens,
        total_time_ms=output.total_time_s * 1000,
        tokens_per_second=round(tps, 1),
        finish_reason=output.finish_reason,
    )


@app.post("/generate/stream")
async def generate_stream(request: GenerateRequest):
    """Stream generated tokens using Server-Sent Events (SSE)."""
    engine = get_engine()

    if _config and len(request.prompt) > _config.server.max_input_tokens * 4:
        raise HTTPException(
            status_code=400,
            detail=f"Prompt exceeds limit of ~{_config.server.max_input_tokens} tokens.",
        )

    async def event_generator():
        request_id = None
        try:
            async for rid, chunk, output in engine.generate_stream(
                prompt=request.prompt,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                repetition_penalty=request.repetition_penalty,
            ):
                request_id = rid
                if output is not None:
                    final = StreamChunk(
                        request_id=rid,
                        token="",
                        finish_reason=output.finish_reason,
                        num_output_tokens=output.num_output_tokens,
                        total_time_ms=output.total_time_s * 1000,
                    )
                    yield f"data: {final.model_dump_json()}\n\n"
                else:
                    sc = StreamChunk(request_id=rid, token=chunk)
                    yield f"data: {sc.model_dump_json()}\n\n"
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            error_chunk = StreamChunk(
                request_id=request_id or "unknown",
                finish_reason="error",
            )
            yield f"data: {error_chunk.model_dump_json()}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    engine = get_engine()
    data = engine.get_health()
    return HealthResponse(**data)


@app.get("/metrics", response_model=MetricsResponse)
async def metrics():
    """Engine performance metrics."""
    engine = get_engine()
    data = engine.get_metrics()
    return MetricsResponse(**data)


@app.get("/config")
async def config():
    """Return the active engine configuration."""
    engine = get_engine()
    return engine.get_config_info()


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="internal_error",
            detail=str(exc),
        ).model_dump(),
    )
