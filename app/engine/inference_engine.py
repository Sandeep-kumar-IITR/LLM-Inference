"""
Inference engine coordinating model backends, KV caching, and metrics collection.
"""

import logging
import time
import uuid
from typing import Optional, AsyncGenerator

from app.config import EngineConfig
from app.engine.model_backend import GenerationOutput
from app.engine.model_loader import TransformersBackend
from app.engine.generation import merge_generation_params
from app.utils.hardware import HardwareInfo, detect_hardware
from app.cache.block_manager import BlockManager
from app.cache.prefix_cache import PrefixCache

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Core inference engine coordinating model execution, KV caching, and observability."""

    def __init__(self, config: EngineConfig):
        self._config = config
        self._hardware = detect_hardware()
        self._backend = TransformersBackend(
            model_config=config.model,
            gen_config=config.generation,
            hardware=self._hardware,
        )
        self._block_manager = BlockManager(
            num_blocks=config.cache.num_blocks,
            block_size=config.cache.block_size,
        )
        self._prefix_cache = (
            PrefixCache(
                block_manager=self._block_manager,
                model_name=config.model.name,
                max_entries=config.cache.max_prefix_entries,
                block_size=config.cache.block_size,
            )
            if config.cache.enable_prefix_cache
            else None
        )
        # Metrics counters
        self._total_requests = 0
        self._completed_requests = 0
        self._failed_requests = 0
        self._active_requests = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_generation_time = 0.0

    def startup(self) -> None:
        """Initialize the engine: load model, set up components."""
        logger.info("=" * 60)
        logger.info("LLM Inference Engine starting up")
        logger.info("=" * 60)
        logger.info("\n" + self._hardware.summary())
        self._backend.load()
        logger.info("Engine ready to serve requests")

    def shutdown(self) -> None:
        """Gracefully shut down the engine."""
        logger.info("Shutting down inference engine...")
        self._backend.unload()
        logger.info("Engine shut down complete")

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        do_sample: Optional[bool] = None,
    ) -> tuple[str, GenerationOutput]:
        """
        Run generation and return (request_id, output).
        """
        request_id = str(uuid.uuid4())[:8]
        self._total_requests += 1
        self._active_requests += 1

        prompt_tokens = []
        if self._backend.tokenizer:
            try:
                formatted = self._backend._apply_chat_template(prompt)
                prompt_tokens = self._backend.tokenizer.encode(formatted, add_special_tokens=False)
            except Exception:
                prompt_tokens = []

        allocated_table = None
        if self._block_manager and prompt_tokens:
            try:
                shared_blocks = []
                if self._prefix_cache:
                    shared_blocks, _ = self._prefix_cache.match_prefix(prompt_tokens)
                allocated_table = self._block_manager.allocate_for_request(
                    request_id, len(prompt_tokens), shared_block_ids=shared_blocks
                )
            except Exception as e:
                logger.debug(f"KV cache allocation notice for {request_id}: {e}")

        try:
            params = merge_generation_params(
                self._config.generation,
                max_new_tokens, temperature, top_p, top_k,
                repetition_penalty, do_sample,
            )
            output = self._backend.generate(prompt, **params)

            self._total_input_tokens += output.num_input_tokens
            self._total_output_tokens += output.num_output_tokens
            self._total_generation_time += output.total_time_s

            if output.finish_reason == "error":
                self._failed_requests += 1
            else:
                self._completed_requests += 1
                if self._prefix_cache and allocated_table is not None and prompt_tokens:
                    try:
                        self._prefix_cache.insert_prefix(prompt_tokens, allocated_table.get_block_ids())
                    except Exception as e:
                        logger.debug(f"Prefix cache insert notice: {e}")

            return request_id, output

        except Exception as e:
            self._failed_requests += 1
            logger.error(f"Generation failed for {request_id}: {e}")
            return request_id, GenerationOutput(
                text="",
                num_input_tokens=0,
                num_output_tokens=0,
                finish_reason="error",
                total_time_s=0.0,
            )
        finally:
            if self._block_manager and allocated_table is not None:
                try:
                    self._block_manager.free_request(request_id)
                except Exception:
                    pass
            self._active_requests -= 1

    async def generate_stream(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        do_sample: Optional[bool] = None,
    ) -> AsyncGenerator[tuple[str, str, Optional[GenerationOutput]], None]:
        """
        Async streaming generation.

        Yields (request_id, token_chunk, None) for each token.
        The final yield has (request_id, "", GenerationOutput).
        """
        import asyncio

        request_id = str(uuid.uuid4())[:8]
        self._total_requests += 1
        self._active_requests += 1

        prompt_tokens = []
        if self._backend.tokenizer:
            try:
                formatted = self._backend._apply_chat_template(prompt)
                prompt_tokens = self._backend.tokenizer.encode(formatted, add_special_tokens=False)
            except Exception:
                prompt_tokens = []

        num_input_tokens = len(prompt_tokens) if prompt_tokens else len(prompt.split())
        self._total_input_tokens += num_input_tokens

        allocated_table = None
        if self._block_manager and prompt_tokens:
            try:
                shared_blocks = []
                if self._prefix_cache:
                    shared_blocks, _ = self._prefix_cache.match_prefix(prompt_tokens)
                allocated_table = self._block_manager.allocate_for_request(
                    request_id, len(prompt_tokens), shared_block_ids=shared_blocks
                )
            except Exception as e:
                logger.debug(f"KV cache allocation notice for {request_id}: {e}")

        try:
            params = merge_generation_params(
                self._config.generation,
                max_new_tokens, temperature, top_p, top_k,
                repetition_penalty, do_sample,
            )

            start = time.perf_counter()
            gen = self._backend.generate_stream(prompt, **params)

            full_text = ""
            num_output_tokens = 0

            try:
                while True:
                    chunk = await asyncio.to_thread(next, gen, None)
                    if chunk is None:
                        break
                    full_text += chunk
                    if self._block_manager and allocated_table is not None:
                        try:
                            self._block_manager.append_slot_for_token(request_id)
                        except Exception:
                            pass
                    yield request_id, chunk, None
            except StopIteration:
                pass

            elapsed = time.perf_counter() - start
            if self._backend.tokenizer:
                num_output_tokens = len(
                    self._backend.tokenizer.encode(full_text, add_special_tokens=False)
                )

            effective_max_tokens = max_new_tokens or self._config.generation.max_new_tokens
            finish_reason = "length" if num_output_tokens >= effective_max_tokens else "eos"

            output = GenerationOutput(
                text=full_text,
                num_input_tokens=num_input_tokens,
                num_output_tokens=num_output_tokens,
                finish_reason=finish_reason,
                total_time_s=elapsed,
            )

            self._total_output_tokens += num_output_tokens
            self._total_generation_time += elapsed
            self._completed_requests += 1

            if self._prefix_cache and allocated_table is not None and prompt_tokens:
                try:
                    self._prefix_cache.insert_prefix(prompt_tokens, allocated_table.get_block_ids())
                except Exception as e:
                    logger.debug(f"Prefix cache insert notice: {e}")

            yield request_id, "", output

        except Exception as e:
            self._failed_requests += 1
            logger.error(f"Streaming generation failed for {request_id}: {e}")
            output = GenerationOutput(
                text="", num_input_tokens=0, num_output_tokens=0,
                finish_reason="error", total_time_s=0.0,
            )
            yield request_id, "", output
        finally:
            if self._block_manager and allocated_table is not None:
                try:
                    self._block_manager.free_request(request_id)
                except Exception:
                    pass
            self._active_requests -= 1

    def get_metrics(self) -> dict:
        """Get current engine metrics."""
        avg_tps = 0.0
        avg_latency = 0.0
        if self._completed_requests > 0:
            avg_tps = self._total_output_tokens / max(self._total_generation_time, 0.001)
            avg_latency = (self._total_generation_time / self._completed_requests) * 1000

        cache_hits = self._prefix_cache.hits if self._prefix_cache else 0
        cache_misses = self._prefix_cache.misses if self._prefix_cache else 0
        kv_allocated = self._block_manager.num_allocated_blocks if self._block_manager else 0
        kv_free = self._block_manager.num_free_blocks if self._block_manager else 0

        return {
            "total_requests": self._total_requests,
            "active_requests": self._active_requests,
            "completed_requests": self._completed_requests,
            "failed_requests": self._failed_requests,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "avg_tokens_per_second": round(avg_tps, 1),
            "avg_latency_ms": round(avg_latency, 1),
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "kv_blocks_allocated": kv_allocated,
            "kv_blocks_free": kv_free,
        }

    def get_health(self) -> dict:
        """Get engine health status."""
        info = self._backend.get_info()
        return {
            "status": "ok" if self._backend.is_loaded() else "not_ready",
            "model_loaded": self._backend.is_loaded(),
            "model_name": info.get("model_name", ""),
            "device": info.get("device", ""),
            "precision": info.get("precision", ""),
            "gpu_memory_allocated_mb": info.get("gpu_memory_allocated_mb"),
            "gpu_memory_reserved_mb": info.get("gpu_memory_reserved_mb"),
        }

    def get_config_info(self) -> dict:
        """Return the active configuration (sanitized)."""
        return {
            "model": self._config.model.model_dump(),
            "generation": self._config.generation.model_dump(),
            "server": self._config.server.model_dump(),
            "cache": self._config.cache.model_dump(),
            "scheduler": self._config.scheduler.model_dump(),
            "hardware": {
                "device": self._hardware.best_device(),
                "gpu": self._hardware.primary_gpu.name if self._hardware.primary_gpu else None,
                "gpu_memory_gb": self._hardware.total_gpu_memory_gb,
                "cuda_version": self._hardware.cuda_version,
                "native_extensions": self._hardware.native_extension_available,
            },
        }
