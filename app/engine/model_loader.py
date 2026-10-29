"""
HuggingFace Transformers model backend implementation.
"""

import logging
import time
import threading
from typing import Any, Generator, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    BitsAndBytesConfig,
)

from app.config import ModelConfig, GenerationConfig
from app.engine.model_backend import ModelBackend, GenerationOutput
from app.utils.hardware import HardwareInfo, get_gpu_memory_stats, format_bytes

logger = logging.getLogger(__name__)


class TransformersBackend(ModelBackend):
    """
    Model execution backend powered by HuggingFace Transformers.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        gen_config: GenerationConfig,
        hardware: HardwareInfo,
    ):
        self._model_config = model_config
        self._gen_config = gen_config
        self._hardware = hardware
        self._model = None
        self._tokenizer = None
        self._device: str = "cpu"
        self._dtype: torch.dtype = torch.float32
        self._actual_precision: str = "float32"

    def load(self) -> None:
        """Load model weights and tokenizer based on configured device and precision."""
        start = time.perf_counter()

        if self._model_config.device == "auto":
            self._device = self._hardware.best_device()
        else:
            self._device = self._model_config.device
        logger.info(f"Using device: {self._device}")

        precision = self._model_config.precision
        quantization_config = None

        if precision == "auto":
            if self._device == "cuda":
                self._dtype = self._hardware.best_dtype()
            else:
                self._dtype = torch.float32
            self._actual_precision = str(self._dtype).replace("torch.", "")

        elif precision == "float16":
            self._dtype = torch.float16
            self._actual_precision = "float16"

        elif precision == "bfloat16":
            self._dtype = torch.bfloat16
            self._actual_precision = "bfloat16"

        elif precision == "float32":
            self._dtype = torch.float32
            self._actual_precision = "float32"

        elif precision == "int4":
            if not self._hardware.supports_precision("int4"):
                logger.warning(
                    "INT4 not supported (missing bitsandbytes or CUDA). "
                    "Falling back to auto precision."
                )
                self._dtype = self._hardware.best_dtype()
                self._actual_precision = str(self._dtype).replace("torch.", "") + " (int4 fallback)"
            else:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                self._dtype = torch.float16
                self._actual_precision = "int4_nf4"
                logger.info("INT4 NF4 quantization enabled")

        elif precision == "fp8":
            if not self._hardware.supports_precision("fp8"):
                logger.warning(
                    "FP8 not supported on this hardware/software. "
                    "Falling back to auto precision."
                )
                self._dtype = self._hardware.best_dtype()
                self._actual_precision = str(self._dtype).replace("torch.", "") + " (fp8 fallback)"
            else:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
                self._dtype = torch.float16
                self._actual_precision = "fp8"
                logger.info("FP8 quantization enabled")
        else:
            logger.warning(f"Unknown precision '{precision}', using float32")
            self._dtype = torch.float32
            self._actual_precision = "float32"

        logger.info(f"Actual precision: {self._actual_precision}")

        logger.info(f"Loading tokenizer: {self._model_config.name}")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_config.name,
            trust_remote_code=self._model_config.trust_remote_code,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        logger.info(f"Loading model: {self._model_config.name}")
        load_kwargs: dict[str, Any] = {
            "pretrained_model_name_or_path": self._model_config.name,
            "trust_remote_code": self._model_config.trust_remote_code,
            "dtype": self._dtype,
        }

        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
            load_kwargs["device_map"] = "auto"
        elif self._device == "cuda":
            load_kwargs["device_map"] = "auto"

        self._model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
        self._model.eval()

        elapsed = time.perf_counter() - start
        logger.info(f"Model loaded in {elapsed:.1f}s")

        if self._device == "cuda":
            stats = get_gpu_memory_stats()
            logger.info(
                f"GPU memory: allocated={format_bytes(stats['allocated_bytes'])}, "
                f"reserved={format_bytes(stats['reserved_bytes'])}"
            )

    def _apply_chat_template(self, prompt: str) -> str:
        """Apply instruction template if supported by the tokenizer."""
        if self._tokenizer is None:
            return prompt

        if hasattr(self._tokenizer, "chat_template") and self._tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            try:
                return self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception as e:
                logger.warning(f"Chat template formatting failed: {e}")
                return prompt
        return prompt

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
        """Generate text for a single input sequence."""
        if not self.is_loaded():
            return GenerationOutput(
                text="",
                num_input_tokens=0,
                num_output_tokens=0,
                finish_reason="error",
                total_time_s=0.0,
            )

        start = time.perf_counter()
        formatted_prompt = self._apply_chat_template(prompt)

        inputs = self._tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self._model_config.max_seq_len,
        ).to(self._model.device)

        num_input_tokens = inputs["input_ids"].shape[1]

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample and temperature > 0,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        if gen_kwargs["do_sample"]:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
            if top_k > 0:
                gen_kwargs["top_k"] = top_k
            gen_kwargs["repetition_penalty"] = repetition_penalty

        with torch.inference_mode():
            outputs = self._model.generate(
                **inputs,
                **gen_kwargs,
            )

        new_token_ids = outputs[0][num_input_tokens:]
        generated_text = self._tokenizer.decode(
            new_token_ids,
            skip_special_tokens=True,
        )
        num_output_tokens = len(new_token_ids)

        eos_id = self._tokenizer.eos_token_id
        if num_output_tokens > 0 and new_token_ids[-1].item() == eos_id:
            finish_reason = "eos"
        elif num_output_tokens >= max_new_tokens:
            finish_reason = "length"
        else:
            finish_reason = "eos"

        elapsed = time.perf_counter() - start

        return GenerationOutput(
            text=generated_text,
            num_input_tokens=num_input_tokens,
            num_output_tokens=num_output_tokens,
            finish_reason=finish_reason,
            total_time_s=elapsed,
        )

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
        """Stream generated token chunks via TextIteratorStreamer."""
        if not self.is_loaded():
            return GenerationOutput(
                text="",
                num_input_tokens=0,
                num_output_tokens=0,
                finish_reason="error",
                total_time_s=0.0,
            )

        start = time.perf_counter()
        formatted_prompt = self._apply_chat_template(prompt)

        inputs = self._tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self._model_config.max_seq_len,
        ).to(self._model.device)

        num_input_tokens = inputs["input_ids"].shape[1]

        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample and temperature > 0,
            "pad_token_id": self._tokenizer.pad_token_id,
            "streamer": streamer,
        }
        if gen_kwargs["do_sample"]:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
            if top_k > 0:
                gen_kwargs["top_k"] = top_k
            gen_kwargs["repetition_penalty"] = repetition_penalty

        # Run generation in a background thread
        generation_error = None

        def _generate():
            nonlocal generation_error
            try:
                with torch.inference_mode():
                    self._model.generate(**inputs, **gen_kwargs)
            except Exception as e:
                generation_error = e
                streamer.end()

        thread = threading.Thread(target=_generate, daemon=True)
        thread.start()

        # Yield text chunks as they arrive
        full_text = ""
        num_output_tokens = 0
        for chunk in streamer:
            if chunk:
                full_text += chunk
                # Approximate token count from chunk (re-tokenize for accuracy)
                chunk_tokens = len(self._tokenizer.encode(chunk, add_special_tokens=False))
                num_output_tokens += chunk_tokens
                yield chunk

        thread.join(timeout=5.0)

        elapsed = time.perf_counter() - start

        if generation_error:
            finish_reason = "error"
        elif num_output_tokens >= max_new_tokens:
            finish_reason = "length"
        else:
            finish_reason = "eos"

        return GenerationOutput(
            text=full_text,
            num_input_tokens=num_input_tokens,
            num_output_tokens=num_output_tokens,
            finish_reason=finish_reason,
            total_time_s=elapsed,
        )

    def unload(self) -> None:
        """Release the model and free GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model unloaded")

    def is_loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    def get_info(self) -> dict[str, Any]:
        info = {
            "model_name": self._model_config.name,
            "device": self._device,
            "precision": self._actual_precision,
            "max_seq_len": self._model_config.max_seq_len,
            "loaded": self.is_loaded(),
        }
        if self.is_loaded() and self._device == "cuda":
            stats = get_gpu_memory_stats()
            info["gpu_memory_allocated_mb"] = stats["allocated_bytes"] / (1024 ** 2)
            info["gpu_memory_reserved_mb"] = stats["reserved_bytes"] / (1024 ** 2)
        return info

    @property
    def tokenizer(self):
        """Expose tokenizer for external use (e.g., prefix caching)."""
        return self._tokenizer

    @property
    def device(self) -> str:
        return self._device

    @property
    def actual_precision(self) -> str:
        return self._actual_precision
