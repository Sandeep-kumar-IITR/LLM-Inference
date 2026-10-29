"""Quantization benchmark evaluating latency, memory footprint, and output consistency across precision modes."""

import time
import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import load_config
from app.engine.inference_engine import InferenceEngine
from app.metrics.benchmark import BenchmarkRunner, BenchmarkResult
from app.utils.hardware import detect_hardware, get_gpu_memory_stats, format_bytes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("benchmark.quantization")


def run_quantization_benchmark(config_path: str = "configs/benchmark.yaml"):
    base_config = load_config(config_path)
    runner = BenchmarkRunner()
    hw = detect_hardware()

    logger.info("=" * 60)
    logger.info("QUANTIZATION BENCHMARK")
    logger.info("=" * 60)
    logger.info(f"Hardware: {hw.primary_gpu.name if hw.primary_gpu else 'CPU'}")
    logger.info(f"Supports BF16: {hw.primary_gpu.supports_bf16 if hw.primary_gpu else False}")
    logger.info(f"Supports FP8: {hw.primary_gpu.supports_fp8 if hw.primary_gpu else False}")

    modes = ["float16", "int4", "fp8"]
    test_prompts = [
        "Explain quantum computing in three sentences.",
        "Write a Python function to compute the Fibonacci sequence recursively.",
    ]

    quality_comparison: Dict[str, List[Dict[str, str]]] = {}

    for mode in modes:
        logger.info(f"\n--- Testing Precision Mode: {mode} ---")
        cfg = load_config(config_path)
        cfg.model.precision = mode

        try:
            engine = InferenceEngine(cfg)
            engine.startup()
        except Exception as e:
            logger.error(f"Failed to start engine with precision '{mode}': {e}")
            continue

        actual_precision = engine._backend.actual_precision
        logger.info(f"Engine reported actual precision: {actual_precision}")

        quality_comparison[actual_precision] = []

        # Run test prompts
        for idx, prompt in enumerate(test_prompts):
            t0 = time.perf_counter()
            rid, out = engine.generate(
                prompt=prompt,
                max_new_tokens=64,
                temperature=0.0,  # Greedy decoding for exact output comparison
                do_sample=False,
            )
            t1 = time.perf_counter()
            total_sec = t1 - t0

            gpu_stats = get_gpu_memory_stats()
            alloc_mb = gpu_stats.get("allocated_bytes", 0) / (1024 ** 2)
            res_mb = gpu_stats.get("reserved_bytes", 0) / (1024 ** 2)

            tps = out.num_output_tokens / total_sec if total_sec > 0 else 0

            runner.record(
                BenchmarkResult(
                    run_id=f"quant_{mode}_{idx}",
                    scenario=f"quantization_{mode}",
                    model_name=cfg.model.name,
                    precision=actual_precision,
                    concurrency=1,
                    prompt_tokens=out.num_input_tokens,
                    output_tokens=out.num_output_tokens,
                    ttft_ms=(total_sec / max(1, out.num_output_tokens)) * 1000,
                    itl_mean_ms=(total_sec / max(1, out.num_output_tokens)) * 1000,
                    total_latency_ms=total_sec * 1000,
                    tokens_per_sec=tps,
                    gpu_mem_allocated_mb=alloc_mb,
                    gpu_mem_reserved_mb=res_mb,
                    success=(out.finish_reason != "error"),
                )
            )

            quality_comparison[actual_precision].append({
                "prompt": prompt,
                "output": out.text.strip()[:200] + "...",
            })

            logger.info(
                f"[{actual_precision}] Prompt {idx+1}: {out.num_output_tokens} tok in {total_sec:.2f}s "
                f"({tps:.1f} tok/s) | GPU Alloc: {alloc_mb:.1f}MB"
            )

        engine.shutdown()

    # Save results
    csv_path = runner.save_csv("quantization_benchmark.csv")
    logger.info("\n" + "=" * 60)
    logger.info("QUANTIZATION BENCHMARK SUMMARY")
    logger.info("=" * 60)
    logger.info(runner.generate_summary_report())

    logger.info("\n--- Output Quality Comparison ---")
    for prec, outputs in quality_comparison.items():
        logger.info(f"\nPrecision: {prec}")
        for item in outputs:
            logger.info(f"  P: {item['prompt']}")
            logger.info(f"  O: {item['output']}")

    return csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/benchmark.yaml")
    args = parser.parse_args()
    run_quantization_benchmark(args.config)
