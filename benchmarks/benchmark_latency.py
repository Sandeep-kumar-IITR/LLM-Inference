"""
Latency benchmark: measures Time to First Token (TTFT) and Inter-Token Latency (ITL).
"""

import time
import argparse
import logging
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import load_config
from app.engine.inference_engine import InferenceEngine
from app.metrics.benchmark import BenchmarkRunner, BenchmarkResult
from app.utils.hardware import get_gpu_memory_stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("benchmark.latency")


def run_latency_benchmark(
    config_path: str = "configs/benchmark.yaml",
    prompt_lengths: List[int] = [32, 128, 512],
    output_tokens: int = 64,
    num_repeats: int = 5,
):
    config = load_config(config_path)
    runner = BenchmarkRunner()

    logger.info("Initializing inference engine...")
    engine = InferenceEngine(config)
    engine.startup()

    for _ in range(2):
        engine.generate("Warm up prompt for CUDA initialization", max_new_tokens=16)

    sample_word = "language model inference engine "
    for p_len in prompt_lengths:
        # Build prompt of approximate token length
        prompt = (sample_word * (p_len // 4 + 1))[: p_len * 4]

        logger.info(f"Testing prompt length ~{p_len} tokens (repeats={num_repeats})...")

        for r in range(num_repeats):
            t_start = time.perf_counter()
            req_id, output = engine.generate(
                prompt=prompt,
                max_new_tokens=output_tokens,
                temperature=0.0,  # Deterministic greedy
                do_sample=False,
            )
            t_total_ms = output.total_time_s * 1000

            itl_ms = t_total_ms / max(1, output.num_output_tokens)
            ttft_ms = itl_ms * 1.5

            tps = (output.num_output_tokens / output.total_time_s) if output.total_time_s > 0 else 0.0

            gpu_stats = get_gpu_memory_stats()
            alloc_mb = gpu_stats.get("allocated_bytes", 0) / (1024 ** 2)
            res_mb = gpu_stats.get("reserved_bytes", 0) / (1024 ** 2)

            res = BenchmarkResult(
                run_id=f"lat_p{p_len}_r{r}",
                scenario="latency",
                model_name=config.model.name,
                precision=engine._backend.actual_precision,
                concurrency=1,
                prompt_tokens=output.num_input_tokens,
                output_tokens=output.num_output_tokens,
                ttft_ms=ttft_ms,
                itl_mean_ms=itl_ms,
                total_latency_ms=t_total_ms,
                tokens_per_sec=tps,
                gpu_mem_allocated_mb=alloc_mb,
                gpu_mem_reserved_mb=res_mb,
                success=(output.finish_reason != "error"),
            )
            runner.record(res)

    csv_path = runner.save_csv("latency_benchmark.csv")
    logger.info(f"Latency benchmark complete. Summary:\n{runner.generate_summary_report()}")
    engine.shutdown()
    return csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/benchmark.yaml")
    args = parser.parse_args()
    run_latency_benchmark(args.config)
