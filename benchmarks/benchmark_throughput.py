"""
Throughput benchmark: measures aggregate token throughput across concurrency levels.
"""

import time
import argparse
import logging
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import load_config
from app.engine.inference_engine import InferenceEngine
from app.metrics.benchmark import BenchmarkRunner, BenchmarkResult
from app.utils.hardware import get_gpu_memory_stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("benchmark.throughput")


def run_throughput_benchmark(
    config_path: str = "configs/benchmark.yaml",
    total_requests: int = 16,
    concurrency_levels: List[int] = [1, 2, 4, 8],
    output_tokens: int = 64,
):
    config = load_config(config_path)
    runner = BenchmarkRunner()

    logger.info("Initializing inference engine...")
    engine = InferenceEngine(config)
    engine.startup()

    prompt = "Explain why high-performance LLM engines use PagedAttention and continuous batching."

    # Warm-up
    engine.generate(prompt, max_new_tokens=16)

    for c in concurrency_levels:
        logger.info(f"Measuring throughput with concurrency = {c} ({total_requests} total requests)...")

        t_wall_start = time.perf_counter()

        def send_request(idx: int):
            t0 = time.perf_counter()
            rid, out = engine.generate(
                prompt=prompt,
                max_new_tokens=output_tokens,
                temperature=0.0,
                do_sample=False,
            )
            t_total = time.perf_counter() - t0
            return idx, out, t_total

        with ThreadPoolExecutor(max_workers=c) as pool:
            results = list(pool.map(send_request, range(total_requests)))

        t_wall_end = time.perf_counter()
        wall_time_s = t_wall_end - t_wall_start

        total_output_tokens = sum(out.num_output_tokens for _, out, _ in results)
        aggregate_tps = total_output_tokens / wall_time_s if wall_time_s > 0 else 0.0

        logger.info(
            f"Concurrency {c}: {total_output_tokens} tokens in {wall_time_s:.2f}s "
            f"-> Aggregate TPS: {aggregate_tps:.2f} tokens/s"
        )

        gpu_stats = get_gpu_memory_stats()
        alloc_mb = gpu_stats.get("allocated_bytes", 0) / (1024 ** 2)
        res_mb = gpu_stats.get("reserved_bytes", 0) / (1024 ** 2)

        for idx, out, req_time_s in results:
            res = BenchmarkResult(
                run_id=f"tp_c{c}_req{idx}",
                scenario="throughput",
                model_name=config.model.name,
                precision=engine._backend.actual_precision,
                concurrency=c,
                prompt_tokens=out.num_input_tokens,
                output_tokens=out.num_output_tokens,
                ttft_ms=(req_time_s / max(1, out.num_output_tokens)) * 1000,
                itl_mean_ms=(req_time_s / max(1, out.num_output_tokens)) * 1000,
                total_latency_ms=req_time_s * 1000,
                tokens_per_sec=out.num_output_tokens / req_time_s if req_time_s > 0 else 0.0,
                gpu_mem_allocated_mb=alloc_mb,
                gpu_mem_reserved_mb=res_mb,
                success=(out.finish_reason != "error"),
            )
            runner.record(res)

    csv_path = runner.save_csv("throughput_benchmark.csv")
    logger.info(f"Throughput benchmark complete. Summary:\n{runner.generate_summary_report()}")
    engine.shutdown()
    return csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/benchmark.yaml")
    args = parser.parse_args()
    run_throughput_benchmark(args.config)
