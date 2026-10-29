"""Concurrency and throughput benchmark under simulated concurrent client load."""

import time
import argparse
import logging
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import load_config
from app.engine.inference_engine import InferenceEngine
from app.metrics.benchmark import BenchmarkRunner, BenchmarkResult
from app.utils.hardware import get_gpu_memory_stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("benchmark.concurrency")


def run_concurrency_benchmark(
    config_path: str = "configs/benchmark.yaml",
    concurrency_levels: list[int] = [1, 2, 4, 8, 16],
    requests_per_level: int = 16,
    output_tokens: int = 48,
):
    config = load_config(config_path)
    runner = BenchmarkRunner()

    logger.info("Starting Concurrency Benchmark...")
    engine = InferenceEngine(config)
    engine.startup()

    # Pre-warm
    engine.generate("Warm-up prompt for concurrency test", max_new_tokens=8)

    prompts = [
        "What is the role of key-value caching in autoregressive transformer inference?",
        "Compare static batching and continuous batching in serving systems.",
        "How does prefix caching reduce prefill latency for common prompts?",
        "Explain memory fragmentation in deep learning inference servers.",
    ]

    for c in concurrency_levels:
        logger.info(f"--- Testing Concurrency Level: {c} simultaneous workers ---")

        latencies_ms = []
        tokens_list = []
        t_wall_start = time.perf_counter()

        def worker_task(task_id: int):
            p = prompts[task_id % len(prompts)]
            t0 = time.perf_counter()
            rid, out = engine.generate(
                prompt=p,
                max_new_tokens=output_tokens,
                temperature=0.7,
                do_sample=True,
            )
            t1 = time.perf_counter()
            total_time = (t1 - t0) * 1000
            return task_id, out, total_time

        with ThreadPoolExecutor(max_workers=c) as executor:
            futures = [executor.submit(worker_task, i) for i in range(requests_per_level)]
            for fut in as_completed(futures):
                task_id, out, total_ms = fut.result()
                latencies_ms.append(total_ms)
                tokens_list.append(out.num_output_tokens)

                gpu_stats = get_gpu_memory_stats()
                alloc_mb = gpu_stats.get("allocated_bytes", 0) / (1024 ** 2)
                res_mb = gpu_stats.get("reserved_bytes", 0) / (1024 ** 2)

                runner.record(
                    BenchmarkResult(
                        run_id=f"conc_{c}_{task_id}",
                        scenario="concurrency",
                        model_name=config.model.name,
                        precision=engine._backend.actual_precision,
                        concurrency=c,
                        prompt_tokens=out.num_input_tokens,
                        output_tokens=out.num_output_tokens,
                        ttft_ms=total_ms / max(1, out.num_output_tokens),
                        itl_mean_ms=total_ms / max(1, out.num_output_tokens),
                        total_latency_ms=total_ms,
                        tokens_per_sec=(out.num_output_tokens / (total_ms / 1000)) if total_ms > 0 else 0,
                        gpu_mem_allocated_mb=alloc_mb,
                        gpu_mem_reserved_mb=res_mb,
                        success=(out.finish_reason != "error"),
                    )
                )

        t_wall_end = time.perf_counter()
        wall_sec = t_wall_end - t_wall_start
        total_tokens = sum(tokens_list)
        agg_tps = total_tokens / wall_sec if wall_sec > 0 else 0

        p50 = np.median(latencies_ms)
        p95 = np.percentile(latencies_ms, 95)
        p99 = np.percentile(latencies_ms, 99)

        logger.info(
            f"Level {c} done in {wall_sec:.2f}s | Agg TPS: {agg_tps:.1f} tok/s | "
            f"p50: {p50:.1f}ms | p95: {p95:.1f}ms | p99: {p99:.1f}ms"
        )

    csv_path = runner.save_csv("concurrency_benchmark.csv")
    engine.shutdown()
    return csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/benchmark.yaml")
    args = parser.parse_args()
    run_concurrency_benchmark(args.config)
