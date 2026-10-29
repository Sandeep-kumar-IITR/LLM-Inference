"""
Benchmark execution, telemetry recording, and CSV report export.
"""

import os
import time
import json
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    run_id: str
    scenario: str
    model_name: str
    precision: str
    concurrency: int
    prompt_tokens: int
    output_tokens: int
    ttft_ms: float
    itl_mean_ms: float
    total_latency_ms: float
    tokens_per_sec: float
    gpu_mem_allocated_mb: float
    gpu_mem_reserved_mb: float
    success: bool
    error: Optional[str] = None


class BenchmarkRunner:
    """Collects individual benchmark runs and exports aggregate statistics."""

    def __init__(self, output_dir: str = "benchmarks/results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results: List[BenchmarkResult] = []

    def record(self, result: BenchmarkResult) -> None:
        self.results.append(result)

    def to_dataframe(self) -> pd.DataFrame:
        if not self.results:
            return pd.DataFrame()
        return pd.DataFrame([asdict(r) for r in self.results])

    def save_csv(self, filename: str = "benchmark_summary.csv") -> str:
        df = self.to_dataframe()
        path = os.path.join(self.output_dir, filename)
        df.to_csv(path, index=False)
        logger.info(f"Saved benchmark results to {path}")
        return path

    def generate_summary_report(self) -> str:
        """Compute aggregate summary stats grouped by scenario and concurrency."""
        df = self.to_dataframe()
        if df.empty:
            return "No benchmark results recorded."

        success_df = df[df["success"]]
        summary = (
            success_df.groupby(["scenario", "precision", "concurrency"])
            .agg(
                samples=("total_latency_ms", "count"),
                avg_ttft_ms=("ttft_ms", "mean"),
                avg_itl_ms=("itl_mean_ms", "mean"),
                p50_latency_ms=("total_latency_ms", "median"),
                p95_latency_ms=("total_latency_ms", lambda x: np.percentile(x, 95)),
                avg_tokens_per_sec=("tokens_per_sec", "mean"),
                aggregate_tps=("tokens_per_sec", "sum"),
                peak_gpu_mem_mb=("gpu_mem_allocated_mb", "max"),
            )
            .reset_index()
        )

        return summary.to_string(index=False)
