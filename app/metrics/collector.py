"""
Thread-safe metrics collection and latency percentile calculations.
"""

import time
import threading
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class RequestMetrics:
    request_id: str
    arrival_time: float
    start_time: float
    end_time: float
    first_token_time: Optional[float] = None
    num_input_tokens: int = 0
    num_output_tokens: int = 0
    finish_reason: str = ""
    error: Optional[str] = None

    @property
    def queue_time_ms(self) -> float:
        return (self.start_time - self.arrival_time) * 1000

    @property
    def total_time_ms(self) -> float:
        return (self.end_time - self.arrival_time) * 1000

    @property
    def generation_time_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000

    @property
    def ttft_ms(self) -> Optional[float]:
        if self.first_token_time and self.start_time:
            return (self.first_token_time - self.start_time) * 1000
        return None

    @property
    def tokens_per_second(self) -> float:
        gen_time = self.end_time - self.start_time
        if gen_time > 0 and self.num_output_tokens > 0:
            return self.num_output_tokens / gen_time
        return 0.0


class MetricsCollector:
    """Thread-safe collector for request latencies, throughput, and cache events."""

    def __init__(self, max_history: int = 10000):
        self._lock = threading.Lock()
        self._max_history = max_history
        self._history: list[RequestMetrics] = []
        self._total_requests = 0
        self._completed_requests = 0
        self._failed_requests = 0
        self._cancelled_requests = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._cache_hits = 0
        self._cache_misses = 0

    def record_request(self, metrics: RequestMetrics) -> None:
        """Record metrics for a completed request."""
        with self._lock:
            self._total_requests += 1
            if metrics.error:
                self._failed_requests += 1
            else:
                self._completed_requests += 1
            self._total_input_tokens += metrics.num_input_tokens
            self._total_output_tokens += metrics.num_output_tokens

            self._history.append(metrics)
            # Trim history
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]

    def record_cache_hit(self) -> None:
        with self._lock:
            self._cache_hits += 1

    def record_cache_miss(self) -> None:
        with self._lock:
            self._cache_misses += 1

    def record_cancellation(self) -> None:
        with self._lock:
            self._cancelled_requests += 1

    def get_summary(self) -> dict:
        """Return aggregate metrics."""
        with self._lock:
            completed = [m for m in self._history if not m.error]
            latencies = [m.total_time_ms for m in completed]
            tps_values = [m.tokens_per_second for m in completed if m.tokens_per_second > 0]

            summary = {
                "total_requests": self._total_requests,
                "completed_requests": self._completed_requests,
                "failed_requests": self._failed_requests,
                "cancelled_requests": self._cancelled_requests,
                "total_input_tokens": self._total_input_tokens,
                "total_output_tokens": self._total_output_tokens,
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
            }

            if latencies:
                latencies_sorted = sorted(latencies)
                summary["avg_latency_ms"] = sum(latencies) / len(latencies)
                summary["p50_latency_ms"] = latencies_sorted[len(latencies_sorted) // 2]
                summary["p95_latency_ms"] = latencies_sorted[int(len(latencies_sorted) * 0.95)]
                summary["p99_latency_ms"] = latencies_sorted[int(len(latencies_sorted) * 0.99)]

            if tps_values:
                summary["avg_tokens_per_second"] = sum(tps_values) / len(tps_values)

            ttfts = [m.ttft_ms for m in completed if m.ttft_ms is not None]
            if ttfts:
                summary["avg_ttft_ms"] = sum(ttfts) / len(ttfts)

            return summary

    def get_history(self) -> list[dict]:
        """Return raw per-request metrics as dicts."""
        with self._lock:
            return [
                {
                    "request_id": m.request_id,
                    "queue_time_ms": round(m.queue_time_ms, 2),
                    "total_time_ms": round(m.total_time_ms, 2),
                    "generation_time_ms": round(m.generation_time_ms, 2),
                    "ttft_ms": round(m.ttft_ms, 2) if m.ttft_ms else None,
                    "num_input_tokens": m.num_input_tokens,
                    "num_output_tokens": m.num_output_tokens,
                    "tokens_per_second": round(m.tokens_per_second, 1),
                    "finish_reason": m.finish_reason,
                    "error": m.error,
                }
                for m in self._history
            ]

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._history.clear()
            self._total_requests = 0
            self._completed_requests = 0
            self._failed_requests = 0
            self._cancelled_requests = 0
            self._total_input_tokens = 0
            self._total_output_tokens = 0
            self._cache_hits = 0
            self._cache_misses = 0
