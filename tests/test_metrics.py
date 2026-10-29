"""
Tests for the metrics collector.

Verifies metric recording, aggregation, percentile calculations,
and thread safety.
"""

import pytest
import time
import threading

from app.metrics.collector import MetricsCollector, RequestMetrics


class TestRequestMetrics:
    """Test individual request metrics calculations."""

    def test_queue_time(self):
        m = RequestMetrics(
            request_id="test-1",
            arrival_time=100.0,
            start_time=100.5,
            end_time=101.0,
        )
        assert abs(m.queue_time_ms - 500.0) < 0.1

    def test_total_time(self):
        m = RequestMetrics(
            request_id="test-1",
            arrival_time=100.0,
            start_time=100.1,
            end_time=100.6,
        )
        assert abs(m.total_time_ms - 600.0) < 0.1

    def test_generation_time(self):
        m = RequestMetrics(
            request_id="test-1",
            arrival_time=100.0,
            start_time=100.1,
            end_time=100.6,
        )
        assert abs(m.generation_time_ms - 500.0) < 0.1

    def test_ttft(self):
        m = RequestMetrics(
            request_id="test-1",
            arrival_time=100.0,
            start_time=100.1,
            end_time=100.6,
            first_token_time=100.2,
        )
        assert abs(m.ttft_ms - 100.0) < 0.1

    def test_ttft_none_when_no_first_token(self):
        m = RequestMetrics(
            request_id="test-1",
            arrival_time=100.0,
            start_time=100.1,
            end_time=100.6,
        )
        assert m.ttft_ms is None

    def test_tokens_per_second(self):
        m = RequestMetrics(
            request_id="test-1",
            arrival_time=100.0,
            start_time=100.0,
            end_time=101.0,
            num_output_tokens=50,
        )
        assert abs(m.tokens_per_second - 50.0) < 0.1

    def test_tokens_per_second_zero_time(self):
        m = RequestMetrics(
            request_id="test-1",
            arrival_time=100.0,
            start_time=100.0,
            end_time=100.0,
            num_output_tokens=10,
        )
        assert m.tokens_per_second == 0.0


class TestMetricsCollector:
    """Test the aggregate metrics collector."""

    def _make_metric(self, request_id="test", duration=1.0, tokens=10, error=None):
        t = time.perf_counter()
        return RequestMetrics(
            request_id=request_id,
            arrival_time=t,
            start_time=t + 0.01,
            end_time=t + 0.01 + duration,
            first_token_time=t + 0.05,
            num_input_tokens=5,
            num_output_tokens=tokens,
            finish_reason="eos" if not error else "error",
            error=error,
        )

    def test_empty_collector(self):
        c = MetricsCollector()
        summary = c.get_summary()
        assert summary["total_requests"] == 0
        assert summary["completed_requests"] == 0

    def test_record_one_request(self):
        c = MetricsCollector()
        c.record_request(self._make_metric("r1"))
        summary = c.get_summary()
        assert summary["total_requests"] == 1
        assert summary["completed_requests"] == 1
        assert summary["failed_requests"] == 0

    def test_record_failed_request(self):
        c = MetricsCollector()
        c.record_request(self._make_metric("r1", error="OOM"))
        summary = c.get_summary()
        assert summary["total_requests"] == 1
        assert summary["completed_requests"] == 0
        assert summary["failed_requests"] == 1

    def test_multiple_requests(self):
        c = MetricsCollector()
        for i in range(10):
            c.record_request(self._make_metric(f"r{i}", duration=0.5, tokens=20))
        summary = c.get_summary()
        assert summary["total_requests"] == 10
        assert summary["completed_requests"] == 10
        assert summary["total_output_tokens"] == 200

    def test_cache_hits_misses(self):
        c = MetricsCollector()
        c.record_cache_hit()
        c.record_cache_hit()
        c.record_cache_miss()
        summary = c.get_summary()
        assert summary["cache_hits"] == 2
        assert summary["cache_misses"] == 1

    def test_history_trimming(self):
        c = MetricsCollector(max_history=5)
        for i in range(10):
            c.record_request(self._make_metric(f"r{i}"))
        history = c.get_history()
        assert len(history) == 5

    def test_reset(self):
        c = MetricsCollector()
        c.record_request(self._make_metric("r1"))
        c.record_cache_hit()
        c.reset()
        summary = c.get_summary()
        assert summary["total_requests"] == 0
        assert summary["cache_hits"] == 0

    def test_thread_safety(self):
        """Record metrics from multiple threads concurrently."""
        c = MetricsCollector()
        errors = []

        def record_batch(start_id):
            try:
                for i in range(100):
                    c.record_request(self._make_metric(f"r{start_id}-{i}"))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_batch, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        summary = c.get_summary()
        assert summary["total_requests"] == 400

    def test_percentiles(self):
        """Test p50/p95/p99 calculations with enough data."""
        c = MetricsCollector()
        for i in range(100):
            c.record_request(self._make_metric(f"r{i}", duration=float(i + 1) / 100))
        summary = c.get_summary()
        assert "p50_latency_ms" in summary
        assert "p95_latency_ms" in summary
        assert "p99_latency_ms" in summary
        # p95 should be higher than p50
        assert summary["p95_latency_ms"] >= summary["p50_latency_ms"]
