#!/usr/bin/env bash
# Run benchmarks on Linux / WSL / Git Bash
SUITE=${1:-all}

echo "========================================================"
echo "Running LLM Inference Benchmarks: $SUITE"
echo "========================================================"

case $SUITE in
    latency)
        python benchmarks/benchmark_latency.py
        ;;
    throughput)
        python benchmarks/benchmark_throughput.py
        ;;
    concurrency)
        python benchmarks/benchmark_concurrency.py
        ;;
    quantization)
        python benchmarks/benchmark_quantization.py
        ;;
    all)
        python benchmarks/benchmark_latency.py
        python benchmarks/benchmark_throughput.py
        python benchmarks/benchmark_concurrency.py
        python benchmarks/benchmark_quantization.py
        ;;
    *)
        echo "Unknown suite: $SUITE"
        exit 1
        ;;
esac
