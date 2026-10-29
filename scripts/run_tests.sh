#!/usr/bin/env bash
# Run the test suite
echo "Running LLM Inference Engine tests..."
python -m pytest tests/ -v --tb=short "$@"
