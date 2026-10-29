#!/usr/bin/env bash
# Start the FastAPI inference server
CONFIG=${1:-configs/default.yaml}

export LLM_ENGINE_CONFIG=$CONFIG
echo "Starting LLM Inference Engine with config: $CONFIG"
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
