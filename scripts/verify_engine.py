"""
Verification script: loads the model, tests generation, streaming, and prints metrics.
"""

import sys
import os
import asyncio
import time
from pathlib import Path

# Force UTF-8 on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import load_config
from app.engine.inference_engine import InferenceEngine


def main():
    print("=" * 60)
    print("LLM INFERENCE ENGINE - VERIFICATION SUITE")
    print("=" * 60)

    # Allow custom model from argv or default
    model_name = sys.argv[1] if len(sys.argv) > 1 else None

    config = load_config()
    if model_name:
        config.model.name = model_name

    engine = InferenceEngine(config)

    print(f"\n1. Starting up engine with model: {config.model.name}")
    t0 = time.perf_counter()
    engine.startup()
    t_load = time.perf_counter() - t0
    print(f"Engine started in {t_load:.2f}s")

    health = engine.get_health()
    print(f"Health Status: {health['status']}")
    print(f"Model: {health['model_name']} on device {health['device']} ({health['precision']})")
    if health.get("gpu_memory_allocated_mb") is not None:
        print(f"GPU Memory Allocated: {health['gpu_memory_allocated_mb']:.1f} MB")
        print(f"GPU Memory Reserved:  {health['gpu_memory_reserved_mb']:.1f} MB")

    print("\n2. Testing Single Generation...")
    prompt = "Explain what a KV cache does in an LLM in one concise sentence."
    rid, out = engine.generate(prompt=prompt, max_new_tokens=32, temperature=0.7)

    # Safe display for all console encodings
    safe_text = out.text.strip().encode("ascii", errors="replace").decode("ascii")

    print(f"Request ID:        {rid}")
    print(f"Input Prompt:      {prompt}")
    print(f"Generated Text:    {safe_text}")
    print(f"Input Tokens:      {out.num_input_tokens}")
    print(f"Output Tokens:     {out.num_output_tokens}")
    print(f"Generation Time:   {out.total_time_s:.3f}s")
    tps = out.num_output_tokens / out.total_time_s if out.total_time_s > 0 else 0
    print(f"Throughput:        {tps:.1f} tokens/sec")

    print("\n3. Testing Async Streaming...")
    async def test_stream():
        stream_prompt = "Count from 1 to 5 with commas."
        print(f"Stream Prompt: {stream_prompt}")
        print("Tokens received: ", end="", flush=True)
        async for s_rid, chunk, final_out in engine.generate_stream(
            prompt=stream_prompt, max_new_tokens=24, temperature=0.0
        ):
            if chunk:
                safe_chunk = chunk.encode("ascii", errors="replace").decode("ascii")
                print(safe_chunk, end="", flush=True)
            if final_out:
                print(f"\n[Stream Finished: {final_out.num_output_tokens} tokens in {final_out.total_time_s:.3f}s]")

    asyncio.run(test_stream())

    print("\n4. Final Metrics:")
    metrics = engine.get_metrics()
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print("\n5. Shutting down engine...")
    engine.shutdown()
    print("Verification complete!")


if __name__ == "__main__":
    main()
