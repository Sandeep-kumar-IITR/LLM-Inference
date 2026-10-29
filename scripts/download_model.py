"""
Model download utility using hf_transfer for fast downloads.
"""

import os
import sys
from huggingface_hub import snapshot_download

# Enable fast Rust-based multi-threaded hf_transfer
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

print(f"Downloading {MODEL_ID} with hf_transfer enabled...")
try:
    path = snapshot_download(repo_id=MODEL_ID)
    print(f"\nModel successfully downloaded to: {path}")
except Exception as e:
    print(f"Download error: {e}")
    sys.exit(1)
