"""
KV Cache block data structure and backing physical tensor pool.
"""

from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class KVBlock:
    """Fixed-size key-value cache block metadata."""
    block_id: int
    block_size: int = 16
    ref_count: int = 0
    num_tokens: int = 0
    is_free: bool = True
    hash_key: Optional[int] = None

    @property
    def is_full(self) -> bool:
        return self.num_tokens >= self.block_size

    @property
    def free_slots(self) -> int:
        return max(0, self.block_size - self.num_tokens)

    def append_tokens(self, count: int = 1) -> int:
        can_add = min(count, self.free_slots)
        self.num_tokens += can_add
        return can_add

    def reset(self) -> None:
        self.ref_count = 0
        self.num_tokens = 0
        self.is_free = True
        self.hash_key = None


class KVCacheTensorPool:
    """Pre-allocated contiguous tensor storage for key-value cache blocks."""

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
        device: str = "cpu",
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        shape = (num_blocks, num_layers, num_kv_heads, block_size, head_dim)
        # Allocate key and value cache tensors
        self.key_cache = torch.zeros(shape, dtype=dtype, device=device)
        self.value_cache = torch.zeros(shape, dtype=dtype, device=device)

    @property
    def total_bytes(self) -> int:
        """Total memory footprint of the physical pool in bytes."""
        element_size = self.key_cache.element_size()
        return 2 * self.key_cache.numel() * element_size

    def clear_block(self, block_id: int) -> None:
        """Zero out the memory for a specific physical block."""
        if 0 <= block_id < self.num_blocks:
            self.key_cache[block_id].zero_()
            self.value_cache[block_id].zero_()
