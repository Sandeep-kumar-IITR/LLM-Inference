"""
Paged KV Cache block manager for dynamic memory allocation and reclamation.
"""

import math
import logging
from collections import deque
from typing import Dict, List, Optional

from app.cache.kv_block import KVBlock
from app.cache.block_table import BlockTable

logger = logging.getLogger(__name__)


class CacheOOMError(RuntimeError):
    """Raised when no free blocks are available in the KV cache."""
    pass


class BlockManager:
    """Manages allocation, reclamation, and mapping of KV cache blocks."""

    def __init__(self, num_blocks: int = 256, block_size: int = 16):
        if num_blocks <= 0 or block_size <= 0:
            raise ValueError("num_blocks and block_size must be positive integers.")

        self.num_blocks = num_blocks
        self.block_size = block_size
        self.blocks: List[KVBlock] = [
            KVBlock(block_id=i, block_size=block_size) for i in range(num_blocks)
        ]
        self.free_pool: deque[int] = deque(range(num_blocks))
        self.block_tables: Dict[str, BlockTable] = {}

        self.peak_allocated_blocks: int = 0
        self.total_allocations: int = 0
        self.total_reclaims: int = 0

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_pool)

    @property
    def num_allocated_blocks(self) -> int:
        return self.num_blocks - len(self.free_pool)

    def can_allocate(self, num_tokens: int) -> bool:
        required_blocks = math.ceil(num_tokens / self.block_size)
        return len(self.free_pool) >= required_blocks

    def can_append_token(self, request_id: str) -> bool:
        if request_id not in self.block_tables:
            return len(self.free_pool) >= 1

        table = self.block_tables[request_id]
        if table.num_blocks() == 0:
            return len(self.free_pool) >= 1

        last_block_id = table.physical_block_ids[-1]
        last_block = self.blocks[last_block_id]
        if not last_block.is_full:
            return True
        return len(self.free_pool) >= 1

    def allocate_for_request(
        self, request_id: str, num_tokens: int, shared_block_ids: Optional[List[int]] = None
    ) -> BlockTable:
        """Allocate blocks for a sequence during prefill."""
        if request_id in self.block_tables:
            raise ValueError(f"Request {request_id} already has allocated blocks.")

        shared_block_ids = shared_block_ids or []
        shared_tokens = len(shared_block_ids) * self.block_size
        remaining_tokens = max(0, num_tokens - shared_tokens)
        required_new_blocks = math.ceil(remaining_tokens / self.block_size)

        if len(self.free_pool) < required_new_blocks:
            raise CacheOOMError(
                f"Out of memory: requested {required_new_blocks} new blocks "
                f"for {remaining_tokens} tokens, but only {len(self.free_pool)} free."
            )

        table = BlockTable(request_id=request_id, block_size=self.block_size)

        for block_id in shared_block_ids:
            block = self.blocks[block_id]
            if block.is_free:
                raise ValueError(f"Cannot share free block {block_id}.")
            block.ref_count += 1
            table.append_block(block_id)

        for _ in range(required_new_blocks):
            block_id = self.free_pool.popleft()
            block = self.blocks[block_id]
            block.is_free = False
            block.ref_count = 1
            block.num_tokens = 0
            table.append_block(block_id)
            self.total_allocations += 1

        table.advance_tokens(num_tokens)
        curr_tokens = num_tokens
        for b_id in table.physical_block_ids:
            blk = self.blocks[b_id]
            tokens_in_blk = min(curr_tokens, self.block_size)
            blk.num_tokens = tokens_in_blk
            curr_tokens -= tokens_in_blk

        self.block_tables[request_id] = table

        current_allocated = self.num_allocated_blocks
        if current_allocated > self.peak_allocated_blocks:
            self.peak_allocated_blocks = current_allocated

        return table

    def append_slot_for_token(self, request_id: str) -> None:
        """Allocate an additional token slot for an active decoding sequence."""
        if request_id not in self.block_tables:
            raise KeyError(f"Request {request_id} not found in active block tables.")

        table = self.block_tables[request_id]

        if table.num_blocks() == 0 or table.num_tokens % self.block_size == 0:
            if not self.free_pool:
                raise CacheOOMError(
                    f"Out of memory during decoding for request {request_id}. "
                    f"Free pool exhausted ({self.num_allocated_blocks}/{self.num_blocks} in use)."
                )
            new_block_id = self.free_pool.popleft()
            new_block = self.blocks[new_block_id]
            new_block.is_free = False
            new_block.ref_count = 1
            new_block.num_tokens = 0
            table.append_block(new_block_id)
            self.total_allocations += 1

            if self.num_allocated_blocks > self.peak_allocated_blocks:
                self.peak_allocated_blocks = self.num_allocated_blocks

        table.advance_tokens(1)
        last_block_id = table.physical_block_ids[-1]
        self.blocks[last_block_id].num_tokens = (
            (table.num_tokens - 1) % self.block_size + 1
        )

    def free_request(self, request_id: str) -> None:
        """Release all physical blocks allocated to the given request."""
        if request_id not in self.block_tables:
            return

        table = self.block_tables.pop(request_id)
        for block_id in table.physical_block_ids:
            self._release_block(block_id)

    def _release_block(self, block_id: int) -> None:
        block = self.blocks[block_id]
        if block.is_free:
            raise RuntimeError(f"Double free detected on block {block_id}!")

        block.ref_count -= 1
        if block.ref_count == 0:
            block.reset()
            self.free_pool.append(block_id)
            self.total_reclaims += 1

    def get_block_table(self, request_id: str) -> Optional[BlockTable]:
        return self.block_tables.get(request_id)

    def get_stats(self) -> dict:
        """Return memory utilization and allocation statistics."""
        allocated = self.num_allocated_blocks
        total_tokens = sum(table.num_tokens for table in self.block_tables.values())
        total_capacity = allocated * self.block_size
        internal_waste = total_capacity - total_tokens if total_capacity > 0 else 0
        utilization = (total_tokens / total_capacity) if total_capacity > 0 else 0.0

        return {
            "total_blocks": self.num_blocks,
            "allocated_blocks": allocated,
            "free_blocks": self.num_free_blocks,
            "peak_allocated_blocks": self.peak_allocated_blocks,
            "total_allocations": self.total_allocations,
            "total_reclaims": self.total_reclaims,
            "active_requests": len(self.block_tables),
            "total_cached_tokens": total_tokens,
            "internal_waste_tokens": internal_waste,
            "cache_utilization": round(utilization, 4),
        }
