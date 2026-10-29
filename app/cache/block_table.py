"""
Logical-to-physical block mapping table for sequence token positions.
"""

from typing import List, Tuple


class BlockTable:
    """Tracks physical blocks assigned to a request sequence."""

    def __init__(self, request_id: str, block_size: int = 16):
        self.request_id = request_id
        self.block_size = block_size
        self.physical_block_ids: List[int] = []
        self.num_tokens: int = 0

    def append_block(self, physical_block_id: int) -> None:
        self.physical_block_ids.append(physical_block_id)

    def num_blocks(self) -> int:
        return len(self.physical_block_ids)

    def capacity(self) -> int:
        return len(self.physical_block_ids) * self.block_size

    def is_last_block_full(self) -> bool:
        if not self.physical_block_ids:
            return True
        return (self.num_tokens % self.block_size == 0) and (self.num_tokens > 0)

    def get_physical_location(self, token_idx: int) -> Tuple[int, int]:
        """Translate a sequence token index to (physical_block_id, block_offset)."""
        if token_idx < 0:
            raise IndexError(f"Token index {token_idx} cannot be negative.")

        logical_block_idx = token_idx // self.block_size
        offset = token_idx % self.block_size

        if logical_block_idx >= len(self.physical_block_ids):
            raise IndexError(
                f"Token index {token_idx} maps to block {logical_block_idx}, "
                f"but only {len(self.physical_block_ids)} blocks allocated."
            )

        return self.physical_block_ids[logical_block_idx], offset

    def get_block_ids(self) -> List[int]:
        return list(self.physical_block_ids)

    def advance_tokens(self, count: int = 1) -> None:
        if self.num_tokens + count > self.capacity():
            raise ValueError(
                f"Cannot advance {count} tokens; sequence has {self.num_tokens} tokens "
                f"and capacity is {self.capacity()}."
            )
        self.num_tokens += count
