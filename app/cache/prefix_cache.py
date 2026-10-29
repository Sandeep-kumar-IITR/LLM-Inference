"""
Prefix cache for zero-copy KV block reuse across shared prompt prefixes.
"""

import logging
import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Tuple, Optional

from app.cache.block_manager import BlockManager

logger = logging.getLogger(__name__)


@dataclass
class PrefixCacheEntry:
    prefix_key: str
    token_ids: Tuple[int, ...]
    block_ids: List[int]
    hit_count: int = 0


class PrefixCache:
    """LRU cache mapping token prefix hashes to allocated physical block IDs."""

    def __init__(
        self,
        block_manager: BlockManager,
        model_name: str = "default",
        max_entries: int = 128,
        block_size: int = 16,
    ):
        self.block_manager = block_manager
        self.model_name = model_name
        self.max_entries = max_entries
        self.block_size = block_size
        self._cache: OrderedDict[str, PrefixCacheEntry] = OrderedDict()

        self.hits: int = 0
        self.misses: int = 0
        self.evictions: int = 0

    def _compute_key(self, token_chunk: Tuple[int, ...]) -> str:
        hasher = hashlib.sha256()
        hasher.update(self.model_name.encode("utf-8"))
        hasher.update(str(token_chunk).encode("utf-8"))
        return hasher.hexdigest()

    def match_prefix(
        self, token_ids: List[int]
    ) -> Tuple[List[int], int]:
        """Find the longest cached prefix matching the given token sequence."""
        if not token_ids:
            return [], 0

        num_full_blocks = len(token_ids) // self.block_size
        if num_full_blocks == 0:
            self.misses += 1
            return [], 0

        matched_blocks: List[int] = []
        matched_token_count = 0

        for i in range(num_full_blocks):
            sub_tokens = tuple(token_ids[: (i + 1) * self.block_size])
            key = self._compute_key(sub_tokens)

            if key in self._cache:
                entry = self._cache[key]
                self._cache.move_to_end(key)
                entry.hit_count += 1
                matched_blocks = list(entry.block_ids)
                matched_token_count = len(sub_tokens)
            else:
                break

        if matched_token_count > 0:
            self.hits += 1
        else:
            self.misses += 1

        return matched_blocks, matched_token_count

    def insert_prefix(
        self, token_ids: List[int], physical_block_ids: List[int]
    ) -> None:
        """Cache block-aligned prompt prefixes for future reuse."""
        num_full_blocks = len(token_ids) // self.block_size
        if num_full_blocks == 0 or len(physical_block_ids) < num_full_blocks:
            return

        for i in range(num_full_blocks):
            sub_tokens = tuple(token_ids[: (i + 1) * self.block_size])
            key = self._compute_key(sub_tokens)
            sub_block_ids = physical_block_ids[: i + 1]

            if key not in self._cache:
                if len(self._cache) >= self.max_entries:
                    self._evict_lru()

                for b_id in sub_block_ids:
                    self.block_manager.blocks[b_id].ref_count += 1

                self._cache[key] = PrefixCacheEntry(
                    prefix_key=key,
                    token_ids=sub_tokens,
                    block_ids=list(sub_block_ids),
                )

    def _evict_lru(self) -> None:
        if not self._cache:
            return

        evicted_key, entry = self._cache.popitem(last=False)
        self.evictions += 1

        for b_id in entry.block_ids:
            self.block_manager._release_block(b_id)

    def clear(self) -> None:
        while self._cache:
            self._evict_lru()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get_stats(self) -> dict:
        total = self.hits + self.misses
        hit_rate = (self.hits / total) if total > 0 else 0.0
        return {
            "cached_entries": len(self._cache),
            "max_entries": self.max_entries,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": round(hit_rate, 4),
        }
