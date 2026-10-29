"""Tests for radix prefix caching, LRU eviction, and block reference management."""

import pytest
from app.cache.block_manager import BlockManager
from app.cache.prefix_cache import PrefixCache


class TestPrefixCache:
    """Test suite for PrefixCache."""

    @pytest.fixture
    def setup_cache(self):
        bm = BlockManager(num_blocks=20, block_size=16)
        pc = PrefixCache(block_manager=bm, model_name="test-model", max_entries=3, block_size=16)
        return bm, pc

    def test_cache_miss_on_empty(self, setup_cache):
        _, pc = setup_cache
        tokens = list(range(32))  # 2 blocks
        shared_blocks, matched_tokens = pc.match_prefix(tokens)
        assert shared_blocks == []
        assert matched_tokens == 0
        assert pc.misses == 1
        assert pc.hits == 0

    def test_insert_and_hit_identical_prefix(self, setup_cache):
        bm, pc = setup_cache
        tokens = list(range(32))  # 32 tokens = 2 blocks of 16

        # Allocate blocks in block manager
        t1 = bm.allocate_for_request("req1", 32)
        pc.insert_prefix(tokens, t1.get_block_ids())

        assert len(pc._cache) == 2  # Block 1 prefix, Block 1+2 prefix
        assert pc.get_stats()["cached_entries"] == 2

        # Second request with identical tokens
        shared_blocks, matched = pc.match_prefix(tokens)
        assert matched == 32
        assert shared_blocks == t1.get_block_ids()
        assert pc.hits == 1

    def test_different_prefixes_do_not_match(self, setup_cache):
        bm, pc = setup_cache
        tokens_a = list(range(100, 132))
        tokens_b = list(range(200, 232))

        t1 = bm.allocate_for_request("reqA", 32)
        pc.insert_prefix(tokens_a, t1.get_block_ids())

        shared_blocks, matched = pc.match_prefix(tokens_b)
        assert shared_blocks == []
        assert matched == 0
        assert pc.misses == 1

    def test_prompt_is_prefix_of_another(self, setup_cache):
        bm, pc = setup_cache
        base_prefix = list(range(32))  # 2 blocks
        extended_prompt = base_prefix + list(range(100, 116))  # 3 blocks

        t1 = bm.allocate_for_request("req1", 32)
        pc.insert_prefix(base_prefix, t1.get_block_ids())

        # Match extended prompt: should match the 32 tokens of the base prefix
        shared_blocks, matched = pc.match_prefix(extended_prompt)
        assert matched == 32
        assert shared_blocks == t1.get_block_ids()
        assert pc.hits == 1

    def test_lru_eviction(self, setup_cache):
        bm, pc = setup_cache
        # pc max_entries is 3
        # Insert entry 1 (1 block)
        tok1 = [1] * 16
        t1 = bm.allocate_for_request("r1", 16)
        pc.insert_prefix(tok1, t1.get_block_ids())

        # Insert entry 2 (1 block)
        tok2 = [2] * 16
        t2 = bm.allocate_for_request("r2", 16)
        pc.insert_prefix(tok2, t2.get_block_ids())

        # Insert entry 3 (1 block)
        tok3 = [3] * 16
        t3 = bm.allocate_for_request("r3", 16)
        pc.insert_prefix(tok3, t3.get_block_ids())

        assert len(pc._cache) == 3
        assert pc.evictions == 0

        # Now insert entry 4; entry 1 should be evicted
        tok4 = [4] * 16
        t4 = bm.allocate_for_request("r4", 16)
        pc.insert_prefix(tok4, t4.get_block_ids())

        assert len(pc._cache) == 3
        assert pc.evictions == 1

        # Match tok1 should now miss
        shared, matched = pc.match_prefix(tok1)
        assert matched == 0
        assert shared == []

    def test_model_name_isolation(self):
        bm = BlockManager(num_blocks=10, block_size=16)
        pc1 = PrefixCache(bm, model_name="model-A", block_size=16)
        pc2 = PrefixCache(bm, model_name="model-B", block_size=16)

        tokens = list(range(16))
        t1 = bm.allocate_for_request("r1", 16)
        pc1.insert_prefix(tokens, t1.get_block_ids())

        # Matching with pc2 (different model) should miss
        shared, matched = pc2.match_prefix(tokens)
        assert matched == 0
        assert shared == []
