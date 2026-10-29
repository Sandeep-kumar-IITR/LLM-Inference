"""Tests for paged KV cache block manager and allocation policies."""

import pytest
from app.cache.block_manager import BlockManager, CacheOOMError
from app.cache.block_table import BlockTable


class TestBlockManager:
    """Test suite for BlockManager memory management."""

    def test_init(self):
        bm = BlockManager(num_blocks=16, block_size=16)
        assert bm.num_blocks == 16
        assert bm.block_size == 16
        assert bm.num_free_blocks == 16
        assert bm.num_allocated_blocks == 0
        assert bm.peak_allocated_blocks == 0

    def test_invalid_init(self):
        with pytest.raises(ValueError):
            BlockManager(num_blocks=0, block_size=16)
        with pytest.raises(ValueError):
            BlockManager(num_blocks=16, block_size=-1)

    def test_allocate_single_block_request(self):
        bm = BlockManager(num_blocks=10, block_size=16)
        table = bm.allocate_for_request(request_id="req1", num_tokens=10)

        assert table.request_id == "req1"
        assert table.num_blocks() == 1
        assert table.num_tokens == 10
        assert bm.num_allocated_blocks == 1
        assert bm.num_free_blocks == 9
        assert bm.peak_allocated_blocks == 1

        # Check block attributes
        block_id = table.physical_block_ids[0]
        block = bm.blocks[block_id]
        assert not block.is_free
        assert block.ref_count == 1
        assert block.num_tokens == 10

    def test_allocate_multi_block_request(self):
        bm = BlockManager(num_blocks=10, block_size=16)
        # 35 tokens requires ceil(35/16) = 3 blocks
        table = bm.allocate_for_request(request_id="req2", num_tokens=35)

        assert table.num_blocks() == 3
        assert bm.num_allocated_blocks == 3
        assert bm.num_free_blocks == 7

        # Check block filling: 16, 16, 3
        b0, b1, b2 = table.physical_block_ids
        assert bm.blocks[b0].num_tokens == 16
        assert bm.blocks[b1].num_tokens == 16
        assert bm.blocks[b2].num_tokens == 3

    def test_allocate_exact_block_boundary(self):
        bm = BlockManager(num_blocks=10, block_size=16)
        table = bm.allocate_for_request(request_id="req_exact", num_tokens=32)
        assert table.num_blocks() == 2
        assert bm.num_allocated_blocks == 2
        assert bm.blocks[table.physical_block_ids[0]].num_tokens == 16
        assert bm.blocks[table.physical_block_ids[1]].num_tokens == 16

    def test_duplicate_request_id_fails(self):
        bm = BlockManager(num_blocks=10, block_size=16)
        bm.allocate_for_request("req1", 10)
        with pytest.raises(ValueError):
            bm.allocate_for_request("req1", 10)

    def test_oom_on_allocation(self):
        bm = BlockManager(num_blocks=2, block_size=16)
        # Needs 3 blocks, but only 2 available
        with pytest.raises(CacheOOMError):
            bm.allocate_for_request("req_large", 40)

        # Ensure no blocks leaked
        assert bm.num_free_blocks == 2
        assert bm.num_allocated_blocks == 0

    def test_append_slot_within_existing_block(self):
        bm = BlockManager(num_blocks=5, block_size=16)
        table = bm.allocate_for_request("req1", 10)
        initial_blocks = table.num_blocks()

        # Adding 1 token when current block has 10/16 slots filled
        bm.append_slot_for_token("req1")
        assert table.num_tokens == 11
        assert table.num_blocks() == initial_blocks
        assert bm.num_allocated_blocks == 1

    def test_append_slot_triggers_new_block(self):
        bm = BlockManager(num_blocks=5, block_size=16)
        # Allocate exact full block (16 tokens)
        table = bm.allocate_for_request("req1", 16)
        assert table.num_blocks() == 1
        assert bm.num_allocated_blocks == 1

        # Adding 1 token should allocate block 2
        bm.append_slot_for_token("req1")
        assert table.num_tokens == 17
        assert table.num_blocks() == 2
        assert bm.num_allocated_blocks == 2

    def test_oom_during_token_append(self):
        bm = BlockManager(num_blocks=1, block_size=16)
        bm.allocate_for_request("req1", 16)

        # No more free blocks available for block 2
        with pytest.raises(CacheOOMError):
            bm.append_slot_for_token("req1")

    def test_free_request_and_reclamation(self):
        bm = BlockManager(num_blocks=5, block_size=16)
        table = bm.allocate_for_request("req1", 30)  # uses 2 blocks
        assert bm.num_allocated_blocks == 2

        bm.free_request("req1")
        assert bm.num_allocated_blocks == 0
        assert bm.num_free_blocks == 5
        assert bm.get_block_table("req1") is None
        assert bm.total_reclaims == 2

    def test_shared_block_reference_counting(self):
        bm = BlockManager(num_blocks=5, block_size=16)
        # Request 1 gets 2 blocks
        t1 = bm.allocate_for_request("req1", 32)
        shared_b0 = t1.physical_block_ids[0]

        # Request 2 shares the first block
        t2 = bm.allocate_for_request("req2", 20, shared_block_ids=[shared_b0])

        assert bm.blocks[shared_b0].ref_count == 2
        # Req1 frees its blocks; shared_b0 should NOT return to free pool
        bm.free_request("req1")
        assert bm.blocks[shared_b0].ref_count == 1
        assert not bm.blocks[shared_b0].is_free
        assert shared_b0 not in bm.free_pool

        # Req2 frees; now shared_b0 should return to free pool
        bm.free_request("req2")
        assert bm.blocks[shared_b0].ref_count == 0
        assert bm.blocks[shared_b0].is_free
        assert shared_b0 in bm.free_pool
        assert bm.num_free_blocks == 5

    def test_block_table_address_translation(self):
        table = BlockTable(request_id="req1", block_size=16)
        table.append_block(10)  # logical block 0 -> physical 10
        table.append_block(25)  # logical block 1 -> physical 25

        # Token 0 -> physical 10, offset 0
        p_id, offset = table.get_physical_location(0)
        assert (p_id, offset) == (10, 0)

        # Token 15 -> physical 10, offset 15
        p_id, offset = table.get_physical_location(15)
        assert (p_id, offset) == (10, 15)

        # Token 16 -> physical 25, offset 0
        p_id, offset = table.get_physical_location(16)
        assert (p_id, offset) == (25, 0)

        # Token 32 -> beyond capacity
        with pytest.raises(IndexError):
            table.get_physical_location(32)

    def test_stats(self):
        bm = BlockManager(num_blocks=10, block_size=16)
        bm.allocate_for_request("req1", 20)  # 2 blocks = 32 capacity, 20 tokens -> waste 12
        stats = bm.get_stats()
        assert stats["total_blocks"] == 10
        assert stats["allocated_blocks"] == 2
        assert stats["free_blocks"] == 8
        assert stats["total_cached_tokens"] == 20
        assert stats["internal_waste_tokens"] == 12
        assert stats["cache_utilization"] == round(20 / 32, 4)
