"""Tests for continuous batching scheduler and dynamic iteration-level admission."""

import pytest
from app.cache.block_manager import BlockManager
from app.cache.prefix_cache import PrefixCache
from app.scheduler.request import InferenceRequest, RequestStatus
from app.scheduler.continuous_batcher import ContinuousBatcher
from app.scheduler.worker import SchedulerWorker


class TestContinuousBatcher:
    """Test suite for ContinuousBatcher."""

    @pytest.fixture
    def setup_batcher(self):
        bm = BlockManager(num_blocks=20, block_size=16)
        pc = PrefixCache(bm, block_size=16)
        cb = ContinuousBatcher(block_manager=bm, prefix_cache=pc, max_batch_size=2)
        return bm, pc, cb

    def _make_req(self, req_id: str, prompt_len: int = 16, max_new_tokens: int = 5) -> InferenceRequest:
        return InferenceRequest(
            request_id=req_id,
            prompt=f"Prompt for {req_id}",
            prompt_token_ids=list(range(prompt_len)),
            max_new_tokens=max_new_tokens,
        )

    def test_enqueue_requests(self, setup_batcher):
        _, _, cb = setup_batcher
        req1 = self._make_req("r1")
        req2 = self._make_req("r2")

        cb.add_request(req1)
        cb.add_request(req2)

        assert cb.num_waiting == 2
        assert cb.num_running == 0
        assert req1.status == RequestStatus.WAITING

    def test_admit_up_to_max_batch_size(self, setup_batcher):
        _, _, cb = setup_batcher
        r1, r2, r3 = self._make_req("r1"), self._make_req("r2"), self._make_req("r3")

        cb.add_request(r1)
        cb.add_request(r2)
        cb.add_request(r3)

        # First schedule step should admit r1 and r2 (max_batch_size=2)
        prefill, decode = cb.schedule_step()

        assert len(prefill) == 2
        assert len(decode) == 0
        assert cb.num_running == 2
        assert cb.num_waiting == 1
        assert r1.status == RequestStatus.RUNNING
        assert r2.status == RequestStatus.RUNNING
        assert r3.status == RequestStatus.WAITING

    def test_continuous_batching_dynamic_completion(self, setup_batcher):
        """
        Demonstrates the key benefit of continuous batching:
        When a short request finishes, a waiting request is immediately admitted
        on the very next step, without waiting for longer requests in the batch to finish!
        """
        _, _, cb = setup_batcher
        # r1 finishes in 1 token; r2 finishes in 10 tokens
        r1 = self._make_req("r1", prompt_len=16, max_new_tokens=1)
        r2 = self._make_req("r2", prompt_len=16, max_new_tokens=10)
        r3 = self._make_req("r3", prompt_len=16, max_new_tokens=5)

        cb.add_request(r1)
        cb.add_request(r2)
        cb.add_request(r3)

        worker = SchedulerWorker(batcher=cb)

        # First step: admits r1 and r2; r1 finishes after single token
        active = worker.step_simulation()
        assert active == 2
        assert r1.status == RequestStatus.COMPLETED
        assert r2.status == RequestStatus.RUNNING

        # Second step: r1 is reclaimed; r3 is admitted into open slot
        active = worker.step_simulation()
        assert r3.status == RequestStatus.RUNNING
        assert cb.num_running == 2
        assert cb.num_waiting == 0

    def test_cancel_waiting_request(self, setup_batcher):
        _, _, cb = setup_batcher
        r1 = self._make_req("r1")
        cb.add_request(r1)

        cancelled = cb.cancel_request("r1")
        assert cancelled is True
        assert cb.num_waiting == 0
        assert r1.status == RequestStatus.CANCELLED

    def test_cancel_running_request_reclaims_memory(self, setup_batcher):
        bm, _, cb = setup_batcher
        r1 = self._make_req("r1", prompt_len=16)
        cb.add_request(r1)

        cb.schedule_step()
        assert cb.num_running == 1
        assert bm.num_allocated_blocks == 1

        cancelled = cb.cancel_request("r1")
        assert cancelled is True
        assert cb.num_running == 0
        assert bm.num_allocated_blocks == 0  # Memory cleanly freed!
        assert r1.status == RequestStatus.CANCELLED
