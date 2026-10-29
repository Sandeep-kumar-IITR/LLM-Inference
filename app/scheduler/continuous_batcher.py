"""
Continuous batching scheduler for iteration-level dynamic batching.
"""

import time
import logging
from collections import deque
from typing import List, Optional, Tuple, Dict

from app.scheduler.request import InferenceRequest, RequestStatus
from app.cache.block_manager import BlockManager, CacheOOMError
from app.cache.prefix_cache import PrefixCache

logger = logging.getLogger(__name__)


class ContinuousBatcher:
    """Manages iteration-level continuous batching and admission control."""

    def __init__(
        self,
        block_manager: BlockManager,
        prefix_cache: Optional[PrefixCache] = None,
        max_batch_size: int = 8,
    ):
        self.block_manager = block_manager
        self.prefix_cache = prefix_cache
        self.max_batch_size = max_batch_size

        self.waiting_queue: deque[InferenceRequest] = deque()
        self.running_batch: List[InferenceRequest] = []
        self.completed_requests: Dict[str, InferenceRequest] = {}

        self.total_scheduled_steps: int = 0
        self.total_admitted_requests: int = 0
        self.total_completed_requests: int = 0
        self.total_cancelled_requests: int = 0

    @property
    def num_waiting(self) -> int:
        return len(self.waiting_queue)

    @property
    def num_running(self) -> int:
        return len(self.running_batch)

    def add_request(self, request: InferenceRequest) -> None:
        request.status = RequestStatus.WAITING
        self.waiting_queue.append(request)

    def cancel_request(self, request_id: str) -> bool:
        for req in list(self.waiting_queue):
            if req.request_id == request_id:
                req.mark_cancelled()
                self.waiting_queue.remove(req)
                self.total_cancelled_requests += 1
                return True

        for req in self.running_batch:
            if req.request_id == request_id:
                req.mark_cancelled()
                self.block_manager.free_request(request_id)
                self.running_batch.remove(req)
                self.total_cancelled_requests += 1
                return True

        return False

    def schedule_step(self) -> Tuple[List[InferenceRequest], List[InferenceRequest]]:
        """
        Execute one scheduling decision cycle.

        Returns (newly_admitted_prefill_requests, active_decode_requests).
        """
        self.total_scheduled_steps += 1

        reclaimed: List[InferenceRequest] = []
        still_running: List[InferenceRequest] = []

        for req in self.running_batch:
            if req.is_finished:
                self.block_manager.free_request(req.request_id)
                self.completed_requests[req.request_id] = req
                self.total_completed_requests += 1
                reclaimed.append(req)
            else:
                still_running.append(req)

        self.running_batch = still_running

        available_slots = self.max_batch_size - len(self.running_batch)
        newly_admitted: List[InferenceRequest] = []

        while self.waiting_queue and available_slots > 0:
            candidate = self.waiting_queue[0]

            shared_blocks: List[int] = []
            matched_tokens = 0
            if self.prefix_cache is not None:
                shared_blocks, matched_tokens = self.prefix_cache.match_prefix(
                    candidate.prompt_token_ids
                )

            unshared_tokens = candidate.num_prompt_tokens - matched_tokens
            if not self.block_manager.can_allocate(unshared_tokens):
                break

            candidate = self.waiting_queue.popleft()
            candidate.prefix_matched_tokens = matched_tokens
            candidate.shared_block_ids = shared_blocks

            try:
                self.block_manager.allocate_for_request(
                    request_id=candidate.request_id,
                    num_tokens=candidate.num_prompt_tokens,
                    shared_block_ids=shared_blocks,
                )
            except CacheOOMError:
                self.waiting_queue.appendleft(candidate)
                break

            candidate.mark_running()
            self.running_batch.append(candidate)
            newly_admitted.append(candidate)
            self.total_admitted_requests += 1
            available_slots -= 1

        decode_candidates: List[InferenceRequest] = []
        for req in self.running_batch:
            if req in newly_admitted:
                continue

            try:
                self.block_manager.append_slot_for_token(req.request_id)
                decode_candidates.append(req)
            except CacheOOMError:
                req.mark_failed("KV Cache OOM during decoding.")
                self.block_manager.free_request(req.request_id)

        self.running_batch = [r for r in self.running_batch if not r.is_finished]

        return newly_admitted, decode_candidates

    def register_completed_prefix(self, req: InferenceRequest) -> None:
        """Register the prompt prefix into prefix cache once prefill is done."""
        if self.prefix_cache is not None and not req.shared_block_ids:
            table = self.block_manager.get_block_table(req.request_id)
            if table is not None:
                self.prefix_cache.insert_prefix(
                    token_ids=req.prompt_token_ids,
                    physical_block_ids=table.get_block_ids(),
                )

    def get_stats(self) -> dict:
        return {
            "waiting_requests": self.num_waiting,
            "running_requests": self.num_running,
            "max_batch_size": self.max_batch_size,
            "total_scheduled_steps": self.total_scheduled_steps,
            "total_admitted": self.total_admitted_requests,
            "total_completed": self.total_completed_requests,
            "total_cancelled": self.total_cancelled_requests,
        }
