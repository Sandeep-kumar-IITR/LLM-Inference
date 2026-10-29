"""
Continuous batching execution worker loop.
"""

import asyncio
import logging
from typing import Optional

from app.scheduler.request import InferenceRequest
from app.scheduler.continuous_batcher import ContinuousBatcher

logger = logging.getLogger(__name__)


class SchedulerWorker:
    """Worker driving continuous scheduling steps."""

    def __init__(
        self,
        batcher: ContinuousBatcher,
        step_interval: float = 0.005,
    ):
        self.batcher = batcher
        self.step_interval = step_interval
        self._running = False
        self._task: Optional[asyncio.Task] = None

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    def step_simulation(self) -> int:
        """Step active requests and emit synthetic tokens for testing."""
        prefill_reqs, decode_reqs = self.batcher.schedule_step()

        for req in prefill_reqs:
            self.batcher.register_completed_prefix(req)
            req.add_output_token(token_id=100, token_text=" token")
            if req.num_output_tokens >= req.max_new_tokens:
                req.mark_completed(reason="length")

        for req in decode_reqs:
            req.add_output_token(token_id=101, token_text=" token")
            if req.num_output_tokens >= req.max_new_tokens:
                req.mark_completed(reason="length")

        return len(prefill_reqs) + len(decode_reqs)

    async def run_loop(self) -> None:
        self._running = True
        while self._running:
            if self.batcher.num_waiting > 0 or self.batcher.num_running > 0:
                self.step_simulation()
            await asyncio.sleep(self.step_interval)
