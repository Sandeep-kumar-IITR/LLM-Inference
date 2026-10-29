"""
Inference request lifecycle and sequence tracking.
"""

import time
import asyncio
from enum import Enum
from typing import List, Optional
from dataclasses import dataclass, field


class RequestStatus(str, Enum):
    WAITING = "WAITING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


@dataclass
class InferenceRequest:
    """State tracking for a single generation sequence."""
    request_id: str
    prompt: str
    prompt_token_ids: List[int]
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1

    status: RequestStatus = RequestStatus.WAITING
    arrival_time: float = field(default_factory=time.perf_counter)
    start_time: Optional[float] = None
    finish_time: Optional[float] = None
    first_token_time: Optional[float] = None

    output_token_ids: List[int] = field(default_factory=list)
    output_text: str = ""
    finish_reason: Optional[str] = None
    error_msg: Optional[str] = None

    prefix_matched_tokens: int = 0
    shared_block_ids: List[int] = field(default_factory=list)
    token_queue: Optional[asyncio.Queue] = None

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self.output_token_ids)

    @property
    def total_tokens(self) -> int:
        return self.num_prompt_tokens + self.num_output_tokens

    @property
    def is_finished(self) -> bool:
        return self.status in (
            RequestStatus.COMPLETED,
            RequestStatus.CANCELLED,
            RequestStatus.FAILED,
        )

    def mark_running(self) -> None:
        self.status = RequestStatus.RUNNING
        self.start_time = time.perf_counter()

    def mark_completed(self, reason: str = "eos") -> None:
        self.status = RequestStatus.COMPLETED
        self.finish_reason = reason
        self.finish_time = time.perf_counter()

    def mark_failed(self, error: str) -> None:
        self.status = RequestStatus.FAILED
        self.finish_reason = "error"
        self.error_msg = error
        self.finish_time = time.perf_counter()

    def mark_cancelled(self) -> None:
        self.status = RequestStatus.CANCELLED
        self.finish_reason = "cancelled"
        self.finish_time = time.perf_counter()

    def add_output_token(self, token_id: int, token_text: str = "") -> None:
        if self.first_token_time is None:
            self.first_token_time = time.perf_counter()
        self.output_token_ids.append(token_id)
        self.output_text += token_text

        # Push to async queue if streaming
        if self.token_queue is not None:
            try:
                self.token_queue.put_nowait((token_text, None))
            except Exception:
                pass
