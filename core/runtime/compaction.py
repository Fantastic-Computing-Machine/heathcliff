"""Structured, provenance-preserving context compaction."""

from __future__ import annotations

import json
from typing import Protocol
from uuid import UUID

from core.runtime.contracts import ContextCheckpoint, EventKind, RuntimeEvent
from db.runtime_store import RuntimeStore


class CheckpointSummarizer(Protocol):
    async def summarize(
        self, thread_id: UUID, events: list[RuntimeEvent]
    ) -> ContextCheckpoint: ...


def estimated_tokens(events: list[RuntimeEvent]) -> int:
    """Stable conservative estimate used before a provider reports token counts."""
    return max(
        1, len(json.dumps([event.model_dump(mode="json") for event in events])) // 4
    )


class ContextCompactor:
    def __init__(
        self,
        store: RuntimeStore,
        summarizer: CheckpointSummarizer,
        context_window: int,
        trigger_ratio: float = 0.75,
        hard_ratio: float = 0.90,
    ) -> None:
        self.store = store
        self.summarizer = summarizer
        self.context_window = context_window
        self.trigger_ratio = trigger_ratio
        self.hard_ratio = hard_ratio

    async def maybe_compact(self, thread_id: UUID) -> ContextCheckpoint | None:
        events = await self.store.events(thread_id)
        if estimated_tokens(events) < self.context_window * self.trigger_ratio:
            return None
        checkpoint = await self.summarizer.summarize(thread_id, events)
        if checkpoint.through_sequence >= (events[-1].sequence if events else 0):
            raise ValueError("Checkpoint cannot compact the active event boundary")
        await self.store.save_checkpoint(checkpoint)
        await self.store.append_event(
            RuntimeEvent(
                thread_id=thread_id,
                kind=EventKind.CONTEXT_COMPACTED,
                payload={
                    "checkpoint_id": str(checkpoint.id),
                    "through_sequence": checkpoint.through_sequence,
                    "estimated_tokens": estimated_tokens(events),
                    "hard_limit_reached": estimated_tokens(events)
                    >= self.context_window * self.hard_ratio,
                },
            )
        )
        return checkpoint
