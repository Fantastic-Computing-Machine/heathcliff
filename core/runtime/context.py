"""Deterministic model-context projection from the canonical event journal."""

from __future__ import annotations

from typing import Iterable
from uuid import UUID

from core.runtime.contracts import (
    ContextCheckpoint,
    EventKind,
    RuntimeEvent,
    RuntimeItem,
)


def build_context(
    thread_id: UUID,
    events: Iterable[RuntimeEvent],
    checkpoint: ContextCheckpoint | None = None,
    recent_turn_limit: int = 6,
    tool_output_limit: int = 2000,
) -> list[RuntimeItem]:
    """Build stable model-visible items without mutating canonical history."""
    visible: list[RuntimeItem] = []
    if checkpoint is not None:
        visible.append(
            RuntimeItem(
                thread_id=thread_id,
                turn_id=UUID(int=0),
                kind="context_checkpoint",
                content=checkpoint.model_dump(mode="json"),
                provider_state=checkpoint.provider_state,
            )
        )
    by_turn: dict[UUID, list[RuntimeEvent]] = {}
    for event in events:
        if event.turn_id is not None and (
            checkpoint is None or event.sequence > checkpoint.through_sequence
        ):
            by_turn.setdefault(event.turn_id, []).append(event)
    for turn_id in list(by_turn)[-recent_turn_limit:]:
        for event in by_turn[turn_id]:
            payload = dict(event.payload)
            if event.kind == EventKind.TOOL_COMPLETED and isinstance(
                payload.get("output"), str
            ):
                payload["output"] = payload["output"][:tool_output_limit]
            visible.append(
                RuntimeItem(
                    thread_id=thread_id,
                    turn_id=turn_id,
                    kind=event.kind.value,
                    content=payload,
                )
            )
    return visible
