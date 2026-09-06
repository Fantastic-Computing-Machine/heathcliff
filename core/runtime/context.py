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
                id=checkpoint.id,
                thread_id=thread_id,
                turn_id=UUID(int=0),
                kind="context_checkpoint",
                content=checkpoint.model_dump(mode="json"),
                provider_state=checkpoint.provider_state,
            )
        )
    ordered = sorted(events, key=lambda event: event.sequence)
    inputs = {event.payload.get("input_id"): event for event in ordered
              if event.kind == EventKind.INPUT_ADMITTED}
    by_turn: dict[UUID, list[RuntimeEvent]] = {}
    for event in ordered:
        if event.turn_id is not None and (
            checkpoint is None or event.sequence > checkpoint.through_sequence
        ):
            by_turn.setdefault(event.turn_id, []).append(event)
    # Compaction, not projection, owns history removal. Never silently discard
    # user decisions or split a native model call from its tool results.
    for turn_id in by_turn:
        for event in by_turn[turn_id]:
            payload = event.payload
            source_id = event.id
            if event.kind == EventKind.TURN_STARTED:
                source = inputs.get(payload.get("input_id"))
                if source is None:
                    continue
                source_id = source.id
                kind = "user_message"
                content = {"text": source.payload["content"]}
            elif event.kind == EventKind.MODEL_COMPLETED:
                kind = "model_message"
                content = payload.get("provider_state", {}).get("content")
                if not content:
                    parts = [{"text": payload["text"]}] if payload.get("text") else []
                    for call in payload.get("calls", []):
                        function = {"name": call["name"], "args": call["arguments"]}
                        if call.get("provider_call_id"):
                            function["id"] = call["provider_call_id"]
                        parts.append({"function_call": function})
                    if not parts:
                        continue
                    content = {"role": "model", "parts": parts}
            elif event.kind in {EventKind.TOOL_COMPLETED, EventKind.TOOL_OUTCOME_UNKNOWN}:
                kind = "tool_result"
                content = {"name": payload["tool"],
                           "provider_call_id": payload.get("provider_call_id"),
                           "response": {key: payload.get(key) for key in
                                        ("outcome", "output", "error", "verification")}}
            else:
                continue
            visible.append(
                RuntimeItem(
                    id=source_id,
                    thread_id=thread_id,
                    turn_id=turn_id,
                    kind=kind,
                    content=content,
                )
            )
    return visible
