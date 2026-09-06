"""Headless FastAPI transport for the durable Runtime V2 daemon."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from core.runtime.bootstrap import create_runtime
from core.runtime.contracts import ApprovalDecision
from core.runtime.engine import HeathcliffRuntime
from logger import logger


class TurnRequest(BaseModel):
    content: str = Field(min_length=1, max_length=10000)
    idempotency_key: str = Field(default_factory=lambda: str(uuid4()))


class ApprovalRequestBody(BaseModel):
    approved: bool


def browser_event(kind: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """Project a durable event into browser-safe event data."""
    safe_payload = dict(payload)
    provider_state = safe_payload.pop("provider_state", None)
    if isinstance(provider_state, dict) and provider_state.get("usage"):
        safe_payload["usage"] = provider_state["usage"]
    return {"kind": getattr(kind, "value", str(kind)), "payload": safe_payload}


def _report_background_failure(task: asyncio.Task[object]) -> None:
    if task.cancelled():
        return
    try:
        task.result()
    except Exception:
        logger.exception("Runtime turn background task failed")


def create_app(runtime: HeathcliffRuntime) -> FastAPI:
    app = FastAPI(title="Heathcliff Runtime V2", version="2")

    @app.post("/v2/threads")
    async def create_thread() -> dict[str, str]:
        thread = await runtime.create_thread()
        return {"thread_id": str(thread.id)}

    @app.post("/v2/threads/{thread_id}/turns", status_code=202)
    async def create_turn(thread_id: UUID, body: TurnRequest) -> dict[str, str]:
        pending = await runtime.admit_input(
            thread_id, body.content, body.idempotency_key
        )
        task = asyncio.create_task(runtime.run_input(pending))
        task.add_done_callback(_report_background_failure)
        events = await runtime.store.events(thread_id)
        return {
            "input_id": str(pending.id),
            "thread_id": str(thread_id),
            "event_cursor": str(events[-1].sequence if events else 0),
        }

    @app.post("/v2/turns/{turn_id}/cancel", status_code=202)
    async def cancel_turn(turn_id: UUID) -> dict[str, str]:
        await runtime.cancel(turn_id)
        return {"turn_id": str(turn_id), "status": "cancellation_requested"}

    @app.post("/v2/approvals/{approval_id}/decision", status_code=202)
    async def decide_approval(
        approval_id: UUID, body: ApprovalRequestBody
    ) -> dict[str, str]:
        try:
            turn = await runtime.decide_approval(
                ApprovalDecision(approval_id=approval_id, approved=body.approved)
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        events = await runtime.store.events(turn.thread_id)
        terminal = next(
            (
                event
                for event in reversed(events)
                if event.turn_id == turn.id and event.kind.value == "turn.completed"
            ),
            None,
        )
        return {
            "turn_id": str(turn.id),
            "status": turn.status.value,
            "response": str((terminal.payload if terminal else {}).get("response", "")),
        }

    @app.get("/v2/threads/{thread_id}/events")
    async def events(
        thread_id: UUID,
        after: int = Query(default=0, ge=0),
    ) -> StreamingResponse:
        async def stream() -> AsyncIterator[str]:
            cursor = after
            idle_polls = 0
            while idle_polls < 120:
                batch = await runtime.store.events(thread_id, cursor)
                if batch:
                    idle_polls = 0
                    for event in batch:
                        cursor = event.sequence
                        payload = browser_event(event.kind, event.payload)
                        yield f"id: {cursor}\nevent: {event.kind.value}\ndata: {json.dumps(payload)}\n\n"
                        if event.kind.value in {
                            "approval.required",
                            "turn.completed",
                            "turn.cancelled",
                            "turn.failed",
                        }:
                            return
                else:
                    idle_polls += 1
                    yield ": heartbeat\n\n"
                    await asyncio.sleep(0.25)

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.get("/v2/runtime/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v2/runtime/ready")
    async def ready() -> dict[str, str]:
        if not await runtime.store.ready():
            raise HTTPException(status_code=503, detail="runtime store unavailable")
        if runtime.artifact_store and not await runtime.artifact_store.ready():
            raise HTTPException(status_code=503, detail="artifact store unavailable")
        return {"status": "ready"}

    return app


async def serve() -> None:
    """Run the standalone daemon after the canonical store is ready."""
    import uvicorn

    runtime = await create_runtime()
    config = uvicorn.Config(create_app(runtime), host="0.0.0.0", port=8700)
    await uvicorn.Server(config).serve()


if __name__ == "__main__":
    asyncio.run(serve())
