"""Async Runtime V2 turn runner."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Any
from uuid import UUID, uuid4

from core.providers.base import ModelProvider
from core.runtime.compaction import ContextCompactor
from core.runtime.context import build_context
from core.runtime.contracts import (
    ApprovalDecision,
    ApprovalRequest,
    EventKind,
    ModelRequest,
    PendingInput,
    PrivacyClass,
    RuntimeEvent,
    RuntimeLease,
    Thread,
    ToolCall,
    ToolOutcome,
    ToolResult,
    Turn,
    TurnStatus,
    utcnow,
)
from core.runtime.memory import MemoryJobWorker
from core.runtime.tools import ToolRegistry
from db.artifact_store import ArtifactStore
from db.runtime_store import RuntimeStore
from utils.langfuse_client import trace_observation, trace_runtime_turn


class HeathcliffRuntime:
    """Provider-neutral runtime with one active turn per in-process thread."""

    def __init__(
        self,
        store: RuntimeStore,
        provider: ModelProvider,
        tools: ToolRegistry,
        system_instruction: str,
        instance_id: str | None = None,
        max_model_steps: int = 20,
        lease_seconds: int = 30,
        compactor: ContextCompactor | None = None,
        memory_worker: MemoryJobWorker | None = None,
        artifact_store: ArtifactStore | None = None,
    ) -> None:
        self.store = store
        self.provider = provider
        self.tools = tools
        self.system_instruction = system_instruction
        self.instance_id = instance_id or str(uuid4())
        self.max_model_steps = max_model_steps
        self.lease_seconds = lease_seconds
        self.compactor = compactor
        self.memory_worker = memory_worker
        self.artifact_store = artifact_store
        self._thread_locks: dict[UUID, asyncio.Lock] = {}
        self._cancelled: set[UUID] = set()

    async def create_thread(self) -> Thread:
        return await self.store.create_thread()

    async def admit_input(
        self, thread_id: UUID, content: str, idempotency_key: str
    ) -> PendingInput:
        await self.store.ensure_thread(thread_id)
        pending = await self.store.admit_input(
            PendingInput(
                thread_id=thread_id,
                content=content,
                idempotency_key=idempotency_key,
            )
        )
        if pending.was_admitted:
            await self.store.append_event(
                RuntimeEvent(
                    thread_id=thread_id,
                    kind=EventKind.INPUT_ADMITTED,
                    payload={"input_id": str(pending.id), "content": pending.content},
                )
            )
        return pending

    async def run_input(self, pending: PendingInput) -> Turn:
        turn = Turn(thread_id=pending.thread_id, input_id=pending.id)
        stored_turn = await self.store.create_turn(turn)
        if not stored_turn.was_created:
            return stored_turn
        return await self._run_turn(stored_turn, user_input=pending.content)

    async def cancel(self, turn_id: UUID) -> None:
        self._cancelled.add(turn_id)

    async def decide_approval(self, decision: ApprovalDecision) -> Turn:
        approval = await self.store.decide_approval(decision)
        await self.store.append_event(
            RuntimeEvent(
                thread_id=approval.thread_id,
                turn_id=approval.turn_id,
                kind=EventKind.APPROVAL_DECIDED,
                payload={
                    "approval_id": str(approval.id),
                    "approved": decision.approved,
                },
            )
        )
        turn = Turn(
            id=approval.turn_id,
            thread_id=approval.thread_id,
            input_id=UUID(int=0),
            status=TurnStatus.RUNNING,
        )
        if not decision.approved:
            return await self._finish(
                turn, TurnStatus.CANCELLED, EventKind.TURN_CANCELLED
            )
        await self.store.set_turn_status(turn.id, TurnStatus.RUNNING)
        result = (await self._execute_calls(turn, [approval.tool_call]))[0]
        await self._append_tool_result(turn, approval.tool_call, result)
        return await self._run_turn(turn, existing=True)

    async def _run_turn(
        self, turn: Turn, existing: bool = False, user_input: str = ""
    ) -> Turn:
        lock = self._thread_locks.setdefault(turn.thread_id, asyncio.Lock())
        async with lock:
            lease = RuntimeLease(
                holder=self.instance_id,
                expires_at=utcnow() + timedelta(seconds=self.lease_seconds),
            )
            if not await self.store.acquire_lease(lease):
                raise RuntimeError("Runtime lease is held by another instance")
            try:
                with trace_runtime_turn(
                    thread_id=str(turn.thread_id),
                    turn_id=str(turn.id),
                    user_input=user_input or "Approval decision",
                ) as trace:
                    if not existing:
                        await self.store.set_turn_status(turn.id, TurnStatus.RUNNING)
                        await self.store.append_event(
                            RuntimeEvent(
                                thread_id=turn.thread_id,
                                turn_id=turn.id,
                                kind=EventKind.TURN_STARTED,
                            )
                        )
                    result = await self._model_loop(turn)
                    if trace is not None:
                        trace.update(
                            output={
                                "turn_id": str(result.id),
                                "status": result.status.value,
                            }
                        )
                    return result
            finally:
                await self.store.release_lease(lease.name, self.instance_id)

    async def _model_loop(self, turn: Turn) -> Turn:
        for _ in range(self.max_model_steps):
            if turn.id in self._cancelled:
                self._cancelled.remove(turn.id)
                return await self._finish(
                    turn, TurnStatus.CANCELLED, EventKind.TURN_CANCELLED
                )
            events = await self.store.events(turn.thread_id)
            if self.compactor is not None:
                await self.compactor.maybe_compact(turn.thread_id)
                events = await self.store.events(turn.thread_id)
            checkpoint = await self.store.latest_checkpoint(turn.thread_id)
            request = ModelRequest(
                thread_id=turn.thread_id,
                turn_id=turn.id,
                system_instruction=self.system_instruction,
                context=build_context(turn.thread_id, events, checkpoint),
                tools=self.tools.declarations(),
                provider=self.provider.capabilities,
            )
            call = await self.provider.prepare(request)
            await self.store.append_event(
                RuntimeEvent(
                    thread_id=turn.thread_id,
                    turn_id=turn.id,
                    kind=EventKind.MODEL_STARTED,
                    payload={
                        "model_call_id": str(call.id),
                        "effective_config": call.effective_config,
                    },
                )
            )
            with trace_observation(
                "runtime.model",
                as_type="generation",
                input={
                    "model_call_id": str(call.id),
                    "provider": request.provider.provider,
                    "model": request.provider.model,
                    "context_items": len(request.context),
                    "tool_count": len(request.tools),
                },
            ) as trace:
                text, calls, provider_state = await self._consume_model(call)
                if trace is not None:
                    trace.update(
                        output={
                            "text": text,
                            "tool_calls": [
                                call.model_dump(mode="json") for call in calls
                            ],
                            "provider_state": provider_state,
                        }
                    )
            if not calls:
                completed = await self.store.append_event(
                    RuntimeEvent(
                        thread_id=turn.thread_id,
                        turn_id=turn.id,
                        kind=EventKind.MODEL_COMPLETED,
                        payload={"text": text, "provider_state": provider_state},
                    )
                )
                finished = await self._finish(
                    turn,
                    TurnStatus.COMPLETED,
                    EventKind.TURN_COMPLETED,
                    {"response": text},
                )
                await self._enqueue_turn_memory(turn)
                return finished
            for tool_call in calls:
                await self.store.append_event(
                    RuntimeEvent(
                        thread_id=turn.thread_id,
                        turn_id=turn.id,
                        kind=EventKind.TOOL_PROPOSED,
                        payload={
                            "call": tool_call.model_dump(mode="json"),
                            "provider_state": provider_state,
                        },
                    )
                )
                if self.tools.requires_approval(tool_call):
                    contract = self.tools.get(tool_call.name).contract
                    approval = ApprovalRequest(
                        thread_id=turn.thread_id,
                        turn_id=turn.id,
                        tool_call=tool_call,
                        resource_scope=contract.resource_scope,
                    )
                    await self.store.save_approval(approval)
                    await self.store.set_turn_status(
                        turn.id, TurnStatus.WAITING_FOR_APPROVAL
                    )
                    await self.store.append_event(
                        RuntimeEvent(
                            thread_id=turn.thread_id,
                            turn_id=turn.id,
                            kind=EventKind.APPROVAL_REQUIRED,
                            payload=approval.model_dump(mode="json"),
                        )
                    )
                    return turn.model_copy(
                        update={"status": TurnStatus.WAITING_FOR_APPROVAL}
                    )
            results = await self._execute_calls(turn, calls)
            for tool_call, result in zip(calls, results):
                await self._append_tool_result(turn, tool_call, result)
        return await self._finish(
            turn,
            TurnStatus.FAILED,
            EventKind.TURN_FAILED,
            {"error": "maximum model steps exceeded"},
        )

    async def _consume_model(
        self, call: Any
    ) -> tuple[str, list[ToolCall], dict[str, Any]]:
        text: list[str] = []
        tool_calls: list[ToolCall] = []
        provider_state: dict[str, Any] = {}
        async for event in self.provider.stream(call):
            if event.kind == "text_delta":
                text.append(str(event.data.get("text", "")))
            elif event.kind == "tool_call":
                tool_calls.append(
                    ToolCall(
                        name=event.data["name"],
                        arguments=event.data.get("arguments", {}),
                    )
                )
            if event.data.get("thought_signature"):
                provider_state["thought_signature"] = event.data["thought_signature"]
        return "".join(text), tool_calls, provider_state

    async def _execute_calls(
        self, turn: Turn, calls: list[ToolCall]
    ) -> list[ToolResult]:
        """Parallelize declared-safe reads, with durable locks for every mutation."""
        results: dict[int, ToolResult] = {}
        reads: list[tuple[int, ToolCall]] = []
        effects: list[tuple[int, ToolCall]] = []
        for index, call in enumerate(calls):
            contract = self.tools.get(call.name).contract
            if (
                contract.effect.value == "read"
                and contract.parallel_safety.value == "safe_read"
            ):
                reads.append((index, call))
            else:
                effects.append((index, call))
        if reads:
            completed = await asyncio.gather(
                *(self._execute_call(turn, call) for _, call in reads)
            )
            results.update(
                {index: result for (index, _), result in zip(reads, completed)}
            )
        for index, call in effects:
            results[index] = await self._execute_call(turn, call)
        return [results[index] for index in range(len(calls))]

    async def _execute_call(self, turn: Turn, call: ToolCall) -> ToolResult:
        contract = self.tools.get(call.name).contract
        trace_input: dict[str, Any] = {"tool": call.name, "call_id": str(call.id)}
        if contract.trace_privacy == PrivacyClass.NORMAL:
            trace_input["arguments"] = call.arguments
        else:
            trace_input["arguments"] = "[redacted]"
        with trace_observation(
            f"runtime.tool.{call.name}", as_type="tool", input=trace_input
        ) as trace:
            if contract.effect.value == "read":
                result = await self.tools.execute(call)
            else:
                lease = RuntimeLease(
                    name=(
                        f"resource:{contract.resource_scope.account}:"
                        f"{contract.resource_scope.resource}"
                    ),
                    holder=f"{self.instance_id}:{turn.id}",
                    expires_at=utcnow()
                    + timedelta(
                        seconds=max(self.lease_seconds, contract.timeout_seconds + 5)
                    ),
                )
                if not await self.store.acquire_lease(lease):
                    result = ToolResult(
                        call_id=call.id,
                        outcome=ToolOutcome.NOT_STARTED,
                        error="Resource is busy; mutation was not dispatched",
                    )
                else:
                    try:
                        result = await self.tools.execute(call)
                    finally:
                        await self.store.release_lease(lease.name, lease.holder)
            if trace is not None:
                trace.update(
                    output={
                        "outcome": result.outcome.value,
                        "output": (
                            result.output
                            if contract.trace_privacy == PrivacyClass.NORMAL
                            else "[redacted]"
                        ),
                        "error": result.error,
                        "verification": result.verification,
                    }
                )
            return result

    async def _append_tool_result(
        self, turn: Turn, call: ToolCall, result: Any
    ) -> None:
        kind = (
            EventKind.TOOL_OUTCOME_UNKNOWN
            if result.outcome == ToolOutcome.OUTCOME_UNKNOWN
            else EventKind.TOOL_COMPLETED
        )
        await self.store.append_event(
            RuntimeEvent(
                thread_id=turn.thread_id,
                turn_id=turn.id,
                kind=kind,
                payload={
                    "call_id": str(call.id),
                    "tool": call.name,
                    "outcome": result.outcome.value,
                    "output": result.output,
                    "error": result.error,
                    "verification": result.verification,
                },
            )
        )

    async def _enqueue_turn_memory(self, turn: Turn) -> None:
        events = await self.store.events(turn.thread_id)
        source = next(
            (
                event
                for event in reversed(events)
                if event.kind == EventKind.INPUT_ADMITTED
                and (
                    turn.input_id == UUID(int=0)
                    or event.payload.get("input_id") == str(turn.input_id)
                )
            ),
            None,
        )
        if source is None:
            return
        await self.store.enqueue_memory_job(source.id)
        worker = self.memory_worker
        if worker is not None:
            asyncio.create_task(self._run_memory_worker_once(worker))

    async def _run_memory_worker_once(self, worker: MemoryJobWorker) -> None:
        try:
            await worker.run_once()
        except Exception:
            # The source event and failed job remain durable for operator retry.
            return

    async def _finish(
        self,
        turn: Turn,
        status: TurnStatus,
        event_kind: EventKind,
        payload: dict[str, Any] | None = None,
    ) -> Turn:
        await self.store.set_turn_status(turn.id, status)
        await self.store.append_event(
            RuntimeEvent(
                thread_id=turn.thread_id,
                turn_id=turn.id,
                kind=event_kind,
                payload=payload or {},
            )
        )
        return turn.model_copy(update={"status": status})
