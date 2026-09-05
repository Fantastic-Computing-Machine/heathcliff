"""Synchronous compatibility adapter for the legacy HeathcliffAgent surface."""

from __future__ import annotations

import asyncio
from uuid import NAMESPACE_URL, UUID, uuid5

from core.runtime.contracts import ApprovalDecision, EventKind
from core.runtime.engine import HeathcliffRuntime


class RuntimeV2CompatibilityAdapter:
    def __init__(self, runtime: HeathcliffRuntime) -> None:
        self.runtime = runtime

    @staticmethod
    def _thread_id(conversation_id: str) -> UUID:
        try:
            return UUID(conversation_id)
        except ValueError:
            return uuid5(NAMESPACE_URL, f"heathcliff:{conversation_id}")

    async def ainvoke(self, user_input: str, conversation_id: str) -> str:
        thread_id = self._thread_id(conversation_id)
        pending = await self.runtime.admit_input(
            thread_id, user_input, str(uuid5(NAMESPACE_URL, user_input))
        )
        await self.runtime.run_input(pending)
        events = await self.runtime.store.events(thread_id)
        for event in reversed(events):
            if event.kind == EventKind.TURN_COMPLETED:
                return str(event.payload.get("response", ""))
            if event.kind == EventKind.TURN_FAILED:
                return "I encountered an error processing your request."
        return "I need your approval before continuing."

    def invoke(self, user_input: str, conversation_id: str) -> str:
        return asyncio.run(self.ainvoke(user_input, conversation_id))

    async def aresume_approval(self, conversation_id: str, approved: bool) -> str:
        approval = await self.runtime.store.pending_approval(
            self._thread_id(conversation_id)
        )
        if approval is None:
            raise ValueError("No pending approval for this conversation")
        await self.runtime.decide_approval(
            ApprovalDecision(approval_id=approval.id, approved=approved)
        )
        events = await self.runtime.store.events(approval.thread_id)
        return next(
            (
                str(event.payload.get("response", ""))
                for event in reversed(events)
                if event.kind == EventKind.TURN_COMPLETED
            ),
            "Action rejected." if not approved else "Action completed.",
        )

    def resume_approval(self, conversation_id: str, approved: bool) -> str:
        return asyncio.run(self.aresume_approval(conversation_id, approved))
