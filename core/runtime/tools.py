"""Typed tool registry and deterministic execution policy."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from core.runtime.contracts import (
    ApprovalPolicy,
    ParallelSafety,
    ToolCall,
    ToolContract,
    ToolEffect,
    ToolOutcome,
    ToolResult,
)

ToolFunction = Callable[[dict[str, Any]], Awaitable[Any]]


@dataclass(frozen=True)
class RegisteredTool:
    contract: ToolContract
    execute: ToolFunction


class ToolRegistry:
    """Static, collision-free registry of trusted integration handlers."""

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}
        self._resource_locks: dict[str, asyncio.Lock] = {}

    def register(self, contract: ToolContract, execute: ToolFunction) -> None:
        if contract.name in self._tools:
            raise ValueError(f"Tool already registered: {contract.name}")
        self._tools[contract.name] = RegisteredTool(contract, execute)

    def get(self, name: str) -> RegisteredTool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ValueError(f"Unknown tool: {name}") from exc

    def declarations(self) -> list[dict[str, Any]]:
        return [
            {
                "name": tool.contract.name,
                "description": tool.contract.description,
                "input_schema": tool.contract.input_schema,
                "schema_revision": tool.contract.schema_revision,
            }
            for tool in self._tools.values()
        ]

    def requires_approval(self, call: ToolCall) -> bool:
        contract = self.get(call.name).contract
        return contract.approval_policy == ApprovalPolicy.ALWAYS or (
            contract.approval_policy == ApprovalPolicy.EXTERNAL_SIDE_EFFECTS
            and contract.effect == ToolEffect.EXTERNAL_SIDE_EFFECT
        )

    async def execute(self, call: ToolCall) -> ToolResult:
        registered = self.get(call.name)
        contract = registered.contract
        lock_key = (
            f"{contract.resource_scope.account}:{contract.resource_scope.resource}"
        )
        lock = self._resource_locks.setdefault(lock_key, asyncio.Lock())
        try:
            if contract.parallel_safety == ParallelSafety.EXCLUSIVE:
                async with lock:
                    output = await asyncio.wait_for(
                        registered.execute(call.arguments), contract.timeout_seconds
                    )
            else:
                output = await asyncio.wait_for(
                    registered.execute(call.arguments), contract.timeout_seconds
                )
            return ToolResult(
                call_id=call.id, outcome=ToolOutcome.SUCCEEDED, output=output
            )
        except asyncio.TimeoutError:
            return ToolResult(
                call_id=call.id,
                outcome=ToolOutcome.OUTCOME_UNKNOWN
                if contract.effect != ToolEffect.READ
                else ToolOutcome.FAILED,
                error=f"Tool timed out after {contract.timeout_seconds} seconds",
            )
        except Exception as exc:
            return ToolResult(
                call_id=call.id, outcome=ToolOutcome.FAILED, error=str(exc)
            )

    async def execute_ready(self, calls: list[ToolCall]) -> list[ToolResult]:
        """Run safe reads concurrently and effects serially, returning call order."""
        results: dict[int, ToolResult] = {}
        reads: list[tuple[int, ToolCall]] = []
        effects: list[tuple[int, ToolCall]] = []
        for index, call in enumerate(calls):
            contract = self.get(call.name).contract
            (
                reads
                if (
                    contract.effect == ToolEffect.READ
                    and contract.parallel_safety == ParallelSafety.SAFE_READ
                )
                else effects
            ).append((index, call))
        if reads:
            read_results = await asyncio.gather(
                *(self.execute(call) for _, call in reads)
            )
            results.update(
                {index: result for (index, _), result in zip(reads, read_results)}
            )
        for index, call in effects:
            results[index] = await self.execute(call)
        return [results[index] for index in range(len(calls))]
