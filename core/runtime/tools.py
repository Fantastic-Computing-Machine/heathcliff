"""Typed tool registry and deterministic execution policy."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from jsonschema import Draft202012Validator, ValidationError
from referencing import Registry

from core.runtime.contracts import (
    ApprovalPolicy,
    ParallelSafety,
    ToolCall,
    ToolContract,
    ToolEffect,
    ToolOutcome,
    ToolResult,
)

ToolFunction = Callable[[dict[str, Any]], Any | Awaitable[Any]]


@dataclass(frozen=True)
class RegisteredTool:
    contract: ToolContract
    execute: ToolFunction
    verification_arguments: dict[str, Any] = field(default_factory=dict)


async def _settle(task: asyncio.Task) -> None:
    """Wait through repeated caller cancellation; never abandon a worker thread."""
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
        except Exception:
            break
    if not task.cancelled():
        task.exception()  # Retrieve errors even when the caller timed out.


def _validate(schema: dict[str, Any], value: Any) -> None:
    # Empty referencing registries resolve local refs but never fetch remote URLs.
    json.dumps(value, allow_nan=False)
    Draft202012Validator(schema, registry=Registry()).validate(value)


class ToolRegistry:
    """Trusted handlers with shared validation, scheduling and result semantics."""

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}
        self._read_slots = asyncio.Semaphore(4)
        self._condition = asyncio.Condition()
        self._readers: dict[tuple[str, str], int] = {}
        self._exclusive: tuple[str, str] | None = None
        self._active: set[asyncio.Task] = set()
        self._generations: dict[str, int] = {}
        self._owners: dict[str, str] = {}

    def register(
        self, contract: ToolContract, execute: ToolFunction, *,
        verification_arguments: dict[str, Any] | None = None,
    ) -> None:
        if contract.name in self._tools:
            raise ValueError(f"Tool already registered: {contract.name}")
        Draft202012Validator.check_schema(contract.input_schema)
        Draft202012Validator.check_schema(contract.output_schema)
        self._tools[contract.name] = RegisteredTool(
            contract.model_copy(deep=True), execute, deepcopy(verification_arguments or {})
        )

    def get(self, name: str) -> RegisteredTool:
        try:
            tool = self._tools[name]
            return RegisteredTool(tool.contract.model_copy(deep=True), tool.execute,
                                  deepcopy(tool.verification_arguments))
        except KeyError as exc:
            raise ValueError(f"Unknown tool: {name}") from exc

    def declarations(self, names: list[str] | None = None) -> list[dict[str, Any]]:
        """Load schemas for exact names; no-argument legacy API remains available."""
        if names is not None and (isinstance(names, str) or len(names) > 50):
            raise ValueError("Select at most 50 explicit tool names")
        selected = self._tools.values() if names is None else [self.get(n) for n in names]
        return [
            {
                "name": tool.contract.name,
                "description": tool.contract.description,
                "input_schema": deepcopy(tool.contract.input_schema),
                "schema_revision": tool.contract.schema_revision,
            }
            for tool in selected
        ]

    def catalog(self, offset: int = 0, limit: int = 50) -> dict[str, Any]:
        """Bounded capability descriptions; schemas require explicit name selection."""
        if offset < 0 or not 1 <= limit <= 50:
            raise ValueError("Catalog offset must be nonnegative and limit between 1 and 50")
        names = sorted(self._tools)
        return {
            "tools": [{"name": n, "description": self._tools[n].contract.description[:512],
                       "effect": self._tools[n].contract.effect.value,
                       "schema_revision": self._tools[n].contract.schema_revision}
                      for n in names[offset:offset + limit]],
            "next_offset": offset + limit if offset + limit < len(names) else None,
        }

    def replace_catalog(self, owner: str, generation: int, tools: list[RegisteredTool]) -> bool:
        """Atomically replace one source; late refreshes cannot resurrect old tools."""
        if generation <= self._generations.get(owner, -1):
            return False
        staged = ToolRegistry()
        for tool in tools:
            name = tool.contract.name
            if name in self._tools and self._owners.get(name) != owner:
                raise ValueError(f"Tool already registered: {name}")
            staged.register(tool.contract, tool.execute,
                            verification_arguments=tool.verification_arguments)
        self._tools = {n: t for n, t in self._tools.items() if self._owners.get(n) != owner} | staged._tools
        self._owners = {n: o for n, o in self._owners.items() if o != owner} | dict.fromkeys(staged._tools, owner)
        self._generations[owner] = generation
        return True

    def requires_approval(self, call: ToolCall) -> bool:
        contract = self.get(call.name).contract
        return contract.approval_policy == ApprovalPolicy.ALWAYS or (
            contract.approval_policy == ApprovalPolicy.EXTERNAL_SIDE_EFFECTS
            and contract.effect == ToolEffect.EXTERNAL_SIDE_EFFECT
        )

    async def execute(self, call: ToolCall) -> ToolResult:
        call = call.model_copy(deep=True)
        invalid = self.validate_call(call)
        if invalid is not None:
            return invalid
        registered = self.get(call.name)
        verifier = self._verifier(registered)
        async with self._slot(registered.contract):
            work = asyncio.create_task(self._execute_verified(registered, call, verifier))
            self._active.add(work)
            try:
                return await asyncio.shield(work)
            except asyncio.CancelledError:
                await _settle(work)
                raise
            finally:
                self._active.discard(work)

    def validate_call(self, call: ToolCall) -> ToolResult | None:
        """Preflight for controller approval/lease admission; execution repeats it."""
        try:
            registered = self.get(call.name)
            _validate(registered.contract.input_schema, call.arguments)
            self._verifier(registered)
        except Exception:
            return ToolResult(call_id=call.id, outcome=ToolOutcome.NOT_STARTED,
                              error="Unknown tool, invalid arguments, or invalid verification policy")
        return None

    def _verifier(self, tool: RegisteredTool) -> RegisteredTool | None:
        contract = tool.contract
        required = getattr(contract, "verification_policy", "none") == "required"
        if not contract.verification_tool:
            if required:
                raise ValueError("Required verification tool is missing")
            return None
        verifier = self.get(contract.verification_tool)
        if (verifier.contract.effect != ToolEffect.READ
            or verifier.contract.resource_scope.account != contract.resource_scope.account
            or verifier.contract.resource_scope.resource != contract.resource_scope.resource
            or verifier.contract.verification_tool
            or verifier.contract.approval_policy == ApprovalPolicy.ALWAYS):
            raise ValueError("Verification must be an approval-free read of the same resource")
        _validate(verifier.contract.input_schema, self._verification_arguments(tool))
        return verifier

    @staticmethod
    def _verification_arguments(tool: RegisteredTool) -> dict[str, Any]:
        return deepcopy(tool.verification_arguments or getattr(tool.contract, "verification_arguments", {}))

    @staticmethod
    def _failure(contract: ToolContract, call: ToolCall, error: str) -> ToolResult:
        return ToolResult(call_id=call.id,
                          outcome=ToolOutcome.FAILED if contract.effect == ToolEffect.READ else ToolOutcome.OUTCOME_UNKNOWN,
                          error=error)

    async def _invoke(self, tool: RegisteredTool, call: ToolCall) -> ToolResult:
        async def invoke():
            if inspect.iscoroutinefunction(tool.execute):
                return await tool.execute(call.arguments)
            result = await asyncio.to_thread(tool.execute, call.arguments)
            return await result if inspect.isawaitable(result) else result

        work = asyncio.create_task(invoke())
        try:
            output = await asyncio.wait_for(asyncio.shield(work), tool.contract.timeout_seconds)
            result = (output.model_copy(update={"call_id": call.id}) if isinstance(output, ToolResult)
                      else ToolResult(call_id=call.id, outcome=ToolOutcome.SUCCEEDED, output=output))
            if result.outcome == ToolOutcome.SUCCEEDED:
                if result.error is not None:
                    return self._failure(tool.contract, call, "Tool returned contradictory success/error metadata")
                _validate(tool.contract.output_schema, result.output)
            return result
        except asyncio.TimeoutError:
            await _settle(work)
            return self._failure(tool.contract, call, f"Tool timed out after {tool.contract.timeout_seconds} seconds")
        except asyncio.CancelledError:
            await _settle(work)
            raise
        except ValidationError:
            return self._failure(tool.contract, call, "Tool output does not match its declared schema")
        except Exception as exc:
            return self._failure(tool.contract, call, f"Tool execution raised {type(exc).__name__}")

    async def _execute_verified(self, tool: RegisteredTool, call: ToolCall,
                                verifier: RegisteredTool | None) -> ToolResult:
        result = await self._invoke(tool, call)
        if result.outcome != ToolOutcome.SUCCEEDED or verifier is None:
            return result
        check = await self._invoke(verifier, ToolCall(name=verifier.contract.name,
                                  arguments=self._verification_arguments(tool)))
        result.verification = check.model_dump(mode="json")
        if not (check.outcome == ToolOutcome.SUCCEEDED and isinstance(check.output, dict)
                and check.output.get("verified") is True):
            result.outcome = self._failure(tool.contract, call, "").outcome
            result.error = "Tool result could not be verified"
        return result

    @asynccontextmanager
    async def _slot(self, contract: ToolContract):
        scope = (contract.resource_scope.account, contract.resource_scope.resource)
        safe_read = contract.effect == ToolEffect.READ and contract.parallel_safety == ParallelSafety.SAFE_READ
        if safe_read:
            await self._read_slots.acquire()
        acquired = False
        try:
            async with self._condition:
                await self._condition.wait_for(lambda: (
                    self._exclusive != scope if safe_read
                    else self._exclusive is None and not self._readers.get(scope)
                ))
                if safe_read:
                    self._readers[scope] = self._readers.get(scope, 0) + 1
                else:
                    self._exclusive = scope
                acquired = True
            yield
        finally:
            if acquired:
                async with self._condition:
                    if safe_read:
                        self._readers[scope] -= 1
                        if not self._readers[scope]:
                            del self._readers[scope]
                    else:
                        self._exclusive = None
                    self._condition.notify_all()
            if safe_read:
                self._read_slots.release()

    async def drain(self) -> None:
        """Wait for dispatched work before shutting down provider resources."""
        while self._active:
            await asyncio.gather(*(_settle(work) for work in tuple(self._active)))

    async def execute_ready(self, calls: list[ToolCall]) -> list[ToolResult]:
        """Run safe reads concurrently and effects serially, returning call order."""
        results: dict[int, ToolResult] = {}
        reads: list[tuple[int, ToolCall]] = []
        effects: list[tuple[int, ToolCall]] = []
        for index, call in enumerate(calls):
            invalid = self.validate_call(call)
            if invalid is not None:
                results[index] = invalid
                continue
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
