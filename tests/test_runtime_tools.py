"""Tool boundary checks; all integrations are local test functions."""

import asyncio
import threading
from uuid import uuid4

import pytest

from core.runtime.contracts import (
    ApprovalPolicy,
    ParallelSafety,
    ResourceScope,
    ToolCall,
    ToolContract,
    ToolEffect,
    ToolOutcome,
    ToolResult,
)
from core.runtime.tools import ToolRegistry


def contract(name="read", **kwargs):
    return ToolContract(
        name=name,
        description="Read a test value",
        input_schema=kwargs.pop("input_schema", {"type": "object"}),
        resource_scope=kwargs.pop("resource_scope", ResourceScope(resource="test")),
        approval_policy=kwargs.pop("approval_policy", ApprovalPolicy.NEVER),
        **kwargs,
    )


@pytest.mark.parametrize("arguments", [{}, {"n": "1"}, {"n": True}, {"n": 0}, {"n": 1, "x": 2}])
def test_invalid_arguments_never_dispatch(arguments):
    dispatched = []
    tools = ToolRegistry()

    async def execute(args):
        dispatched.append(args)

    tools.register(contract(input_schema={
        "type": "object", "properties": {"n": {"type": "integer", "minimum": 1}},
        "required": ["n"], "additionalProperties": False,
    }), execute)
    result = asyncio.run(tools.execute(ToolCall(name="read", arguments=arguments)))
    assert result.outcome == ToolOutcome.NOT_STARTED
    assert not dispatched


def test_unknown_call_is_not_started_and_does_not_abort_batch():
    tools = ToolRegistry()
    async def read(args):
        return 7
    tools.register(contract(), read)
    results = asyncio.run(tools.execute_ready([ToolCall(name="missing"), ToolCall(name="read")]))
    assert [r.outcome for r in results] == [ToolOutcome.NOT_STARTED, ToolOutcome.SUCCEEDED]


def test_typed_failure_is_preserved_and_call_identity_is_bound():
    tools = ToolRegistry()
    async def read(args):
        return ToolResult(call_id=uuid4(), outcome=ToolOutcome.FAILED, error="typed failure")
    tools.register(contract(), read)
    call = ToolCall(name="read")
    result = asyncio.run(tools.execute(call))
    assert result.outcome == ToolOutcome.FAILED
    assert result.call_id == call.id
    assert result.error == "typed failure"


@pytest.mark.parametrize("effect,expected", [(ToolEffect.READ, ToolOutcome.FAILED), (ToolEffect.EXTERNAL_SIDE_EFFECT, ToolOutcome.OUTCOME_UNKNOWN)])
def test_invalid_output_cannot_claim_success(effect, expected):
    tools = ToolRegistry()
    async def read(args):
        return {"count": "wrong"}
    tools.register(contract(effect=effect, output_schema={"type": "object", "properties": {"count": {"type": "integer"}}}), read)
    assert asyncio.run(tools.execute(ToolCall(name="read"))).outcome == expected


def test_plain_data_is_not_classified_by_error_words():
    tools = ToolRegistry()
    async def read(args):
        return {"error": "a documented field", "text": "failed examples"}
    tools.register(contract(), read)
    assert asyncio.run(tools.execute(ToolCall(name="read"))).outcome == ToolOutcome.SUCCEEDED


def test_sync_python_handlers_use_same_contract():
    tools = ToolRegistry()
    tools.register(contract(output_schema={"type": "integer"}), lambda args: 8)
    assert asyncio.run(tools.execute(ToolCall(name="read"))).output == 8


def test_safe_reads_are_bounded_across_direct_calls():
    async def run():
        tools = ToolRegistry()
        active = maximum = 0
        async def read(args):
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            await asyncio.sleep(0.01)
            active -= 1
            return args["n"]
        tools.register(contract(), read)
        results = await asyncio.gather(*(tools.execute(ToolCall(name="read", arguments={"n": n})) for n in range(12)))
        assert 1 < maximum <= 4
        assert [r.output for r in results] == list(range(12))
    asyncio.run(run())


def test_writes_serialize_across_resources_and_direct_calls():
    async def run():
        tools = ToolRegistry()
        active = maximum = 0
        async def write(args):
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            await asyncio.sleep(0.01)
            active -= 1
        for name in ("one", "two"):
            tools.register(contract(name, effect=ToolEffect.REVERSIBLE_WRITE, resource_scope=ResourceScope(resource=name)), write)
        await asyncio.gather(tools.execute(ToolCall(name="one")), tools.execute(ToolCall(name="two")))
        assert maximum == 1
    asyncio.run(run())


@pytest.mark.parametrize("cancel", [False, True])
def test_thread_work_keeps_locks_and_caller_pending_until_it_finishes(cancel):
    async def run():
        tools = ToolRegistry()
        started, release = threading.Event(), threading.Event()
        dispatched = []
        def block(args):
            started.set()
            assert release.wait(5)
        async def legacy(args):
            await asyncio.to_thread(block, args)
        async def next_write(args):
            dispatched.append(True)
        tools.register(contract("block", effect=ToolEffect.EXTERNAL_SIDE_EFFECT, parallel_safety=ParallelSafety.EXCLUSIVE, timeout_seconds=1), legacy)
        tools.register(contract("next", effect=ToolEffect.EXTERNAL_SIDE_EFFECT), next_write)
        first = asyncio.create_task(tools.execute(ToolCall(name="block")))
        try:
            assert await asyncio.to_thread(started.wait, 2)
            if cancel:
                first.cancel()
                await asyncio.sleep(0)
                first.cancel()  # Repeated cancellation must not abandon the worker.
            else:
                await asyncio.sleep(1.1)
            second = asyncio.create_task(tools.execute(ToolCall(name="next")))
            await asyncio.sleep(0.02)
            assert not dispatched
            assert not first.done()
        finally:
            release.set()
            completed = await asyncio.gather(first, return_exceptions=True)
        if cancel:
            assert isinstance(completed[0], asyncio.CancelledError)
        else:
            assert completed[0].outcome == ToolOutcome.OUTCOME_UNKNOWN
        assert (await second).outcome == ToolOutcome.SUCCEEDED
    asyncio.run(run())


def test_read_cannot_overlap_write_in_same_scope():
    async def run():
        tools = ToolRegistry()
        started, release = asyncio.Event(), asyncio.Event()
        reads = []
        async def write(args):
            started.set()
            await release.wait()
        async def read(args):
            reads.append(True)
        tools.register(contract("write", effect=ToolEffect.REVERSIBLE_WRITE), write)
        tools.register(contract(), read)
        first = asyncio.create_task(tools.execute(ToolCall(name="write")))
        await started.wait()
        second = asyncio.create_task(tools.execute(ToolCall(name="read")))
        await asyncio.sleep(0.02)
        try:
            assert not reads
        finally:
            release.set()
            await asyncio.gather(first, second)
    asyncio.run(run())


def test_catalog_is_bounded_and_schema_loading_requires_exact_names():
    tools = ToolRegistry()
    for n in range(55):
        tools.register(contract(f"read_{n:02}"), lambda args: None)
    page = tools.catalog(limit=3)
    assert len(page["tools"]) == 3
    assert page["next_offset"] == 3
    assert "input_schema" not in page["tools"][0]
    assert [x["name"] for x in tools.declarations(names=["read_32"])] == ["read_32"]
    with pytest.raises(ValueError):
        tools.declarations(names=["Read a test value"])
    with pytest.raises(ValueError):
        tools.catalog(limit=51)


def test_verification_runs_in_scope_and_cannot_claim_unverified_mutation():
    async def run():
        tools = ToolRegistry()
        observed = []
        async def write(args):
            return {"accepted": True}
        async def verify(args):
            observed.append(args)
            return {"verified": False}
        tools.register(contract("verify"), verify)
        tools.register(contract("write", effect=ToolEffect.REVERSIBLE_WRITE, verification_tool="verify"), write, verification_arguments={"id": 7})
        result = await tools.execute(ToolCall(name="write"))
        assert result.outcome == ToolOutcome.OUTCOME_UNKNOWN
        assert result.verification is not None
        assert observed == [{"id": 7}]
    asyncio.run(run())


def test_unsafe_verifier_prevents_mutation():
    tools = ToolRegistry()
    dispatched = []
    tools.register(contract("verify", effect=ToolEffect.REVERSIBLE_WRITE), lambda args: True)
    tools.register(contract("write", effect=ToolEffect.REVERSIBLE_WRITE, verification_tool="verify"), lambda args: dispatched.append(True))
    assert asyncio.run(tools.execute(ToolCall(name="write"))).outcome == ToolOutcome.NOT_STARTED
    assert not dispatched
