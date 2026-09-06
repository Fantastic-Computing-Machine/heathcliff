"""Trust-boundary checks: malformed requests never reach integrations."""

import asyncio
from uuid import uuid4

from core.runtime.contracts import EventKind, ResourceScope, RuntimeEvent, ToolCall, ToolContract
from core.runtime.tools import ToolRegistry
from db.runtime_store import PostgresRuntimeStore


def test_invalid_arguments_are_not_dispatched():
    async def run():
        executed = []
        async def handler(args):
            executed.append(args)
            return {"ok": True}
        registry = ToolRegistry()
        registry.register(ToolContract(name="check", description="check",
            resource_scope=ResourceScope(resource="check"), input_schema={
                "type": "object", "properties": {"count": {"type": "integer"}},
                "required": ["count"], "additionalProperties": False,
            }), handler)
        result = await registry.execute(ToolCall(name="check", arguments={"count": "wrong"}))
        assert executed == []
        assert result.outcome.value == "not_started"
    asyncio.run(run())


def test_invalid_output_is_not_success():
    async def run():
        async def handler(args):
            return "not a structured result"
        registry = ToolRegistry()
        registry.register(ToolContract(name="check", description="check",
            resource_scope=ResourceScope(resource="check"), input_schema={"type": "object"},
            output_schema={"type": "object", "required": ["ok"]}), handler)
        result = await registry.execute(ToolCall(name="check"))
        assert result.outcome.value == "failed"
    asyncio.run(run())


def test_postgres_event_decodes_default_asyncpg_json_string():
    event = RuntimeEvent(thread_id=uuid4(), kind=EventKind.INPUT_ADMITTED,
                         payload={"content": "hello"}, sequence=1)
    row = event.model_dump()
    row["payload"] = '{"content":"hello"}'
    assert PostgresRuntimeStore._event_from_row(row) == event
