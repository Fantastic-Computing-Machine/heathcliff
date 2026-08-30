from contextvars import ContextVar

from core.coordinator_graph import _execute_with_timeout
from core.subagents._runner import (
    capture_agent_invocations,
    record_tool_invocation,
    run_agent,
    use_agent_callbacks,
)


class _Message:
    def __init__(self, message_type, content="", tool_calls=None, name=None):
        self.type = message_type
        self.content = content
        self.tool_calls = tool_calls or []
        self.name = name
        self.tool_call_id = "call-1" if message_type == "tool" else None


class _Agent:
    def invoke(self, _input):
        return {
            "messages": [
                _Message(
                    "ai",
                    tool_calls=[{"name": "lookup", "args": {"q": "x"}, "id": "call-1"}],
                ),
                _Message("tool", "found", name="lookup"),
                _Message("ai", "Complete"),
            ]
        }


def test_capture_agent_invocations_records_tool_calls_and_results():
    with capture_agent_invocations() as trace:
        assert run_agent(_Agent(), "find x", "test_agent", "failed") == "Complete"

    assert trace == [
        {
            "agent": "test_agent",
            "request": "find x",
            "tool_trace": [
                {
                    "type": "tool_call",
                    "name": "lookup",
                    "args": {"q": "x"},
                    "id": "call-1",
                },
                {
                    "type": "tool_result",
                    "name": "lookup",
                    "tool_call_id": "call-1",
                    "content": "found",
                },
            ],
        }
    ]


def test_run_agent_forwards_request_callbacks_to_the_nested_agent():
    class CallbackAwareAgent(_Agent):
        def invoke(self, _input, config=None):
            self.config = config
            return super().invoke(_input)

    callback = object()
    agent = CallbackAwareAgent()
    with use_agent_callbacks([callback]):
        assert run_agent(agent, "find x", "test_agent", "failed") == "Complete"

    assert agent.config == {"callbacks": [callback]}


def test_programmatic_tool_calls_are_recorded_for_research_history():
    with capture_agent_invocations() as trace:
        record_tool_invocation("info_agent", "tavily_search", {"query": "x"}, {"ok": True})

    assert trace[0]["agent"] == "info_agent"
    assert [event["type"] for event in trace[0]["tool_trace"]] == [
        "tool_call",
        "tool_result",
    ]
    assert trace[0]["tool_trace"][0]["name"] == "tavily_search"


def test_timeout_worker_preserves_diagnostic_context():
    marker = ContextVar("marker", default="missing")
    marker.set("present")

    assert _execute_with_timeout(lambda: marker.get(), timeout_ms=100) == "present"
