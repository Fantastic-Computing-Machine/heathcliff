from contextvars import ContextVar

from core.coordinator_graph import _execute_with_timeout
from core.subagents._runner import capture_agent_invocations, run_agent


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


def test_timeout_worker_preserves_diagnostic_context():
    marker = ContextVar("marker", default="missing")
    marker.set("present")

    assert _execute_with_timeout(lambda: marker.get(), timeout_ms=100) == "present"
