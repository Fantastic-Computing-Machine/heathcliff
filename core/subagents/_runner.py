# ABOUTME: Shared result handling for the simple domain-agent wrappers.

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Generator

from logger import logger

_trace_buffer: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "subagent_trace_buffer", default=None
)


@contextmanager
def capture_agent_invocations() -> Generator[list[dict[str, Any]], None, None]:
    """Collect subagent tool calls for an opt-in diagnostic run."""
    buffer: list[dict[str, Any]] = []
    token = _trace_buffer.set(buffer)
    try:
        yield buffer
    finally:
        _trace_buffer.reset(token)


def _tool_trace(messages: list[Any]) -> list[dict[str, Any]]:
    trace: list[dict[str, Any]] = []
    for message in messages:
        calls = getattr(message, "tool_calls", None) or []
        for call in calls:
            trace.append(
                {
                    "type": "tool_call",
                    "name": call.get("name"),
                    "args": call.get("args", {}),
                    "id": call.get("id"),
                }
            )
        if getattr(message, "type", "") == "tool":
            trace.append(
                {
                    "type": "tool_result",
                    "name": getattr(message, "name", None),
                    "tool_call_id": getattr(message, "tool_call_id", None),
                    "content": str(getattr(message, "content", "")),
                }
            )
    return trace


def record_agent_invocation(name: str, request: str, messages: list[Any]) -> None:
    """Record one subagent's tool activity when diagnostic tracing is enabled."""
    trace_buffer = _trace_buffer.get()
    if trace_buffer is not None:
        trace_buffer.append(
            {"agent": name, "request": request, "tool_trace": _tool_trace(messages)}
        )


def run_agent(agent: Any, request: str, name: str, failure: str) -> str:
    """Invoke a LangGraph agent and return its final text or a tool fallback."""
    try:
        logger.info("[%s] %s", name, request[:80])
        messages = agent.invoke(
            {"messages": [{"role": "user", "content": request}]}
        ).get("messages", [])
        record_agent_invocation(name, request, messages)
        if not messages:
            return "No response generated."

        content = messages[-1].content
        response = (
            "".join(part.get("text", "") for part in content if isinstance(part, dict))
            if isinstance(content, list)
            else str(content or "")
        )
        if response.strip():
            return response.strip()

        for message in reversed(messages):
            if getattr(message, "type", "") == "tool":
                return str(message.content)
        return "Action completed, but no text response was generated."
    except Exception as exc:
        logger.error("[%s] error: %s", name, exc, exc_info=True)
        return f"{failure}: {exc}"
