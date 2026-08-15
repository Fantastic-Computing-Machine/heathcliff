# ABOUTME: Shared result handling for the simple domain-agent wrappers.

from typing import Any

from logger import logger


def run_agent(agent: Any, request: str, name: str, failure: str) -> str:
    """Invoke a LangGraph agent and return its final text or a tool fallback."""
    try:
        logger.info("[%s] %s", name, request[:80])
        messages = agent.invoke(
            {"messages": [{"role": "user", "content": request}]}
        ).get("messages", [])
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
