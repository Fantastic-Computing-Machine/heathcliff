# ABOUTME: Middleware configuration for agent execution control
# ABOUTME: Includes moderation, tool selection, rate limiting, and summarization

from typing import Any, List, Optional

from langchain.agents.middleware import (
    LLMToolSelectorMiddleware,
    ToolCallLimitMiddleware,
    TodoListMiddleware,
)

from logger import logger

ALWAYS_INCLUDE_TOOL_NAMES = ["recent_context"]


def create_middleware_stack(
    llm: Optional[Any] = None,
) -> List[Any]:
    """
    Create middleware stack for agent execution.

    Middleware order matters - they execute in the order they're added:
    1. Tool selection (filter tools before model call)
    2. Model call limits (prevent excessive LLM calls)
    3. Tool call limits (prevent excessive tool calls)
    4. Tool retry (handle transient failures)
    5. Summarization (manage conversation history)

    Args:
        config: Configuration object with middleware settings
        llm: Optional LLM instance for tool selection and summarization

    Returns:
        List of middleware instances
    """
    if not llm:
        raise ValueError("LLM instance is required for middleware.")

    middleware_stack = []

    middleware_stack.append(ToolCallLimitMiddleware(thread_limit=20, run_limit=10))
    middleware_stack.append(TodoListMiddleware())
    middleware_stack.append(
        LLMToolSelectorMiddleware(model=llm, always_include=ALWAYS_INCLUDE_TOOL_NAMES)
    )

    return middleware_stack
