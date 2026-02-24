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


class RobustLLMToolSelectorMiddleware(LLMToolSelectorMiddleware):
    def _process_selection_response(
        self,
        response: dict[str, Any],
        available_tools: List[Any],
        valid_tool_names: List[str],
        request: Any,
    ) -> Any:
        cleaned_tools = []
        for tool_name in response.get("tools", []):
            # Handle cases where the LLM might append arguments (e.g. "tool_name(...)")
            if "(" in tool_name:
                tool_name = tool_name.split("(")[0].strip()

            if tool_name in valid_tool_names:
                cleaned_tools.append(tool_name)
            else:
                logger.warning(
                    "Model selected invalid tool, ignoring it: %s", tool_name
                )

        # Update response and delegate to parent
        response["tools"] = cleaned_tools
        return super()._process_selection_response(
            response, available_tools, valid_tool_names, request
        )


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
        RobustLLMToolSelectorMiddleware(
            model=llm, always_include=ALWAYS_INCLUDE_TOOL_NAMES
        )
    )

    return middleware_stack
