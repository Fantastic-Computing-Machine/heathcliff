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

# Map commonly hallucinated tool names to actual supervisor tool names.
# When the LLM emits a tool name on the left, silently rewrite it to the right.
TOOL_NAME_ALIASES: dict[str, str] = {
    "load_skill_tool": "load_skill",
    "skill_loader_tool": "load_skill",
    "research_agent_tool": "info_agent_tool",
    "search_agent_tool": "info_agent_tool",
    "weather_tool": "info_agent_tool",
    "get_weather": "info_agent_tool",
    "search_web": "info_agent_tool",
    "wikipedia_search": "info_agent_tool",
    "play_track": "music_agent_tool",
    "pause_playback": "music_agent_tool",
    "current_track": "music_agent_tool",
    "send_email": "email_agent_tool",
    "read_emails": "email_agent_tool",
}


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

            # Rewrite known aliases to actual tool names
            if tool_name in TOOL_NAME_ALIASES:
                canonical = TOOL_NAME_ALIASES[tool_name]
                logger.debug(
                    "Rewriting hallucinated tool name %r → %r", tool_name, canonical
                )
                tool_name = canonical

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
