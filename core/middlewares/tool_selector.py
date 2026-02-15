# ABOUTME: Custom tool selector middleware using structured outputs with enums
# ABOUTME: Prevents tool name hallucination by constraining LLM to valid enum values

from __future__ import annotations

from enum import Enum
from typing import Any, Callable, List, Optional

from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain.agents.middleware.types import ModelCallResult
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from logger import logger

_SYSTEM_PROMPT = (
    "You are a tool selection assistant. Given the user's query "
    "and the list of available tools, select the most relevant tools. "
    "You MUST select from the provided enum values ONLY. "
    "Rank by relevance — most relevant first."
)


def _build_tool_enum(tool_names: List[str]) -> type:
    """Dynamically create an Enum class from a list of tool names.

    Args:
        tool_names: List of valid tool name strings.

    Returns:
        An Enum class whose members correspond to tool names.
    """
    return Enum("ToolName", {name: name for name in tool_names})


def _build_selection_schema(tool_enum: type) -> type:
    """Build a Pydantic model that constrains tool selection to enum values.

    Args:
        tool_enum: An Enum class of valid tool names.

    Returns:
        A Pydantic BaseModel class for structured output.
    """

    class ToolSelection(BaseModel):
        """Selected tools for the user's query."""

        tools: List[tool_enum] = Field(  # type: ignore[valid-type]
            description="List of selected tool names, ordered by relevance."
        )

    return ToolSelection


class StructuredToolSelectorMiddleware(AgentMiddleware):
    """Middleware that uses structured output with enums to select tools.

    Dynamically creates an ``Enum`` from registered tool names and uses
    ``with_structured_output`` to force the selector LLM to pick only
    valid values. This prevents tool-name hallucination entirely.

    Falls back gracefully to passing all tools if selection fails.

    Args:
        model: LLM to use for tool selection.
        max_tools: Maximum number of tools to select per turn.
        always_include: Tool names that are always included regardless of
            selection (do not count against ``max_tools``).
    """

    def __init__(
        self,
        *,
        model: BaseChatModel,
        max_tools: int = 5,
        always_include: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self._model = model
        self._max_tools = max_tools
        self._always_include = set(always_include or [])

    def _select_tools(self, request: ModelRequest) -> List[Any]:
        """Use structured output to select the most relevant tools.

        Args:
            request: The model request containing tools and messages.

        Returns:
            Filtered list of tools to pass to the agent.
        """
        base_tools = [t for t in request.tools if not isinstance(t, dict)]
        provider_tools = [t for t in request.tools if isinstance(t, dict)]

        if not base_tools:
            return request.tools

        # Separate always-included tools from selectable tools
        always_tools = [t for t in base_tools if t.name in self._always_include]
        selectable_tools = [t for t in base_tools if t.name not in self._always_include]

        if not selectable_tools:
            return request.tools

        # Build the enum and schema from selectable tool names
        tool_names = [t.name for t in selectable_tools]
        tool_descriptions = {t.name: t.description[:120] for t in selectable_tools}
        tool_enum = _build_tool_enum(tool_names)
        selection_schema = _build_selection_schema(tool_enum)

        # Compose the prompt with tool descriptions
        tool_list = "\n".join(
            f"- {name}: {desc}" for name, desc in tool_descriptions.items()
        )
        system_msg = (
            f"{_SYSTEM_PROMPT}\n\n"
            f"Available tools (select up to {self._max_tools}):\n{tool_list}"
        )

        # Extract the last user message
        last_user_msg = None
        for msg in reversed(request.messages):
            if isinstance(msg, HumanMessage):
                last_user_msg = msg
                break

        if last_user_msg is None:
            logger.warning("No user message found — passing all tools.")
            return request.tools

        # Call the selector LLM with structured output
        structured_model = self._model.with_structured_output(selection_schema)
        response = structured_model.invoke(
            [
                {"role": "system", "content": system_msg},
                last_user_msg,
            ]
        )

        # Extract selected names from the enum values
        selected_names: List[str] = []
        for tool_val in response.tools:
            name = tool_val.value if hasattr(tool_val, "value") else str(tool_val)
            if name not in selected_names:
                selected_names.append(name)
            if len(selected_names) >= self._max_tools:
                break

        logger.info(
            "Tool selector chose: %s (always included: %s)",
            selected_names,
            [t.name for t in always_tools],
        )

        # Build the final tool list
        selected = [t for t in selectable_tools if t.name in selected_names]
        return [*selected, *always_tools, *provider_tools]

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Filter tools via structured-output selection before invoking the model.

        Args:
            request: Model request with all registered tools.
            handler: Callback to invoke the model.

        Returns:
            The model call result with filtered tools.
        """
        try:
            filtered_tools = self._select_tools(request)
            return handler(request.override(tools=filtered_tools))
        except Exception as exc:
            logger.warning(
                "Tool selection failed (%s) — falling back to all tools.", exc
            )
            return handler(request)
