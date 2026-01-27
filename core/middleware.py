# ABOUTME: Middleware configuration for agent execution control
# ABOUTME: Includes moderation, tool selection, rate limiting, and summarization

from typing import Any, List, Literal, Optional

from langchain.agents.middleware import (
    AgentMiddleware,
    LLMToolSelectorMiddleware,
    SummarizationMiddleware,
)

from logger import logger


class Middlewares:
    def __init__(self, llm) -> None:
        """Initialize middleware components with given LLM."""

        self._llm = llm

    def _tool_selctor(self) -> AgentMiddleware:
        return LLMToolSelectorMiddleware(
            model=self._llm,
            max_tools=3,
            always_include=["search"],
        )

    def _summarization(self) -> AgentMiddleware:
        return SummarizationMiddleware(
            model=self._llm,
            trigger=("tokens", 4000),
            keep=("messages", 20),
        )

    def get(self) -> List[Any]:
        """Return list of middleware instances."""

        return [
            self._tool_selctor,
            self._summarization,
        ]
