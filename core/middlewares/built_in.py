# ABOUTME: Middleware configuration for agent execution control
# ABOUTME: Includes tool selection (structured output) and summarization

from typing import Any, List

from langchain.agents.middleware import (
    AgentMiddleware,
    SummarizationMiddleware,
)
from langchain_google_genai import ChatGoogleGenerativeAI

from config import Config
from core.middlewares.tool_selector import StructuredToolSelectorMiddleware
from logger import logger

SUMMARY_MIDDLEWARE_LLM = Config.SUMMARY_MIDDLEWARE_LLM


class BuiltInMiddlewares:
    def __init__(self) -> None:
        """Initialize middleware components with given LLM."""

        self._llm = ChatGoogleGenerativeAI(
            model=SUMMARY_MIDDLEWARE_LLM,
            google_api_key=Config.GEMINI_API_KEY,
        )

    def _tool_selector(self) -> AgentMiddleware:
        return StructuredToolSelectorMiddleware(
            model=self._llm,
            max_tools=5,
            always_include=[
                "search_and_scrape",
                "search_web",
            ],
        )

    def _summarization(self) -> AgentMiddleware:
        return SummarizationMiddleware(
            model=self._llm,
            trigger=("tokens", 4000),
            keep=("messages", 20),
        )

    def get(self) -> List[Any]:
        """Return list of middleware instances."""

        middlewares = []

        middlewares.append(self._tool_selector())
        middlewares.append(self._summarization())

        logger.info(f"Initialized {len(middlewares)} built-in middlewares.")

        return middlewares
