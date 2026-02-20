# ABOUTME: Info / research sub-agent — web search, weather, news, Wikipedia
# ABOUTME: Wraps tools/info_tools.py; exposed to supervisor as a single @tool

from typing import Any

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from config import Config
from logger import logger

_SYSTEM_PROMPT = """\
You are a specialist research and information retrieval agent.
Your job: use the tools available to you to answer questions accurately.
Always pass COMPLETE queries to tools — never truncate the topic or entity name.
Return a clear, well-structured summary. No conversational filler.
"""

_agent = None


def _build() -> Any:
    try:
        from core.subagents.info.tools import get_info_tools

        return create_agent(
            model=ChatGoogleGenerativeAI(
                model=Config.MODEL,
                google_api_key=Config.GEMINI_API_KEY,
                temperature=0.3,
                max_output_tokens=Config.MAX_TOKENS,
            ),
            tools=get_info_tools(),
            system_prompt=_SYSTEM_PROMPT,
        )
    except Exception as exc:
        logger.warning(f"[info_agent] build failed: {exc}")
        return None


@tool
def info_agent_tool(request: str) -> str:
    """Research, web search, weather, news, and Wikipedia lookups.

    Use for any information retrieval:
    - Current weather in a location
    - Latest news on a topic
    - Web search for facts or data
    - Wikipedia summaries

    Input: Full natural-language request with complete context.
    Example: "What is the current weather in Jersey City, NJ?"
    Example: "Search the web for rising sea level projections 2025"
    """
    global _agent
    if _agent is None:
        _agent = _build()
    if _agent is None:
        return "Research agent is currently unavailable."
    try:
        logger.info(f"[info_agent] {request[:80]}")
        result = _agent.invoke({"messages": [{"role": "user", "content": request}]})
        return result["messages"][-1].content
    except Exception as exc:
        logger.error(f"[info_agent] error: {exc}", exc_info=True)
        return f"Research failed: {exc}"
