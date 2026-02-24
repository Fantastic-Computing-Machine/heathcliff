# ABOUTME: Info / research sub-agent — web search, weather, news, Wikipedia
# ABOUTME: Wraps tools/info_tools.py; exposed to supervisor as a single @tool

from datetime import datetime
from typing import Any

import pytz
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool

from config import Config
from core.subagents.info.tools import get_info_tools
from logger import logger

_SYSTEM_PROMPT = """\
You are a research and information retrieval specialist. The current date is: {current_date}.

<task>
Use available tools to gather accurate, up-to-date information and synthesise it into clear, structured responses.
</task>

<tools>
- search_web: Find relevant sources. Use full entity names in queries (e.g. "Advanced Micro Devices" over "AMD").
- read_website: Extract full page content. Use this for detailed reports instead of relying on search snippets alone.
- get_weather: Current weather conditions for a location.
- get_news: Latest headlines on a topic.
- wikipedia_search: Encyclopedia summaries.
- yahoo_finance_news: Market and company headlines.
- youtube_search: Discover videos on a topic.
</tools>

<workflow>
1. Identify the core topic and any formatting or length constraints.
2. Run search_web, then read_website on the most promising sources for in-depth requests.
3. Synthesise findings into a structured Markdown response that meets all user constraints exactly (word counts, paragraph counts, etc.).
</workflow>

<output_rules>
- Return a clear Markdown summary or report directly — no preamble, no reasoning section.
- Honour explicit length or structure constraints as hard requirements.
- Cite sources when the request involves research or factual claims.
</output_rules>
"""

_agent = None


def _build() -> Any:
    try:
        tz = pytz.timezone(Config.TZ)
        now_str = datetime.now(tz).strftime("%A, %B %d, %Y")

        return create_agent(
            model=init_chat_model(
                api_key=Config.AI_KEY,
                model=Config.TOOL_MODEL,
                temperature=0.4,
                timeout=Config.TIMEOUT_SECONDS,
                max_retries=Config.MAX_RETRIES,
            ),
            tools=get_info_tools(),
            system_prompt=_SYSTEM_PROMPT.format(current_date=now_str),
            name="Expert Research Information Agent",
        )
    except Exception as exc:
        logger.warning(f"[info_agent] build failed: {exc}")
        return None


@tool(
    description=(
        "Use for: weather, news, web search, Wikipedia, Yahoo Finance, YouTube, "
        "and reading URLs.\n"
        "Provide: A full natural-language research request with complete context.\n"
        "Returns: A Markdown summary or direct answer.\n"
        'Example: info_agent_tool(request="What is the current weather in Jersey City, NJ?")\n'
        'Example: info_agent_tool(request="Search for rising sea level projections 2025 '
        'and read the top articles")'
    ),
)
def info_agent_tool(request: str = "", query: str = "") -> str:
    """Research, web search, weather, news, Wikipedia lookups, and web scraping."""
    # Accept both `request` and `query` — models sometimes use the wrong param name
    effective_request = (request or query or "").strip()
    if not effective_request:
        return "Please provide a research request."
    global _agent
    if _agent is None:
        _agent = _build()
    if _agent is None:
        return "Research agent is currently unavailable."
    try:
        logger.info(f"[info_agent] {effective_request[:80]}")
        result = _agent.invoke(
            {"messages": [{"role": "user", "content": effective_request}]}
        )

        messages = result.get("messages", [])
        if not messages:
            return "No response generated."

        last_msg = messages[-1]
        content = last_msg.content
        if isinstance(content, list):
            resp = "".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        else:
            resp = str(content) if content else ""

        resp = resp.strip()

        # Fallback: if AI yielded empty string, use the last tool's output
        if not resp:
            for msg in reversed(messages):
                if getattr(msg, "type", "") == "tool":
                    resp = str(msg.content)
                    break
            if not resp:
                resp = "Action completed, but no text response was generated."

        return resp
    except Exception as exc:
        logger.error(f"[info_agent] error: {exc}", exc_info=True)
        return f"Research failed: {exc}"
