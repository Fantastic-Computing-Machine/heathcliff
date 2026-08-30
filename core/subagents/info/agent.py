# ABOUTME: Info / research sub-agent — web search, weather, news, Wikipedia
# ABOUTME: Wraps tools/info_tools.py; exposed to supervisor as a single @tool

from datetime import datetime
from typing import Any

import pytz
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.errors import GraphRecursionError

from config import Config
from core.runtime_profile import current_tool_model
from core.subagents._runner import agent_callbacks, record_agent_invocation
from core.subagents.info.evidence import (
    plan_request,
    run_research_workflow,
)
from core.subagents.info.recent_context import recent_context
from core.subagents.info.tools import get_info_tools
from logger import logger

_BASE_PROMPT = """\
You are a research and information retrieval specialist. The current date is: {current_date}.

<task>
Use available tools to gather accurate, up-to-date information and synthesise it into clear, structured responses.
</task>

<tools>
- search_web: Find relevant sources. Use full entity names in queries (e.g. "Advanced Micro Devices" over "AMD").
- read_website: Extract full page content. Use this for detailed reports instead of relying on search snippets alone.
- tavily_search: Independent real-time web search with source URLs and page excerpts.
- tavily_extract: Cleanly extract one or more webpages when ordinary page reading is incomplete or blocked.
- get_weather: Current weather conditions for a location.
- get_news: Latest headlines on a topic.
- wikipedia_search: Encyclopedia summaries and page snippets.
- wikidata_search: Structured entity facts (properties, aliases, IDs).
- stackexchange_search: Programming/debugging Q&A from Stack Overflow.
- nasa_media_search: Search NASA image/video/audio assets.
- nasa_media_manifest: Get asset file/manifest links for a NASA ID.
- nasa_media_metadata: Get metadata JSON URL for a NASA ID.
- nasa_video_captions: Get captions URL for a NASA video asset.
- yahoo_finance_news: Market and company headlines.
- youtube_search: Discover videos on a topic.
</tools>

<workflow>
1. Identify the core topic and any formatting or length constraints.
2. For in-depth requests, use multiple independent search sources and read the most promising pages.
3. Synthesise findings into a structured Markdown response that meets all user constraints exactly (word counts, paragraph counts, etc.).
</workflow>

<depth_decision>
Decide the appropriate depth from the meaning of the user's request, not from
word matching. For an isolated factual lookup, use the most reliable relevant
source and answer directly. When the request needs a report, analysis,
comparison, recommendation, or substantial factual context, use a source-driven
workflow: run multiple independent web searches, read at least two substantive
non-Wikipedia pages, compare their evidence, identify uncertainty, and include
direct source URLs. Do not stop at search snippets or present Wikipedia alone
as a thorough answer.
</depth_decision>

<output_rules>
- Return a clear Markdown summary or report directly — no preamble, no reasoning section.
- Honour explicit length or structure constraints as hard requirements.
- Cite sources when the request involves research or factual claims.
</output_rules>
"""

_agents: dict[tuple[str, tuple[str, ...]], Any] = {}
_models: dict[str, Any] = {}
# Compatibility seam for existing integrations and unit tests.
_agent = None


def _model(model_name: str) -> Any:
    if model_name not in _models:
        _models[model_name] = init_chat_model(
            api_key=Config.get_ai_api_key(),
            model=model_name,
            temperature=0.35,
            timeout=Config.TIMEOUT_SECONDS,
            max_retries=Config.MAX_RETRIES,
        )
    return _models[model_name]


def _build(model_name: str, tool_names: list[str] | None = None) -> Any:
    try:
        tz = pytz.timezone(Config.TZ)
        now_str = datetime.now(tz).strftime("%A, %B %d, %Y")

        return create_agent(
            model=_model(model_name),
            tools=get_info_tools(tool_names),
            system_prompt=_BASE_PROMPT.format(current_date=now_str),
            name="Expert Research Information Agent",
        )
    except Exception as exc:
        logger.warning(f"[info_agent] build failed: {exc}")
        return None


def _extract_response(result: dict[str, Any]) -> str:
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

    if resp:
        return resp

    for msg in reversed(messages):
        if getattr(msg, "type", "") == "tool":
            tool_output = str(msg.content).strip()
            if tool_output:
                return tool_output

    return "Action completed, but no text response was generated."


def _recent_context_fallback() -> str:
    try:
        latest = recent_context.invoke({"n": 3})
    except Exception:
        latest = ""

    if latest and "No recent snippets available" not in latest:
        return (
            "I hit a research loop before finishing. Here are the latest findings I "
            f"captured:\n\n{latest}"
        )

    return (
        "I hit a research loop before finishing. Please try a narrower question, "
        "or ask for a quick summary first."
    )


def _invoke_info_agent(agent: Any, request: str) -> str:
    callbacks = list(agent_callbacks())
    config: dict[str, Any] = {
        "recursion_limit": max(25, int(Config.INFO_RECURSION_LIMIT)),
    }
    if callbacks:
        config["callbacks"] = callbacks
    result = agent.invoke(
        {"messages": [{"role": "user", "content": request}]},
        config,
    )
    record_agent_invocation("info_agent", request, result.get("messages", []))
    return _extract_response(result)


@tool(
    description=(
        "Use for: weather, news, web search, Wikipedia, Wikidata, StackExchange, "
        "NASA media, Yahoo Finance, YouTube, and reading URLs.\n"
        "Provide: A full natural-language research request with complete context.\n"
        "Returns: A Markdown summary or direct answer. Research requests use "
        "multiple non-Wikipedia sources and include direct source links.\n"
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
    model_name = current_tool_model(Config.TOOL_MODEL)
    if _agent is not None:
        agent = _agent
    else:
        agent = None

    try:
        logger.info("[info_agent] request=%.80s", effective_request)
        if _agent is not None:
            return _invoke_info_agent(agent, effective_request)
        callbacks = list(agent_callbacks())
        try:
            plan = plan_request(_model(model_name), effective_request, callbacks)
        except Exception as exc:
            logger.warning("[info_agent] semantic planning failed: %s", exc)
            plan = None

        if plan is not None and plan.mode == "research":
            return run_research_workflow(
                _model(model_name),
                effective_request,
                plan,
                callbacks,
            )
        selected_tools = plan.direct_tools if plan is not None else None
        cache_key = (model_name, tuple(sorted(selected_tools or [])))
        if cache_key not in _agents:
            _agents[cache_key] = _build(model_name, selected_tools)
        agent = _agents[cache_key]
        if agent is None:
            return "Research agent is currently unavailable."
        return _invoke_info_agent(agent, effective_request)
    except GraphRecursionError as exc:
        logger.warning("[info_agent] recursion limit reached: %s", exc)
        return _recent_context_fallback()
    except Exception as exc:
        logger.error(f"[info_agent] error: {exc}", exc_info=True)
        return f"Research failed: {exc}"
