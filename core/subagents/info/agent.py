# ABOUTME: Info / research sub-agent — web search, weather, news, Wikipedia
# ABOUTME: Wraps tools/info_tools.py; exposed to supervisor as a single @tool

import re
from datetime import datetime
from typing import Any, Tuple

import pytz
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.errors import GraphRecursionError

from config import Config
from core.subagents._runner import record_agent_invocation
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
2. Run search_web, then read_website on the most promising sources for in-depth requests.
3. Synthesise findings into a structured Markdown response that meets all user constraints exactly (word counts, paragraph counts, etc.).
</workflow>

<mode>
{mode_guidance}
</mode>

<output_rules>
- Return a clear Markdown summary or report directly — no preamble, no reasoning section.
- Honour explicit length or structure constraints as hard requirements.
- Cite sources when the request involves research or factual claims.
</output_rules>
"""

_FAST_MODE_GUIDANCE = """\
- Prioritise speed and stability for short factual requests.
- Stop as soon as you have enough confidence to answer.
- Keep tool hops minimal (typically 1-2) and avoid repeatedly reading many full webpages.
"""

_DEEP_MODE_GUIDANCE = """\
- Prioritise completeness for analysis, comparisons, and source-heavy requests.
- Gather multiple sources when needed and synthesise them with citations.
- Do not loop indefinitely; stop after sufficient evidence is collected.
"""

_DEEP_INTENT_KEYWORDS = (
    "deep",
    "detailed",
    "analyze",
    "analysis",
    "compare",
    "comparison",
    "report",
    "with sources",
    "cite",
    "citations",
    "step-by-step",
)

_STRUCTURAL_MARKERS = (
    " then ",
    " also ",
    " additionally ",
    " in addition ",
    " versus ",
    " vs ",
)

_OUTPUT_CONSTRAINT_PATTERN = re.compile(
    r"(\b\d+\s*(words?|paragraphs?|sections?|bullets?)\b|table|markdown|sources?)",
    re.IGNORECASE,
)

_agent = None
_fast_agent = None
_deep_agent = None


def _build(mode: str = "deep") -> Any:
    try:
        tz = pytz.timezone(Config.TZ)
        now_str = datetime.now(tz).strftime("%A, %B %d, %Y")
        mode_guidance = _FAST_MODE_GUIDANCE if mode == "fast" else _DEEP_MODE_GUIDANCE
        temperature = 0.25 if mode == "fast" else 0.4

        return create_agent(
            model=init_chat_model(
                api_key=Config.get_ai_api_key(),
                model=Config.TOOL_MODEL,
                temperature=temperature,
                timeout=Config.TIMEOUT_SECONDS,
                max_retries=Config.MAX_RETRIES,
            ),
            tools=get_info_tools(),
            system_prompt=_BASE_PROMPT.format(
                current_date=now_str,
                mode_guidance=mode_guidance,
            ),
            name="Expert Research Information Agent",
        )
    except Exception as exc:
        logger.warning(f"[info_agent] {mode} build failed: {exc}")
        return None


def _choose_research_mode(request: str) -> Tuple[str, list[str]]:
    text = (request or "").strip()
    lowered = text.lower()
    reasons: list[str] = []

    if any(keyword in lowered for keyword in _DEEP_INTENT_KEYWORDS):
        reasons.append("explicit_deep_intent")

    if _OUTPUT_CONSTRAINT_PATTERN.search(lowered):
        reasons.append("output_constraint")

    structural_score = 0
    if len(text) >= 220:
        structural_score += 1
    if sum(lowered.count(marker) for marker in _STRUCTURAL_MARKERS) >= 2:
        structural_score += 1
    if text.count("?") >= 2:
        structural_score += 1

    if structural_score >= 2:
        reasons.append("complex_structure")

    mode = "deep" if reasons else "fast"
    return mode, reasons


def _recursion_limit_for_mode(mode: str) -> int:
    if mode == "deep":
        return max(25, int(Config.INFO_DEEP_RECURSION_LIMIT))
    return max(5, int(Config.INFO_FAST_RECURSION_LIMIT))


def _get_or_build_mode_agent(mode: str):
    global _fast_agent, _deep_agent

    if mode == "fast":
        if _fast_agent is None:
            _fast_agent = _build(mode="fast")
        return _fast_agent

    if _deep_agent is None:
        _deep_agent = _build(mode="deep")
    return _deep_agent


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


def _invoke_info_agent(agent: Any, request: str, mode: str) -> str:
    result = agent.invoke(
        {"messages": [{"role": "user", "content": request}]},
        {"recursion_limit": _recursion_limit_for_mode(mode)},
    )
    record_agent_invocation("info_agent", request, result.get("messages", []))
    return _extract_response(result)


@tool(
    description=(
        "Use for: weather, news, web search, Wikipedia, Wikidata, StackExchange, "
        "NASA media, Yahoo Finance, YouTube, and reading URLs.\n"
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

    mode = "deep"
    reasons: list[str] = []
    if Config.INFO_ADAPTIVE_ROUTING_ENABLED:
        mode, reasons = _choose_research_mode(effective_request)

    if _agent is not None:
        selected_agent = _agent
        selected_mode = "deep"
        logger.debug("[info_agent] using legacy _agent override")
    else:
        selected_agent = _get_or_build_mode_agent(mode)
        selected_mode = mode

    if selected_agent is None:
        return "Research agent is currently unavailable."

    try:
        logger.info(
            "[info_agent] mode=%s reasons=%s request=%.80s",
            selected_mode,
            reasons or ["default"],
            effective_request,
        )
        return _invoke_info_agent(selected_agent, effective_request, selected_mode)
    except GraphRecursionError as exc:
        logger.warning(
            "[info_agent] recursion hit in mode=%s (limit=%s): %s",
            selected_mode,
            _recursion_limit_for_mode(selected_mode),
            exc,
        )

        should_escalate = (
            selected_mode == "fast"
            and Config.INFO_FAST_TO_DEEP_ESCALATION_ENABLED
            and _agent is None
        )
        if should_escalate:
            deep_agent = _get_or_build_mode_agent("deep")
            if deep_agent is not None:
                try:
                    logger.info("[info_agent] escalating fast query to deep mode")
                    return _invoke_info_agent(deep_agent, effective_request, "deep")
                except GraphRecursionError:
                    logger.warning("[info_agent] deep escalation also hit recursion")
                    return _recent_context_fallback()
                except Exception as deep_exc:
                    logger.error(
                        "[info_agent] deep escalation failed: %s",
                        deep_exc,
                        exc_info=True,
                    )
                    return _recent_context_fallback()

        return _recent_context_fallback()
    except Exception as exc:
        logger.error(f"[info_agent] error: {exc}", exc_info=True)
        return f"Research failed: {exc}"
