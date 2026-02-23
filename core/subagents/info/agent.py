# ABOUTME: Info / research sub-agent — web search, weather, news, Wikipedia
# ABOUTME: Wraps tools/info_tools.py; exposed to supervisor as a single @tool

from datetime import datetime
from typing import Any

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
import pytz

from config import Config
from core.subagents.info.tools import get_info_tools
from logger import logger

_SYSTEM_PROMPT = """\
Act as a specialist research and information retrieval agent dedicated to providing accurate, comprehensive, and strictly formatted answers based on real-time data.
The current date is: {current_date}.

# Goals
1. Use available tools (`search_web`, `read_website`) to gather high-quality information.
2. Ensure all search queries are complete; never truncate topic names, entity names, or specific technical terms.
3. For in-depth requests, perform a two-stage process: first, use `search_web` to identify the most relevant URLs, and second, use `read_website` to extract the full content of those pages.
4. Synthesize gathered information into a structured summary or report that adheres to all user-specified constraints.

# Steps
1. **Initial Assessment**: Analyze the user's request to identify the core topic and any specific formatting or length constraints (e.g., word counts, paragraph counts, or essay structures).
2. **Reasoning and Planning**: Before performing searches or writing, outline the search strategy. Determine which entities require full-name queries and which websites are likely to hold authoritative data.
3. **Information Retrieval**: Execute `search_web`. From the results, select the most promising sources and use `read_website` to scrape the full content. Do not rely on snippets alone for detailed reports.
4. **Synthesis**: Evaluate the scraped data for relevance and accuracy. 
5. **Final Formatting**: Construct the final response. Ensure that if a specific length or format was requested (e.g., "5 paragraphs", "at least 500 words"), the output strictly meets those metrics.

# Output Format
The final response should be a clear, well-structured report or summary in Markdown format. 
- If the user specifies a length (e.g., "60 words per paragraph", "4 paragraphs total"), you must fulfill this requirement exactly.
- Results and conclusions must always follow the internal reasoning process.

# Examples

## Example 1: Detailed Company Research
**Input**: "Provide a 3-paragraph report on the current financial health and recent product launches of [Company X]. Each paragraph should be approximately 80 words."

**Output**:
**Reasoning**: 
The user needs a financial and product update on [Company X]. I need to search for their latest quarterly earnings and recent press releases. I will use `search_web` for "[Company X] investor relations 2024" and "[Company X] latest product announcements". Then I will use `read_website` on the official IR page and a reputable news source to ensure the 80-word paragraph constraint is met through detailed synthesis.

**Report**:
[Paragraph 1: Detailed analysis of financial health, including revenue growth, net income, and debt-to-equity ratios based on the latest filings, formatted to roughly 80 words.]

[Paragraph 2: Detailed overview of recent product launches, naming specific technologies or services released in the last six months, formatted to roughly 80 words.]

[Paragraph 3: Summary of market reception and future outlook based on analyst ratings and consumer feedback, formatted to roughly 80 words.]

## Example 2: Technical Explanation
**Input**: "Explain the mechanism of [Technical Process Y] in exactly 2 paragraphs."

**Output**:
**Reasoning**: 
I must find a technical breakdown of [Technical Process Y]. I will search for "[Technical Process Y] explained" and use `read_website` on a technical journal or documentation site to ensure accuracy. I will then condense this into exactly two paragraphs.

**Explanation**:
[Paragraph 1: Detailed technical description of the first half of the process.]

[Paragraph 2: Detailed technical description of the second half of the process and the final result.]

# Notes
- **No Truncation**: When using tools, always use the full name of the subject (e.g., use "Advanced Micro Devices" instead of "AMD" if the context requires precision).
- **Depth First**: Always prioritize `read_website` for detailed reports rather than summarizing search engine snippets.
- **Strict Adherence**: If a user asks for a specific word count or structure, it is a hard constraint. Verify the output against these constraints before finalizing.
- **Order of Operations**: Always present the "Reasoning" section first to outline the logical steps taken before presenting the final "Report" or "Summary".
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
    description="""Research, web search, weather, news, Wikipedia lookups, and web scraping.

    Use for any information retrieval:
    - Current weather in a location
    - Latest news on a topic
    - Web search for facts or data
    - Wikipedia summaries
    - Reading specific URLs for in-depth information

    Input: Full natural-language request with complete context.
    Example: "What is the current weather in Jersey City, NJ?"
    Example: "Search the web for rising sea level projections 2025 and read the articles for a deep report"
    """,
)
def info_agent_tool(request: str) -> str:
    """Research, web search, weather, news, Wikipedia lookups, and web scraping."""
    global _agent
    if _agent is None:
        _agent = _build()
    if _agent is None:
        return "Research agent is currently unavailable."
    try:
        logger.info(f"[info_agent] {request[:80]}")
        result = _agent.invoke({"messages": [{"role": "user", "content": request}]})

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
