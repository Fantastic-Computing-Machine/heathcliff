from __future__ import annotations

from typing import List

from langchain.tools import tool
from pydantic import BaseModel, ConfigDict, Field

_RECENT_SNIPPETS: List[str] = []


def _add_recent_snippet(snippet: str) -> None:
    _RECENT_SNIPPETS.append(snippet)
    if len(_RECENT_SNIPPETS) > 12:
        del _RECENT_SNIPPETS[0 : len(_RECENT_SNIPPETS) - 12]


def _capture_recent_result(tool_name: str, content: str) -> None:
    """Store a compact, recency-biased snippet from a successful tool result."""

    if not content:
        return

    text = content.strip()
    if not text:
        return

    if text.lower().startswith("error"):
        return

    if len(text) > 1200:
        text = text[:1200] + "..."

    _add_recent_snippet(f"{tool_name}: {text}")


class RecentContextArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n: int = Field(3, description="Number of recent snippets to return. Default 3.")


@tool(args_schema=RecentContextArgs)
def recent_context(n: int = 3) -> str:
    """
    Return the freshest snippets captured from recent tool calls (max n).
    Use to ground answers with recency. Falls back with guidance if empty.
    """

    if not _RECENT_SNIPPETS:
        return "No recent snippets available. Run a search tool first."

    n = max(1, min(n, 5))
    latest = _RECENT_SNIPPETS[-n:][::-1]
    return "\n---\n".join(latest)
