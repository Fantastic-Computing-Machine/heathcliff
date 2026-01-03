# ABOUTME: Tools package initialization
# ABOUTME: Exports all LangChain tools for the Heathcliff assistant

from typing import Any, Iterable, List, Sequence

from logger import logger
from tools.calendar_tools import get_calendar_toolkit_tools
from tools.comm_tools import get_comm_tools
from tools.drive_tools import get_drive_tools
from tools.gmail_tools import get_gmail_toolkit_tools
from tools.info_tools import get_info_tools
from tools.people_tools import get_people_tools
from tools.spotify_tool import get_spotify_tools

__all__ = [
    "get_all_tools",
]


def _dedupe_tools(tool_groups: Sequence[Iterable[Any]]) -> List[Any]:
    """Flatten tool providers while keeping the first tool per unique name."""

    merged: List[Any] = []
    seen = set()

    for group in tool_groups:
        if not group:
            continue

        for tool in group:
            name = getattr(tool, "name", getattr(tool, "__name__", None))
            if not name:
                continue

            key = name.lower()
            if key in seen:
                continue

            merged.append(tool)
            seen.add(key)

    return merged


def get_all_tools() -> List[Any]:
    """
    Get all tools as a single list for agent registration.

    Returns:
        List of tool callables/BaseTool instances ready for registration.
    """
    tool_groups: List[Iterable[Any]] = []

    try:
        tool_groups.append(get_gmail_toolkit_tools())
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"Failed to load Gmail toolkit tools: {exc}")

    try:
        tool_groups.append(get_calendar_toolkit_tools())
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"Failed to load Google Calendar toolkit tools: {exc}")

    try:
        tool_groups.append(get_drive_tools())
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"Failed to load Google Drive tools: {exc}")

    tool_groups.extend(
        [
            get_spotify_tools(),
            get_info_tools(),
            get_comm_tools(),
            get_people_tools(),
        ]
    )

    merged = _dedupe_tools(tool_groups)

    if not merged:  # pragma: no cover - defensive
        logger.warning("No tools registered; returning empty list")

    return merged
