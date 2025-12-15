# ABOUTME: Tools package initialization
# ABOUTME: Exports all LangChain tools for the Heathcliff assistant

from importlib import import_module
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from config import get_config
from logger import logger

__all__ = [
    # Spotify tools
    "play_track",
    "pause_playback",
    "current_track",
    "get_spotify_tools",
    # Info tools
    "get_weather",
    "get_news",
    "search_web",
    "wikipedia_search",
    "get_info_tools",
    # Communication tools
    "send_to_telegram",
    "read_gdrive_file",
    "get_comm_tools",
    # People tools
    "search_contacts",
    "get_people_tools",
    # Aggregator
    "get_all_tools",
]

_TOOL_EXPORTS: Dict[str, Tuple[str, str]] = {
    # Spotify tools
    "play_track": ("tools.spotify_tool", "play_track"),
    "pause_playback": ("tools.spotify_tool", "pause_playback"),
    "current_track": ("tools.spotify_tool", "current_track"),
    "get_spotify_tools": ("tools.spotify_tool", "get_spotify_tools"),
    # Info tools
    "get_weather": ("tools.info_tools", "get_weather"),
    "get_news": ("tools.info_tools", "get_news"),
    "search_web": ("tools.info_tools", "search_web"),
    "wikipedia_search": ("tools.info_tools", "wikipedia_search"),
    "get_info_tools": ("tools.info_tools", "get_info_tools"),
    # Communication tools
    "send_to_telegram": ("tools.comm_tools", "send_to_telegram"),
    "read_gdrive_file": ("tools.comm_tools", "read_gdrive_file"),
    "get_comm_tools": ("tools.comm_tools", "get_comm_tools"),
    # People tools
    "search_contacts": ("tools.people_tools", "search_contacts"),
    "get_people_tools": ("tools.people_tools", "get_people_tools"),
}

_INTERNAL_EXPORTS = {
    "get_gmail_toolkit_tools": ("tools.gmail_tools", "get_gmail_toolkit_tools"),
    "get_calendar_toolkit_tools": ("tools.calendar_tools", "get_calendar_toolkit_tools"),
}


def __getattr__(name: str) -> Any:
    if name not in _TOOL_EXPORTS:
        raise AttributeError(f"module 'tools' has no attribute '{name}'")

    module_name, attr_name = _TOOL_EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> List[str]:  # pragma: no cover - debug helper
    return sorted(set(list(globals().keys()) + list(__all__)))


def _import_attr(name: str) -> Any:
    if name in globals():
        return globals()[name]

    if name in _TOOL_EXPORTS:
        return __getattr__(name)

    if name in _INTERNAL_EXPORTS:
        module_name, attr_name = _INTERNAL_EXPORTS[name]
        module = import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value

    raise AttributeError(f"No tool export registered for '{name}'")


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


def get_all_tools(prefer_langchain_toolkits: bool | None = None) -> List[Any]:
    """
    Get all tools as a single list for agent registration.

    Args:
        prefer_langchain_toolkits: Override config when prioritizing LangChain community toolkits.

    Returns:
        List of tool callables/BaseTool instances ready for registration.
    """

    config = get_config()
    prefer_toolkits = (
        prefer_langchain_toolkits
        if prefer_langchain_toolkits is not None
        else config.get("tools.prefer_langchain_toolkits", True)
    )

    tool_groups: List[Iterable[Any]] = []

    if prefer_toolkits:
        try:
            tool_groups.append(_import_attr("get_gmail_toolkit_tools")())
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Failed to load Gmail toolkit tools: {exc}")

        try:
            tool_groups.append(_import_attr("get_calendar_toolkit_tools")())
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Failed to load Google Calendar toolkit tools: {exc}")

    # Custom/fallback implementations
    tool_groups.extend(
        [
            _import_attr("get_spotify_tools")(),
            _import_attr("get_info_tools")(),
            _import_attr("get_comm_tools")(),
            _import_attr("get_people_tools")(),
        ]
    )

    merged = _dedupe_tools(tool_groups)

    if not merged:  # pragma: no cover - defensive
        logger.warning("No tools registered; returning empty list")

    return merged
