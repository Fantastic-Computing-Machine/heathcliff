# ABOUTME: Master info skill — user's live profile, actively updated during conversations
# ABOUTME: _ACTIVE stores the current state; update_master_info @tool patches it at runtime
# ABOUTME: Heathcliff calls update_master_info() whenever he learns something new about the user

import copy
import json
import threading
from typing import Any, Dict, List

from langchain.tools import tool

from config import Config
from logger import logger


# ---------------------------------------------------------------------------
# Seed data: pulled from master_info.toml via Config at startup
# ---------------------------------------------------------------------------
def _load_seed() -> Dict[str, Any]:
    try:
        return copy.deepcopy(Config.MASTER_INFO)
    except Exception as exc:
        logger.warning(f"[master_info_skill] Could not load seed data: {exc}")
        return {"name": "User", "full_name": "User"}


# Thread-safe mutable profile — the single source of truth at runtime
_lock = threading.Lock()
_ACTIVE: Dict[str, Any] = _load_seed()


# ---------------------------------------------------------------------------
# Public accessors
# ---------------------------------------------------------------------------


def get_master_info() -> Dict[str, Any]:
    """Return a snapshot of the current active master-info dict."""
    with _lock:
        return copy.deepcopy(_ACTIVE)


def set_master_info_field(key: str, value: Any) -> None:
    """Directly set a top-level field (for programmatic use, e.g. tests)."""
    with _lock:
        _ACTIVE[key] = value
    logger.debug(f"[master_info_skill] set '{key}' = {value!r}")


def append_to_list_field(field: str, value: str) -> None:
    """Append a value to a list field (e.g. 'interests', 'notes')."""
    with _lock:
        if not isinstance(_ACTIVE.get(field), list):
            _ACTIVE[field] = []
        if value not in _ACTIVE[field]:
            _ACTIVE[field].append(value)
    logger.debug(f"[master_info_skill] appended '{value}' → {field}")


def _format_profile(info: Dict[str, Any]) -> str:
    """Render the active profile as a structured skill prompt."""
    name = info.get("name", "User")
    full_name = info.get("full_name", name)
    location = info.get("location", "unknown")
    tz = Config.TZ
    wake = info.get("typical_wake_time", "10:00")
    sleep = info.get("typical_sleep_time", "03:00")
    work = info.get("work_hours", {}) or {}
    work_start = work.get("start", "09:00")
    work_end = work.get("end", "18:00")
    artists = info.get("favorite_artists", [])
    interests = info.get("interests", [])
    notes = info.get("notes", [])
    formality = info.get("formality_preference", "casual_professional")
    humor = info.get("humor_tolerance", "high")

    # Dynamic fields set at runtime
    extras = {
        k: v
        for k, v in info.items()
        if k
        not in {
            "name",
            "full_name",
            "location",
            "timezone",
            "typical_wake_time",
            "typical_sleep_time",
            "work_hours",
            "favorite_artists",
            "interests",
            "notes",
            "formality_preference",
            "humor_tolerance",
        }
    }

    lines = [
        f"# Active Master Profile — {name} ({full_name})",
        "",
        "## Location & Time",
        f"- Location: {location}",
        f"- Timezone: {tz}",
        "",
        "## Daily Schedule",
        f"- Wake: {wake}  |  Sleep: {sleep}  (often works late)",
        f"- Work hours: {work_start} – {work_end}",
        "",
        "## Music",
        f"- Favourite artists: {', '.join(artists) if artists else 'various'}",
        "",
        "## Interests",
        *[f"- {i}" for i in interests],
        "",
        "## Communication Style",
        f"- Formality: {formality}  |  Humour tolerance: {humor}",
        "",
        "## Personal Notes",
        *[f"- {n}" for n in notes],
    ]

    if extras:
        lines += ["", "## Additional (learned at runtime)"]
        for k, v in extras.items():
            lines.append(f"- {k}: {v}")

    lines += [
        "",
        "## Rules",
        f"- Address user as '{name}' always",
        "- Use location for local queries (weather, time, etc.)",
        "- Respect schedule when suggesting times or alarms",
        "- Reference favourite artists for unsolicited music suggestions",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Short description for the system prompt (pre-loaded, brief)
# ---------------------------------------------------------------------------
MASTER_INFO_DESCRIPTION = (
    "User's live profile: location, schedule, music preferences, interests, and notes. "
    "Load when personalising responses or referencing the user's preferences."
)


def get_skill_content() -> str:
    """Return the full rendered skill content from the current active profile."""
    with _lock:
        return _format_profile(_ACTIVE)


# ---------------------------------------------------------------------------
# @tool — supervisor calls this to update the user's profile at runtime
# ---------------------------------------------------------------------------


@tool(
    description=(
        "Use for: recording new information about the user discovered during conversation.\n"
        "Provide: A field name and value to set or append.\n"
        "Returns: Confirmation that the profile was updated.\n"
        "List fields (appends): interests, favorite_artists, notes.\n"
        "String fields (replaces): location, timezone, typical_wake_time, "
        "typical_sleep_time, formality_preference, humor_tolerance.\n"
        'Example: update_master_info(field="interests", value="Rock climbing")\n'
        'Example: update_master_info(field="location", value="Brooklyn, NY")'
    ),
)
def update_master_info(field: str, value: str) -> str:
    """Update the user's active profile when you learn something new about them."""
    with _lock:
        if field in ("interests", "favorite_artists", "notes"):
            if not isinstance(_ACTIVE.get(field), list):
                _ACTIVE[field] = []
            if value not in _ACTIVE[field]:
                _ACTIVE[field].append(value)
                logger.info(f"[master_info_skill] appended '{value}' → '{field}'")
                return f"Noted: added '{value}' to {field}."
            else:
                return f"Already knew that: '{value}' is already in {field}."
        elif field == "work_hours":
            # Accept JSON string like '{"start": "10:00", "end": "19:00"}'
            try:
                _ACTIVE["work_hours"] = json.loads(value)
                logger.info(f"[master_info_skill] updated work_hours: {value}")
                return f"Updated work hours: {value}"
            except json.JSONDecodeError:
                return 'work_hours must be a JSON string like \'{"start": "10:00", "end": "19:00"}\''
        else:
            _ACTIVE[field] = value
            logger.info(f"[master_info_skill] set '{field}' = '{value}'")
            return f"Noted: {field} updated to '{value}'."


def get_master_info_tools() -> List[Any]:
    """Return the master-info management tools for supervisor registration."""
    return [update_master_info]
