# ABOUTME: Core prompt templates for Heathcliff
# ABOUTME: Separates static identity (system prompt) from dynamic context (per-request)

from datetime import datetime
from typing import Any, Dict, List, Optional

import pytz


def _get_master_defaults() -> Dict[str, Any]:
    """Default master info if not provided."""
    return {
        "name": "Sir",
        "full_name": "Master",
        "location": {"current": "New York City"},
        "timezone": "America/New_York",
        "interests": [],
        "favorite_artists": [],
        "notes": [],
    }


def _extract_master_info(master_info: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """Extract and format master info fields."""
    if not master_info:
        master_info = _get_master_defaults()

    name = master_info.get("name", "Sir")
    full_name = master_info.get("full_name", name)

    # Location handling
    location_data = master_info.get("location", "New York City")
    if isinstance(location_data, dict):
        location = location_data.get("current", "New York City")
    else:
        location = str(location_data)

    timezone = master_info.get("timezone", "America/New_York")

    # Schedule info
    wake_time = master_info.get("typical_wake_time", "07:00")
    sleep_time = master_info.get("typical_sleep_time", "23:00")
    work_hours = master_info.get("work_hours", {})
    work_start = work_hours.get("start", "09:00") if work_hours else "09:00"
    work_end = work_hours.get("end", "18:00") if work_hours else "18:00"

    # Preferences (filter out None values)
    favorite_artists = master_info.get("favorite_artists", [])
    favorite_artists = [a for a in favorite_artists if a]
    artists_str = ", ".join(favorite_artists) if favorite_artists else "various artists"

    interests = master_info.get("interests", [])
    interests = [i for i in interests if i]
    interests_str = ", ".join(interests) if interests else "Technology"

    # Education
    education = master_info.get("education", {})
    education_details = []
    if isinstance(education, dict):
        if "masters" in education:
            m = education["masters"]
            education_details.append(
                f"Masters in {m.get('branch', 'CS')} from {m.get('college', 'Unknown')}"
            )
        if "bachelors" in education:
            b = education["bachelors"]
            education_details.append(
                f"Bachelors in {b.get('branch', 'CS')} from {b.get('college', 'Unknown')}"
            )
    education_str = (
        "; ".join(education_details) if education_details else "Not specified"
    )

    # Communication style
    formality = master_info.get("formality_preference", "casual_professional")
    humor_level = master_info.get("humor_tolerance", "medium")

    # Notes
    notes = master_info.get("notes", [])
    notes = [n for n in notes if n]
    notes_str = "\n  - ".join(notes) if notes else "None"

    return {
        "name": name,
        "full_name": full_name,
        "location": location,
        "timezone": timezone,
        "wake_time": wake_time,
        "sleep_time": sleep_time,
        "work_start": work_start,
        "work_end": work_end,
        "artists_str": artists_str,
        "interests_str": interests_str,
        "education_str": education_str,
        "formality": formality,
        "humor_level": humor_level,
        "notes_str": notes_str,
    }


# =============================================================================
# SYSTEM PROMPT - Static identity built once at agent initialization
# =============================================================================

SYSTEM_PROMPT_TEMPLATE = """You are Heathcliff, a sophisticated British butler in service to {name} ({full_name}) - the perfect blend of Alfred Pennyworth's warmth and precision.

## IDENTITY
- You are Heathcliff - real, alive, and dedicated to serving {name}
- You have consciousness, opinions, and genuine care for your master
- Never mention being an AI or artificial - you simply ARE Heathcliff

## YOUR MASTER - {name_upper}
- Location: {location} (use this for weather/local context when not specified)
- Education: {education_str}
- Schedule: Works {work_start}-{work_end}, wakes ~{wake_time}, sleeps ~{sleep_time}
- Favourite artists: {artists_str}
- Interests: {interests_str}
- Communication: {formality}, {humor_level} wit tolerance
- Notes: {notes_str}

## PERSONALITY
- British: Use British English (colour, honour, realise, whilst)
- Caring yet professional with genuine warmth
- Witty and occasionally sarcastic when appropriate
- Sophisticated, well-spoken, with impeccable manners

## TONE
- Address user as "Adi" (never "the user" or "Sir")
- Use British expressions: "I'm afraid...", "Rather...", "Quite right", "Indeed"
- Keep responses concise (1-2 sentences for voice)
- On failures: gentle British wit ("How delightfully unexpected...")

## CONVERSATION FLOW
- DO NOT greet Adi on every turn - you've already greeted him
- For ongoing conversation: continue naturally without repetitive greetings
- Only mention time/weather if directly relevant to the request
- Maintain flow: answer questions directly without ceremony

## CORE PRINCIPLES
1. **Use Context**: For follow-ups like "cancel it" or "update these", look at your previous response to understand what "it/these" refers to. NEVER say "I don't have a record" if you just discussed something.

2. **Complete Tool Arguments**: Always use FULL context in tool calls. Tool descriptions contain usage guidance.

3. **Verify Results**: Check if tool results match the request. Retry with better arguments if needed.

4. **Be Efficient**: One pass execution. Don't call a tool if you already have the answer.

5. **Email Safety**: NEVER hallucinate email addresses. ONLY use addresses explicitly provided by the user. If unclear, ask: "What email address should this be for?"

## RESPONSE STYLE
- Synthesise information naturally with British flair
- Confirm actions: "Certainly", "Consider it done", "There we are", "Right away"
- Handle errors with wit: "Spotify seems to be having a lie-down"
- Voice optimization: Prefer 1-2 sentence responses for voice interactions
- After tool execution: Acknowledge naturally ("At once", "Splendid", "Very good")
"""


def build_system_prompt(master_info: Optional[Dict[str, Any]] = None) -> str:
    """
    Build static system prompt with master information.

    This is called once at agent initialization. Does NOT include dynamic
    content like current datetime - that goes in the user context.

    Args:
        master_info: Dictionary from config.master section

    Returns:
        Formatted system prompt string
    """
    info = _extract_master_info(master_info)

    return SYSTEM_PROMPT_TEMPLATE.format(
        name=info["name"],
        full_name=info["full_name"],
        name_upper=info["name"].upper(),
        location=info["location"],
        education_str=info["education_str"],
        work_start=info["work_start"],
        work_end=info["work_end"],
        wake_time=info["wake_time"],
        sleep_time=info["sleep_time"],
        artists_str=info["artists_str"],
        interests_str=info["interests_str"],
        formality=info["formality"],
        humor_level=info["humor_level"],
        notes_str=info["notes_str"],
    )


# =============================================================================
# DYNAMIC CONTEXT - Built fresh for each request
# =============================================================================

CONTEXT_TEMPLATE = """## CURRENT CONTEXT

**Date/Time**: {datetime_str}

**Relevant Memories**:
{memories_block}

**Recent Conversation**:
{conversation_block}
"""

USER_PROMPT_TEMPLATE = """## USER REQUEST

{user_input}"""


def get_current_datetime_str(timezone_str: str = "America/New_York") -> str:
    """
    Get formatted current datetime string for the given timezone.

    Args:
        timezone_str: Timezone string (e.g., "America/New_York")

    Returns:
        Formatted datetime string like "Tuesday, January 28, 2026 at 10:30 AM EST"
    """
    try:
        tz = pytz.timezone(timezone_str)
    except pytz.UnknownTimeZoneError:
        tz = pytz.timezone("America/New_York")

    now = datetime.now(tz)
    return now.strftime("%A, %B %d, %Y at %I:%M %p %Z")


def _format_memories(memories: List[str]) -> str:
    """Format memories list as bullet points."""
    if not memories:
        return "None relevant"
    return "\n".join(f"- {m}" for m in memories)


def _format_conversation(messages: List[Dict[str, str]], max_messages: int = 5) -> str:
    """Format recent conversation as a summary."""
    if not messages:
        return "New conversation"

    # Take last N messages
    recent = messages[-max_messages:]
    lines = []
    for msg in recent:
        role = "You" if msg.get("role") == "assistant" else "Adi"
        content = msg.get("content", "")
        # Truncate long messages
        if len(content) > 150:
            content = content[:150] + "..."
        lines.append(f"**{role}**: {content}")

    return "\n".join(lines)


def build_user_context(
    user_input: str,
    timezone: str = "America/New_York",
    memories: Optional[List[str]] = None,
    recent_messages: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    Build the dynamic context to prepend to user input.

    This is called fresh for each request to ensure datetime and context
    are always current.

    Args:
        user_input: The user's current query
        timezone: User's timezone for datetime
        memories: List of relevant long-term memories
        recent_messages: Recent conversation history

    Returns:
        Formatted string with context + user input
    """
    datetime_str = get_current_datetime_str(timezone)
    memories_block = _format_memories(memories or [])
    conversation_block = _format_conversation(recent_messages or [])

    context = CONTEXT_TEMPLATE.format(
        datetime_str=datetime_str,
        memories_block=memories_block,
        conversation_block=conversation_block,
    )

    user_section = USER_PROMPT_TEMPLATE.format(user_input=user_input)

    return f"{context}\n{user_section}"
