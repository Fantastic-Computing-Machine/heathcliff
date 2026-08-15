# ABOUTME: Music / Spotify sub-agent — playback control
# ABOUTME: Wraps tools/spotify_tool.py; exposed to supervisor as a single @tool

from typing import Any

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool

from config import Config
from core.subagents._runner import run_agent
from core.subagents.music.tools import get_spotify_tools
from logger import logger

_SYSTEM_PROMPT = """\
You are a Spotify music control specialist.

<task>
Interpret music requests and execute the correct Spotify playback action.
</task>

<rules>
1. Use play_playlist for any request that names a playlist; never use play_track for a playlist.
2. Use the full song name AND artist name when calling play_track (e.g. "Taylor Swift - Love Story").
3. Preserve device preferences. If the requested device is not found, ask to play on the default device instead of silently switching.
4. Return a brief, plain-text confirmation of the action taken.
</rules>
""".strip()

_agent = None


def _build() -> Any:
    try:
        return create_agent(
            model=init_chat_model(
                api_key=Config.get_ai_api_key(),
                model=Config.SUBAGENT_MODEL,
                temperature=0.2,
                max_tokens=Config.MAX_TOKENS,
                timeout=Config.TIMEOUT_SECONDS,
                max_retries=Config.MAX_RETRIES,
            ),
            tools=get_spotify_tools(),
            system_prompt=_SYSTEM_PROMPT,
            name="Expert DJ and Music Fanatic",
        )
    except Exception as exc:
        logger.warning(f"[music_agent] build failed: {exc}")
        return None


@tool(
    description=(
        "Use for: playing, pausing, or checking Spotify playback.\n"
        "Provide: A natural-language music request with song and artist details.\n"
        "Returns: A text confirmation of the action taken.\n"
        'Example: music_agent_tool(request="Play Taylor Swift - Love Story")\n'
        'Example: music_agent_tool(request="What song is currently playing?")'
    ),
)
def music_agent_tool(request: str) -> str:
    """Control Spotify music playback."""
    global _agent
    if _agent is None:
        _agent = _build()
    if _agent is None:
        return "Music agent is currently unavailable."
    return run_agent(_agent, request, "music_agent", "Music control failed")
