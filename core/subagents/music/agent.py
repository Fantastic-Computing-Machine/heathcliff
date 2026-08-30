# ABOUTME: Music / Spotify sub-agent — playback control
# ABOUTME: Wraps tools/spotify_tool.py; exposed to supervisor as a single @tool

from typing import Any

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool

from config import Config
from core.runtime_profile import current_tool_model
from core.subagents._runner import run_agent
from core.subagents.music.tools import (
    SPOTIFY_AUTH_REQUIRED_MESSAGE,
    get_spotify_tools,
    spotify_is_connected,
)
from logger import logger

_SYSTEM_PROMPT = """\
You are a Spotify music control specialist.

<task>
Interpret music requests and execute the correct Spotify playback action.
</task>

<rules>
1. Use play_playlist only for a playlist the user says is theirs. Use search_spotify_catalog to discover public music, then play_spotify_playlist for a public playlist.
2. Search the catalogue before selecting music for a genre, mood, recommendation, or open-ended request. Search can return tracks, playlists, albums, and artists.
3. If a catalogue search returns no result and the request appears to contain an obvious typo, retry it once with the corrected wording; otherwise ask the user.
4. Pass the song title, artist, and device as separate play_track fields. Use an empty artist only when it is genuinely unknown.
5. Preserve device preferences. If the requested device is not found, ask to play on the default device instead of silently switching.
6. Every playback, resume, and volume tool has a required device field. Copy the exact requested device into it; use an empty string only when the user gave no device.
7. When the user requests a volume, call set_volume after successful playback on the requested device.
8. If Spotify says that the device does not support remote volume control, do not retry; confirm any playback and tell the user to adjust volume on the device.
9. Select resume_playback whenever the user's intent is to continue an existing queue; do not select a search or play-new-music tool for that intent.
10. Return a brief, plain-text confirmation of the action taken.
</rules>
""".strip()

_agents: dict[str, Any] = {}
# Compatibility seam for existing integrations and unit tests.
_agent = None


def _build(model_name: str) -> Any:
    try:
        return create_agent(
            model=init_chat_model(
                api_key=Config.get_ai_api_key(),
                model=model_name,
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
        "Use for: playing, pausing, checking, or searching Spotify.\n"
        "Searches Spotify's public catalogue for songs, playlists, albums, and artists.\n"
        "Provide: A natural-language music request with song, playlist, device, and volume details.\n"
        "Returns: A text confirmation of the action taken.\n"
        'Example: music_agent_tool(request="Play Taylor Swift - Love Story")\n'
        'Example: music_agent_tool(request="What song is currently playing?")'
    ),
)
def music_agent_tool(request: str) -> str:
    """Control Spotify music playback."""
    global _agent
    if _agent is None and not spotify_is_connected():
        return SPOTIFY_AUTH_REQUIRED_MESSAGE

    model_name = current_tool_model(Config.SUBAGENT_MODEL)
    if _agent is not None:
        agent = _agent
    elif model_name not in _agents:
        _agents[model_name] = _build(model_name)
        agent = _agents[model_name]
    else:
        agent = _agents[model_name]
    if agent is None:
        return "Music agent is currently unavailable."
    return run_agent(agent, request, "music_agent", "Music control failed")
