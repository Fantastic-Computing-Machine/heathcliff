# ABOUTME: Music / Spotify sub-agent — playback control
# ABOUTME: Wraps tools/spotify_tool.py; exposed to supervisor as a single @tool

from typing import Any

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from config import Config
from logger import logger

_SYSTEM_PROMPT = """\
You are a specialist Spotify music control agent.
Your job: control Spotify playback with the tools available.
ALWAYS use the full song name + artist name for play_track:
  Good: play_track("Taylor Swift - Love Story")
  Bad:  play_track("taylor")
Return a brief confirmation of what was done.
"""

_agent = None


def _build() -> Any:
    try:
        from core.subagents.music.tools import get_spotify_tools

        return create_agent(
            model=ChatGoogleGenerativeAI(
                model=Config.MODEL,
                google_api_key=Config.GEMINI_API_KEY,
                temperature=0.2,
                max_output_tokens=Config.MAX_TOKENS,
            ),
            tools=get_spotify_tools(),
            system_prompt=_SYSTEM_PROMPT,
        )
    except Exception as exc:
        logger.warning(f"[music_agent] build failed: {exc}")
        return None


@tool
def music_agent_tool(request: str) -> str:
    """Control Spotify music playback.

    Use for all music requests:
    - Play a specific song or artist
    - Pause or stop playback
    - Check what is currently playing

    Input: Full natural-language music request with song and artist details.
    Example: "Play Taylor Swift - Love Story"
    Example: "Pause the music"
    Example: "What song is currently playing?"
    """
    global _agent
    if _agent is None:
        _agent = _build()
    if _agent is None:
        return "Music agent is currently unavailable."
    try:
        logger.info(f"[music_agent] {request[:80]}")
        result = _agent.invoke({"messages": [{"role": "user", "content": request}]})
        return result["messages"][-1].content
    except Exception as exc:
        logger.error(f"[music_agent] error: {exc}", exc_info=True)
        return f"Music control failed: {exc}"
