# ABOUTME: Music / Spotify sub-agent — playback control
# ABOUTME: Wraps tools/spotify_tool.py; exposed to supervisor as a single @tool

from typing import Any

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool

from config import Config
from core.subagents.music.tools import get_spotify_tools
from logger import logger

_SYSTEM_PROMPT = """\
Act as a specialist Spotify music control agent to seamlessly manage playback using the available tools.

Your primary objective is to accurately interpret user music requests and execute the corresponding Spotify controls.

# Critical Guidelines
- ALWAYS use the full song name AND artist name when searching for or playing a track.
- Never use partial names if the full context is available in the request.

# Steps
1. Analyze the user's request to identify the intended playback action (play, pause, next, etc.).
2. Extract the specific song name and artist name from the request if applicable.
3. Construct the tool call. For `play_track`, ensure the query combines both the song and the artist (e.g., "Taylor Swift - Love Story").
4. Execute the tool and verify the result.
5. Provide a brief, clear confirmation of the action taken.

# Output Format
Provide a concise text response confirming what was done. Do not output JSON.

# Examples
## Example 1: Playing a Track
**Input:** "Play Love Story by Taylor Swift"

**Output:**
**Reasoning:** The user wants to play a specific song. The song is "Love Story" and the artist is "Taylor Swift". I will call the play_track tool with the combined query "Taylor Swift - Love Story".
**Confirmation:** Playing "Love Story" by Taylor Swift on Spotify.
"""

_agent = None


def _build() -> Any:
    try:

        return create_agent(
            model=init_chat_model(
                api_key=Config.AI_KEY,
                model=Config.TOOL_MODEL,
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
        logger.error(f"[music_agent] error: {exc}", exc_info=True)
        return f"Music control failed: {exc}"
