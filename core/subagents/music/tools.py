# ABOUTME: Spotify integration for playback control and track information
# ABOUTME: Uses Spotipy library with OAuth authentication

from typing import Any, List, Literal, Optional

import spotipy
from langchain.tools import tool
from spotipy.oauth2 import SpotifyOAuth

from config import Config
from logger import logger

_spotify_client: Optional[spotipy.Spotify] = None

from spotipy.oauth2 import CacheFileHandler
import os

SPOTIFY_CACHE_PATH = "keys/.spotify_cache"


def _get_spotify_client() -> spotipy.Spotify:
    """Get authenticated Spotify client (singleton).

    Token Management:
    - Access tokens expire after ~1 hour
    - Refresh tokens are long-lived and stored in .spotify_cache
    - SpotifyOAuth automatically refreshes expired access tokens
    - Re-authentication only required if cache is deleted or refresh token revoked
    """
    global _spotify_client

    if _spotify_client is None:
        logger.debug("Initializing Spotify client with OAuth")

        # OAuth Configuration:
        # - redirect_uri: http://127.0.0.1:8100/callback (port 8100 avoids Streamlit conflict)
        # - Uses 127.0.0.1 instead of localhost (Spotify's new requirement as of April 2025)
        # - cache_path: .spotify_cache stores refresh token for automatic token renewal
        # - open_browser: False (manual OAuth flow works better in WSL environment)
        cache_dir = os.path.dirname(SPOTIFY_CACHE_PATH)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        auth_manager = SpotifyOAuth(
            client_id=Config.SPOTIFY_CLIENT_ID,
            client_secret=Config.SPOTIFY_CLIENT_SECRET,
            redirect_uri="http://127.0.0.1:8100/callback",
            scope="user-modify-playback-state user-read-playback-state user-read-currently-playing",
            # cache_path=SPOTIFY_CACHE_PATH,
            cache_handler=CacheFileHandler(cache_path=SPOTIFY_CACHE_PATH),
            open_browser=False,
        )

        # Check if we need to authenticate
        # get_cached_token() automatically refreshes expired tokens using refresh token
        token_info = auth_manager.get_cached_token()

        if token_info:
            logger.info("Spotify token loaded from cache (auto-refresh enabled)")

        if not token_info:
            auth_url = auth_manager.get_authorize_url()
            logger.info("Spotify authentication required - prompting user for OAuth")

            # This will wait for the callback - user needs to paste the redirect URL when prompted
            print("\n" + "=" * 80)
            print("SPOTIFY AUTHENTICATION REQUIRED")
            print("=" * 80)
            print(f"\n1. Open this URL in your browser:\n{auth_url}\n")
            print("2. After authorizing, you'll be redirected to 127.0.0.1:8100")
            print("3. Copy the FULL URL from your browser's address bar")
            print("4. Paste it here and press Enter\n")
            print("=" * 80 + "\n")

            redirect_url = input("Paste the redirect URL: ").strip()
            code = auth_manager.parse_response_code(redirect_url)
            token_info = auth_manager.get_access_token(code)

        _spotify_client = spotipy.Spotify(auth_manager=auth_manager)
        logger.info("Spotify client initialized successfully")

    return _spotify_client


def _get_active_device(sp: spotipy.Spotify) -> Optional[str]:
    """Get the ID of an active Spotify device.

    Prefers active devices, falls back to first available device.
    Returns None if no devices found.
    """
    try:
        devices = sp.devices()
        logger.debug(f"Available Spotify devices: {devices}")

        if not devices or not devices.get("devices"):
            logger.warning("No Spotify devices found")
            return None

        # Prefer active device
        for device in devices["devices"]:
            if device.get("is_active"):
                logger.info(
                    f"Using active device: {device.get('name')} ({device.get('type')})"
                )
                return device["id"]

        # Fall back to first available device
        first_device = devices["devices"][0]
        logger.info(
            f"No active device, using first available: {first_device.get('name')} ({first_device.get('type')})"
        )
        return first_device["id"]

    except Exception as e:
        logger.error(f"Error getting Spotify devices: {e}", exc_info=True)
        return None


@tool
def play_track(query: str) -> str:
    """
    Search for a track on Spotify and play it. Use this to play music.

    Args:
        query: Song name, artist, or search query

    Returns:
        Confirmation message with track name and artist
    """
    try:
        logger.debug(f"Searching Spotify for: {query}")
        sp = _get_spotify_client()

        # Search for track
        logger.debug("Calling Spotify search API")
        results = sp.search(q=query, type="track", limit=1)

        if not results["tracks"]["items"]:
            logger.warning(f"No tracks found for query: {query}")
            return f"No tracks found for: {query}"

        track = results["tracks"]["items"][0]
        track_uri = track["uri"]
        track_name = track["name"]
        artist_name = track["artists"][0]["name"]

        logger.info(f"Found track: {track_name} by {artist_name}")

        # Get active device
        device_id = _get_active_device(sp)
        if not device_id:
            logger.error("No active Spotify device found")
            return "No active Spotify device found. Please open Spotify on your phone, computer, or web player and try again."

        # Play track on device
        logger.debug(f"Starting playback on device {device_id}")
        sp.start_playback(device_id=device_id, uris=[track_uri])
        logger.info(f"✓ Playing: {track_name} by {artist_name}")

        return f"Now playing: {track_name} by {artist_name}"

    except spotipy.exceptions.SpotifyException as e:
        error_msg = str(e)
        logger.error(f"Spotify API error: {error_msg}", exc_info=True)

        if "Premium required" in error_msg or "PREMIUM_REQUIRED" in error_msg:
            return "Spotify Premium is required to control playback"
        elif "NO_ACTIVE_DEVICE" in error_msg:
            return "No active Spotify device. Please open Spotify and try again"
        else:
            return f"Spotify error: {error_msg}"

    except Exception as e:
        logger.error(
            f"Unexpected error playing Spotify track '{query}': {e}", exc_info=True
        )
        return f"Error playing track: {str(e)}"


@tool
def pause_playback() -> str:
    """
    Pause current Spotify playback. Use this to pause music.

    Returns:
        Confirmation message
    """
    try:
        logger.debug("Pausing Spotify playback")
        sp = _get_spotify_client()
        sp.pause_playback()
        logger.info("Spotify playback paused")
        return "Playback paused"

    except Exception as e:
        logger.error(f"Error pausing Spotify playback: {e}", exc_info=True)
        return f"Error pausing playback: {str(e)}"


@tool
def current_track() -> str:
    """
    Get information about the currently playing track on Spotify.

    Returns:
        String with current track name, artist, and album
    """
    try:
        logger.debug("Getting current Spotify track")
        sp = _get_spotify_client()
        current = sp.current_playback()

        if not current or not current.get("item"):
            logger.debug("Nothing currently playing on Spotify")
            return "Nothing is currently playing"

        track = current["item"]
        track_name = track["name"]
        artist_name = track["artists"][0]["name"]
        album_name = track["album"]["name"]
        is_playing = current["is_playing"]

        status: Literal["Playing"] | Literal["Paused"] = (
            "Playing" if is_playing else "Paused"
        )
        logger.debug(f"Current Spotify track: {track_name} by {artist_name} ({status})")

        return f"{status}: {track_name} by {artist_name} (Album: {album_name})"

    except Exception as e:
        logger.error(f"Error getting current Spotify track: {e}", exc_info=True)
        return f"Error getting current track: {str(e)}"


def get_spotify_tools() -> List[Any]:
    """
    Get all Spotify tools as a list for agent registration.

    Returns:
        List of LangChain tools
    """
    return [play_track, pause_playback, current_track]
