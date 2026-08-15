# ABOUTME: Spotify integration for playback control and track information
# ABOUTME: Uses Spotipy library with OAuth authentication

import os
import re
import string
from typing import Any, List, Literal, Optional, Tuple

import spotipy
from langchain.tools import tool
from spotipy.oauth2 import CacheFileHandler, SpotifyOAuth

from config import Config
from logger import logger

_spotify_client: Optional[spotipy.Spotify] = None

SPOTIFY_CACHE_PATH = "keys/.spotify_cache"


def _get_spotify_client() -> spotipy.Spotify:
    """Get authenticated Spotify client (singleton)."""
    global _spotify_client

    if _spotify_client is None:
        logger.debug("Initializing Spotify client with OAuth")
        cache_dir = os.path.dirname(SPOTIFY_CACHE_PATH)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        auth_manager = SpotifyOAuth(
            client_id=Config.SPOTIFY_CLIENT_ID,
            client_secret=Config.SPOTIFY_CLIENT_SECRET,
            redirect_uri="http://127.0.0.1:8100/callback",
            scope=(
                "user-modify-playback-state user-read-playback-state "
                "user-read-currently-playing playlist-read-private "
                "playlist-read-collaborative"
            ),
            cache_handler=CacheFileHandler(cache_path=SPOTIFY_CACHE_PATH),
            open_browser=False,
        )

        token_info = auth_manager.get_cached_token()
        if token_info:
            logger.info("Spotify token loaded from cache (auto-refresh enabled)")

        if not token_info:
            auth_url = auth_manager.get_authorize_url()
            logger.info("Spotify authentication required - prompting user for OAuth")
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
    """Get the ID of an active Spotify device. Prefers active, falls back to first available."""
    try:
        devices = sp.devices()
        logger.debug(f"Available Spotify devices: {devices}")

        if not devices or not devices.get("devices"):
            logger.warning("No Spotify devices found")
            return None

        for device in devices["devices"]:
            if device.get("is_active"):
                logger.info(
                    f"Using active device: {device.get('name')} ({device.get('type')})"
                )
                return device["id"]

        first_device = devices["devices"][0]
        logger.info(
            f"No active device, using first available: {first_device.get('name')}"
        )
        return first_device["id"]

    except Exception as e:
        logger.error(f"Error getting Spotify devices: {e}", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Text/normalization helpers
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    table = str.maketrans("", "", string.punctuation)
    return re.sub(r"\s+", " ", text.translate(table)).strip().lower()


def _extract_music_and_device(query: str) -> Tuple[str, Optional[str]]:
    """Split user query into music part and optional device phrase."""

    separators = [" on my ", " on the ", " to my ", " to the ", " on "]
    lowered = query.lower()

    for sep in separators:
        if sep in lowered:
            idx = lowered.rfind(sep)
            music_part = query[:idx].strip()
            device_part = query[idx + len(sep) :].strip()
            return music_part, device_part

    return query.strip(), None


def _match_device(device_query: str, devices: List[dict[str, Any]]) -> Optional[str]:
    """Deterministic device match by normalized substring (case/punct insensitive)."""

    target = _normalize(device_query)
    for d in devices:
        name_norm = _normalize(d.get("name", ""))
        if not name_norm:
            continue
        if target in name_norm or name_norm in target:
            return d.get("id")
    return None


def _split_title_artist(query: str) -> Tuple[str, Optional[str]]:
    if " by " in query.lower():
        parts = re.split(r"\s+by\s+", query, maxsplit=1, flags=re.IGNORECASE)
        return parts[0].strip(), parts[1].strip()
    return query.strip(), None


def _build_track_query(music_query: str) -> Tuple[str, str, Optional[str]]:
    """Return (search_query, req_title, req_artist)."""
    title, artist = _split_title_artist(music_query)
    if artist:
        return f"track:{title} artist:{artist}", title, artist
    return music_query, title, artist


def _score_candidate(track: dict, req_title: str, req_artist: Optional[str]) -> int:
    title_norm = _normalize(req_title)
    candidate_title = _normalize(track.get("name", ""))
    title_exact = int(bool(title_norm) and candidate_title == title_norm)
    title_close = int(
        bool(title_norm)
        and (title_norm in candidate_title or candidate_title in title_norm)
    )

    artist_score = 0
    if req_artist:
        req_artist_norm = _normalize(req_artist)
        for artist in track.get("artists", []):
            name_norm = _normalize(artist.get("name", ""))
            if name_norm == req_artist_norm:
                artist_score = 2
                break
            if req_artist_norm in name_norm or name_norm in req_artist_norm:
                artist_score = max(artist_score, 1)

    return title_exact * 3 + title_close * 1 + artist_score


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def _playlist_name(music_query: str) -> str:
    name = re.sub(r"^play\s+(?:my\s+|the\s+)?", "", music_query, flags=re.I)
    name = re.sub(r"\s+playlist$", "", name, flags=re.I)
    return name.strip(" '\"")


@tool
def play_playlist(query: str) -> str:
    """Find one of the user's Spotify playlists by name and play it."""
    sp = _get_spotify_client()
    music_query, device_query = _extract_music_and_device(query)
    playlist_name = _playlist_name(music_query)
    if not playlist_name:
        return "Please tell me which playlist you would like to play."

    playlists = sp.current_user_playlists(limit=50).get("items", [])
    target = _normalize(playlist_name)
    exact = [p for p in playlists if _normalize(p.get("name", "")) == target]
    close = [
        p
        for p in playlists
        if target in _normalize(p.get("name", ""))
        or _normalize(p.get("name", "")) in target
    ]
    matches = exact or close
    if len(matches) != 1:
        available = ", ".join(p.get("name", "") for p in playlists[:10])
        return (
            f"I couldn't find a unique playlist named '{playlist_name}'. "
            f"Available playlists: {available or 'none found'}."
        )

    devices = sp.devices().get("devices", [])
    device_id = _match_device(device_query, devices) if device_query else None
    if device_query and device_id is None:
        available = ", ".join(d.get("name", "") for d in devices)
        return (
            f"I couldn't find a device matching '{device_query}'. "
            f"Available devices: {available or 'none found'}."
        )
    if device_id is None:
        device_id = _get_active_device(sp)
    if device_id is None:
        return "No active Spotify device found. Please open Spotify and try again."

    playlist = matches[0]
    sp.start_playback(device_id=device_id, context_uri=playlist["uri"])
    logger.info("✓ Playing playlist: %s", playlist["name"])
    return f"Now playing playlist: {playlist['name']}"


@tool
def play_track(query: str) -> str:
    """
    Search for a track on Spotify and play it. Use this to play music.

    Args:
        query: Song name, artist, or search query (may include a device target)

    Returns:
        Confirmation message with track name and artist
    """

    if "playlist" in query.lower():
        return play_playlist.invoke({"query": query})

    sp = _get_spotify_client()
    devices = sp.devices().get("devices", [])

    music_query, device_query = _extract_music_and_device(query)
    logger.debug(
        f"Parsed music query='{music_query}', device query='{device_query or ''}'"
    )

    device_id = None
    if device_query:
        device_id = _match_device(device_query, devices)
        if device_id is None:
            available = ", ".join(d.get("name", "") for d in devices)
            return (
                f"I couldn't find a device matching '{device_query}'. "
                f"Available devices: {available or 'none found'}. "
                "Try again with one of those names, or ask me to play on the default device."
            )

    search_query, req_title, req_artist = _build_track_query(music_query)
    logger.debug(f"Spotify search query: {search_query!r}")

    results = sp.search(q=search_query, type="track", limit=5)
    items = results.get("tracks", {}).get("items", [])

    if not items:
        logger.warning(f"No tracks found for query: {search_query!r}")
        return f"No tracks found for: {music_query}"

    best = None
    best_score = -1
    for track in items:
        score = _score_candidate(track, req_title, req_artist)
        if score > best_score:
            best_score = score
            best = track

    if best is None:
        return f"No tracks found for: {music_query}"

    track_uri = best.get("uri")
    track_name = best.get("name")
    artist_name = best.get("artists", [{}])[0].get("name", "")
    logger.info(f"Selected track: {track_name} by {artist_name} (score={best_score})")

    # If the user specified an artist, require a stronger match; otherwise play the best title match.
    if req_artist and best_score < 3:
        return (
            f"I found '{track_name}' by {artist_name}, but I'm not confident it matches '{music_query}'. "
            "Should I play it?"
        )

    if device_id is None:
        device_id = _get_active_device(sp)
        if device_id is None:
            return (
                "No active Spotify device found. "
                "Please open Spotify on your phone, computer, or web player and try again."
            )

    sp.start_playback(device_id=device_id, uris=[track_uri])
    logger.info(f"✓ Playing: {track_name} by {artist_name}")
    return f"Now playing: {track_name} by {artist_name}"


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
    """Get all Spotify tools as a list for agent registration."""
    return [play_track, play_playlist, pause_playback, current_track]
