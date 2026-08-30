# ABOUTME: Spotify integration for playback control and track information
# ABOUTME: Uses Spotipy library with OAuth authentication

import os
import re
import string
from typing import Any, List, Optional

import spotipy
from langchain.tools import tool
from spotipy.oauth2 import CacheFileHandler, SpotifyOAuth

from config import Config
from logger import logger

_spotify_client: Optional[spotipy.Spotify] = None

SPOTIFY_CACHE_PATH = "keys/.spotify_cache"
SPOTIFY_AUTH_REQUIRED_MESSAGE = (
    "Spotify is not connected. Open Agent Controls, connect Spotify, then retry."
)


class SpotifyAuthenticationRequired(RuntimeError):
    """Raised when a playback request has no cached Spotify authorization."""


def _spotify_auth_manager() -> SpotifyOAuth:
    """Build the shared OAuth configuration without prompting for input."""
    if not Config.SPOTIFY_CLIENT_ID or not Config.SPOTIFY_CLIENT_SECRET:
        raise ValueError("Spotify credentials are not configured.")

    cache_dir = os.path.dirname(SPOTIFY_CACHE_PATH)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    return SpotifyOAuth(
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


def _cached_token(auth_manager: SpotifyOAuth) -> Any:
    """Read and refresh the cache through Spotipy's current API."""
    return auth_manager.validate_token(auth_manager.cache_handler.get_cached_token())


def spotify_is_connected() -> bool:
    """Return whether playback can proceed without interactive OAuth."""
    if _spotify_client is not None:
        return True
    try:
        return bool(_cached_token(_spotify_auth_manager()))
    except ValueError:
        return False


def spotify_authorization_url() -> str:
    """Return the one-time authorization URL for the control panel."""
    return _spotify_auth_manager().get_authorize_url()


def complete_spotify_authorization(redirect_url: str) -> None:
    """Store Spotify authorization from a callback URL without terminal input."""
    global _spotify_client

    auth_manager = _spotify_auth_manager()
    code = auth_manager.parse_response_code(redirect_url.strip())
    if not code:
        raise ValueError("Paste the full Spotify callback URL containing the code.")
    auth_manager.get_access_token(code, as_dict=False)
    _spotify_client = spotipy.Spotify(auth_manager=auth_manager)


def _get_spotify_client() -> spotipy.Spotify:
    """Get authenticated Spotify client (singleton)."""
    global _spotify_client

    if _spotify_client is None:
        logger.debug("Initializing Spotify client with OAuth")
        auth_manager = _spotify_auth_manager()
        token_info = _cached_token(auth_manager)
        if not token_info:
            raise SpotifyAuthenticationRequired(SPOTIFY_AUTH_REQUIRED_MESSAGE)

        _spotify_client = spotipy.Spotify(auth_manager=auth_manager)
        logger.info("Spotify client initialized successfully")

    return _spotify_client


def _spotify_items(response: Any, collection: str = "items") -> List[dict[str, Any]]:
    """Return usable Spotify objects, ignoring unavailable null entries."""
    if not isinstance(response, dict):
        return []
    source = response if collection == "items" else response.get(collection)
    if isinstance(source, dict):
        source = source.get("items") or []
    return (
        [item for item in source if isinstance(item, dict)]
        if isinstance(source, list)
        else []
    )


def _get_active_device(sp: spotipy.Spotify) -> Optional[str]:
    """Get the ID of an active Spotify device. Prefers active, falls back to first available."""
    try:
        devices = _spotify_items(sp.devices(), "devices")
        logger.debug(f"Available Spotify devices: {devices}")

        if not devices:
            logger.warning("No Spotify devices found")
            return None

        for device in devices:
            if device.get("is_active"):
                logger.info(
                    f"Using active device: {device.get('name')} ({device.get('type')})"
                )
                return device["id"]

        first_device = devices[0]
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


@tool
def search_spotify_catalog(query: str) -> str:
    """Search Spotify's public catalogue for tracks, playlists, albums, and artists."""
    query = query.strip()
    if not query:
        return "Please say what you would like to search for on Spotify."
    sp = _get_spotify_client()
    results = sp.search(q=query, type="track,playlist,album,artist", limit=5)
    sections: list[str] = []

    for key, heading in (
        ("playlists", "Playlists"),
        ("tracks", "Tracks"),
        ("albums", "Albums"),
        ("artists", "Artists"),
    ):
        items = _spotify_items(results, key)
        if not items:
            continue
        names = []
        for item in items:
            name = item.get("name", "Untitled")
            artists = ", ".join(
                artist.get("name", "") for artist in item.get("artists", [])
            )
            names.append(f"{name} — {artists}" if artists else name)
        sections.append(f"{heading}:\n- " + "\n- ".join(names))

    return "\n\n".join(sections) or f"No Spotify results found for: {query}"


@tool
def play_playlist(query: str, device: str) -> str:
    """Play a named personal playlist on an exact device, or '' for the default."""
    sp = _get_spotify_client()
    playlist_name = query.strip(" '\"")
    if not playlist_name:
        return "Please tell me which playlist you would like to play."

    playlists = _spotify_items(sp.current_user_playlists(limit=50))
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

    devices = _spotify_items(sp.devices(), "devices")
    device_id = _match_device(device, devices) if device else None
    if device and device_id is None:
        available = ", ".join(d.get("name", "") for d in devices)
        return (
            f"I couldn't find a device matching '{device}'. "
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
def play_spotify_playlist(query: str, device: str) -> str:
    """Play a public playlist on an exact device, or '' for the default."""
    sp = _get_spotify_client()
    results = sp.search(q=query, type="playlist", limit=5)
    playlists = _spotify_items(results, "playlists")
    if not playlists:
        return f"No public Spotify playlists found for: {query}"

    devices = _spotify_items(sp.devices(), "devices")
    device_id = _match_device(device, devices) if device else None
    if device and device_id is None:
        available = ", ".join(d.get("name", "") for d in devices)
        return (
            f"I couldn't find a device matching '{device}'. "
            f"Available devices: {available or 'none found'}."
        )
    if device_id is None:
        device_id = _get_active_device(sp)
    if device_id is None:
        return "No active Spotify device found. Please open Spotify and try again."

    playlist = playlists[0]
    sp.start_playback(device_id=device_id, context_uri=playlist["uri"])
    logger.info("✓ Playing Spotify catalogue playlist: %s", playlist["name"])
    return f"Now playing Spotify playlist: {playlist['name']}"


@tool
def play_track(title: str, artist: str, device: str) -> str:
    """
    Search for a track on Spotify and play it. Use this to play music.

    Args:
        title: Exact song title
        artist: Exact artist name, or an empty string when unknown
        device: Exact requested device name, or an empty string for the default

    Returns:
        Confirmation message with track name and artist
    """

    sp = _get_spotify_client()
    devices = _spotify_items(sp.devices(), "devices")
    logger.debug("Track title='%s', artist='%s', device='%s'", title, artist, device)

    device_id = None
    if device:
        device_id = _match_device(device, devices)
        if device_id is None:
            available = ", ".join(d.get("name", "") for d in devices)
            return (
                f"I couldn't find a device matching '{device}'. "
                f"Available devices: {available or 'none found'}. "
                "Try again with one of those names, or ask me to play on the default device."
            )

    search_query = f"track:{title} artist:{artist}" if artist else title
    logger.debug(f"Spotify search query: {search_query!r}")

    results = sp.search(q=search_query, type="track", limit=5)
    items = _spotify_items(results, "tracks")

    if not items:
        logger.warning(f"No tracks found for query: {search_query!r}")
        return f"No tracks found for: {title}"

    best = None
    best_score = -1
    for track in items:
        score = _score_candidate(track, title, artist or None)
        if score > best_score:
            best_score = score
            best = track

    if best is None:
        return f"No tracks found for: {title}"

    track_uri = best.get("uri")
    track_name = best.get("name")
    artist_name = best.get("artists", [{}])[0].get("name", "")
    logger.info(f"Selected track: {track_name} by {artist_name} (score={best_score})")

    # Never execute a search result that has no title overlap with the request.
    if best_score < 1 or (artist and best_score < 3):
        return (
            f"I found '{track_name}' by {artist_name}, but I'm not confident it matches '{title}'. "
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
def resume_playback(device: str) -> str:
    """Resume the existing queue on an exact device, or '' for the default."""
    sp = _get_spotify_client()
    devices = _spotify_items(sp.devices(), "devices")
    device_id = _match_device(device, devices) if device else None
    if device and device_id is None:
        available = ", ".join(d.get("name", "") for d in devices)
        return (
            f"I couldn't find a device matching '{device}'. "
            f"Available devices: {available or 'none found'}."
        )
    if device_id is None:
        device_id = _get_active_device(sp)
    if device_id is None:
        return "No active Spotify device found. Please open Spotify and try again."

    sp.start_playback(device_id=device_id)
    logger.info("Spotify playback resumed")
    return "Spotify playback resumed."


@tool
def set_volume(volume_percent: int, device: str) -> str:
    """Set volume on an exact Spotify device, or '' for the default."""
    if not 0 <= volume_percent <= 100:
        return "Spotify volume must be between 0 and 100."

    sp = _get_spotify_client()
    devices = _spotify_items(sp.devices(), "devices")
    device_id = _match_device(device, devices) if device else None
    if device and device_id is None:
        playback = sp.current_playback() or {}
        active_device = playback.get("device") if isinstance(playback, dict) else None
        device_id = _match_device(device, [active_device]) if active_device else None
    if device and device_id is None:
        available = ", ".join(d.get("name", "") for d in devices)
        return (
            f"I couldn't find a device matching '{device}'. "
            f"Available devices: {available or 'none found'}."
        )
    if device_id is None:
        device_id = _get_active_device(sp)
    if device_id is None:
        return "No active Spotify device found. Please open Spotify and try again."

    try:
        sp.volume(volume_percent=volume_percent, device_id=device_id)
    except Exception as exc:
        if "VOLUME_CONTROL_DISALLOW" in str(exc):
            return (
                "Spotify cannot control volume on that device. "
                "Please adjust it directly on the device."
            )
        logger.warning("Spotify volume update failed: %s", exc)
        return (
            "Spotify could not set the volume. Please adjust it directly on the device."
        )
    return f"Spotify volume set to {volume_percent}%."


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
        playback = get_current_playback_snapshot()
        if not playback:
            logger.debug("Nothing currently playing on Spotify")
            return "Nothing is currently playing"

        status = playback["status"]
        logger.debug(
            "Current Spotify track: %s by %s (%s)",
            playback["name"],
            playback["artist"],
            status,
        )
        return (
            f"{status}: {playback['name']} by {playback['artist']} "
            f"(Album: {playback['album']})"
        )

    except Exception as e:
        logger.error(f"Error getting current Spotify track: {e}", exc_info=True)
        return f"Error getting current track: {str(e)}"


def get_current_playback_snapshot() -> Optional[dict[str, Any]]:
    """Return verified Spotify playback data for UI rendering."""
    try:
        current = _get_spotify_client().current_playback()
        if not isinstance(current, dict) or not isinstance(current.get("item"), dict):
            return None

        track = current["item"]
        album = track.get("album") if isinstance(track.get("album"), dict) else {}
        artists = track.get("artists") if isinstance(track.get("artists"), list) else []
        artist_names = [
            artist.get("name", "")
            for artist in artists
            if isinstance(artist, dict) and artist.get("name")
        ]
        raw_images = album.get("images")
        images: list[Any] = raw_images if isinstance(raw_images, list) else []
        cover_url = next(
            (
                image.get("url")
                for image in images
                if isinstance(image, dict) and image.get("url")
            ),
            None,
        )
        device = (
            current.get("device") if isinstance(current.get("device"), dict) else {}
        )
        return {
            "status": "Playing" if current.get("is_playing") else "Paused",
            "name": str(track.get("name") or "Unknown track"),
            "artist": ", ".join(artist_names) or "Unknown artist",
            "album": str(album.get("name") or "Unknown album"),
            "cover_url": cover_url,
            "device": str(device.get("name") or "Spotify"),
            "uri": track.get("uri"),
        }
    except Exception as exc:
        logger.warning("Unable to read current Spotify playback: %s", exc)
        return None


def get_spotify_tools() -> List[Any]:
    """Get all Spotify tools as a list for agent registration."""
    return [
        search_spotify_catalog,
        resume_playback,
        play_track,
        play_playlist,
        play_spotify_playlist,
        set_volume,
        pause_playback,
        current_track,
    ]
