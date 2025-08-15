# Spotify integration using Spotipy
import spotipy
from spotipy.oauth2 import SpotifyOAuth

from langchain.agents import Tool


class SpotifyTool(Tool):
    def __init__(self):
        self.sp = spotipy.Spotify(
            auth_manager=SpotifyOAuth(
                client_id="YOUR_CLIENT_ID",
                client_secret="YOUR_CLIENT_SECRET",
                redirect_uri="YOUR_REDIRECT_URI",
                scope="user-library-read user-read-playback-state user-modify-playback-state",
            )
        )

    def play_song(self, song_name):
        results = self.sp.search(q=song_name, limit=1)
        if results["tracks"]["items"]:
            track_uri = results["tracks"]["items"][0]["uri"]
            self.sp.start_playback(uris=[track_uri])
            return f"Playing {results['tracks']['items'][0]['name']}"
        return "Song not found"
