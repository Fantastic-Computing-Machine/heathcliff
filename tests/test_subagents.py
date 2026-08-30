# ABOUTME: Comprehensive tests for core/subagents/ — each domain agent wrapper
# ABOUTME: Tests tool registration, lazy init, graceful degradation, tool metadata

import os
import sys
from unittest.mock import Mock, patch

import pytest
from langgraph.errors import GraphRecursionError

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Individual tool metadata
# ---------------------------------------------------------------------------


class TestToolMetadata:
    @pytest.mark.parametrize(
        "tool_name,module_path",
        [
            ("info_agent_tool", "core.subagents.info.agent"),
            ("music_agent_tool", "core.subagents.music.agent"),
            ("email_agent_tool", "core.subagents.email.agent"),
            ("calendar_agent_tool", "core.subagents.calendar.agent"),
            ("contacts_agent_tool", "core.subagents.contacts.agent"),
            ("comms_agent_tool", "core.subagents.comms.agent"),
        ],
    )
    def test_tool_exported_from_module(self, tool_name, module_path):
        """Each agent tool is importable from its own module."""
        import importlib

        mod = importlib.import_module(module_path)
        tool = getattr(mod, tool_name)
        assert tool.name == tool_name

    def test_info_agent_description_mentions_search(self):
        from core.subagents.info.agent import info_agent_tool

        desc = info_agent_tool.description.lower()
        assert any(
            kw in desc for kw in ["search", "research", "weather", "news", "wikipedia"]
        )

    def test_music_agent_description_mentions_spotify(self):
        from core.subagents.music.agent import music_agent_tool

        desc = music_agent_tool.description.lower()
        assert any(kw in desc for kw in ["spotify", "music", "play", "pause"])

    def test_email_agent_description_mentions_gmail(self):
        from core.subagents.email.agent import email_agent_tool

        desc = email_agent_tool.description.lower()
        assert any(kw in desc for kw in ["email", "gmail", "send", "draft"])

    def test_email_agent_description_requires_address(self):
        """Email agent description must warn about needing email address."""
        from core.subagents.email.agent import email_agent_tool

        desc = email_agent_tool.description.lower()
        assert "address" in desc or "email" in desc

    def test_calendar_agent_description_mentions_events(self):
        from core.subagents.calendar.agent import calendar_agent_tool

        desc = calendar_agent_tool.description.lower()
        assert any(kw in desc for kw in ["calendar", "event", "schedule", "meeting"])

    def test_contacts_agent_description_mentions_contacts(self):
        from core.subagents.contacts.agent import contacts_agent_tool

        desc = contacts_agent_tool.description.lower()
        assert any(kw in desc for kw in ["contact", "email", "phone", "lookup"])

    def test_contacts_description_mentions_fallback_behavior(self):
        """Contacts tool must describe what to do if contact not found."""
        from core.subagents.contacts.agent import contacts_agent_tool

        desc = contacts_agent_tool.description.lower()
        assert "not found" in desc or "ask" in desc or "provide" in desc

    def test_comms_agent_description_mentions_telegram(self):
        from core.subagents.comms.agent import comms_agent_tool

        desc = comms_agent_tool.description.lower()
        assert any(kw in desc for kw in ["telegram", "message", "notification"])


# ---------------------------------------------------------------------------
# Music device selection and fallback behaviour
# ---------------------------------------------------------------------------


class TestMusicDeviceSelection:
    def _mock_spotify(self, devices):
        sp = Mock()
        sp.search = Mock()
        sp.devices.return_value = {"devices": devices}
        sp.start_playback = Mock()
        return sp

    def test_play_track_prompts_when_device_missing(self, monkeypatch):
        import core.subagents.music.tools as music_tools

        sp = self._mock_spotify(
            [
                {
                    "id": "1",
                    "name": "Office Speaker",
                    "type": "Speaker",
                    "is_active": True,
                }
            ]
        )
        monkeypatch.setattr(music_tools, "_get_spotify_client", lambda: sp)

        # Search returns a wrong top candidate to exercise the fallback
        sp.search.return_value = {
            "tracks": {
                "items": [
                    {
                        "uri": "spotify:track:oops",
                        "name": "Vanishing Point",
                        "artists": [{"name": "logical_miracle"}],
                    }
                ]
            }
        }

        resp = music_tools.play_track.invoke(
            {"title": "Royals", "artist": "Lorde", "device": "Echo Dot"}
        )

        assert "couldn't find a device" in resp.lower()
        sp.start_playback.assert_not_called()

    def test_play_track_uses_requested_device_when_found(self, monkeypatch):
        import core.subagents.music.tools as music_tools

        sp = self._mock_spotify(
            [{"id": "2", "name": "Echo Dot", "type": "Speaker", "is_active": False}]
        )
        monkeypatch.setattr(music_tools, "_get_spotify_client", lambda: sp)

        sp.search.return_value = {
            "tracks": {
                "items": [
                    {
                        "uri": "spotify:track:123",
                        "name": "Royals",
                        "artists": [{"name": "Lorde"}],
                    }
                ]
            }
        }

        resp = music_tools.play_track.invoke(
            {"title": "Royals", "artist": "", "device": "Echo Dot"}
        )

        assert "now playing: royals by lorde" in resp.lower()
        sp.start_playback.assert_called_once_with(
            device_id="2", uris=["spotify:track:123"]
        )

    def test_track_search_receives_music_separately_from_device(self, monkeypatch):
        import core.subagents.music.tools as music_tools

        sp = self._mock_spotify(
            [{"id": "2", "name": "Echo Dot", "type": "Speaker", "is_active": False}]
        )
        monkeypatch.setattr(music_tools, "_get_spotify_client", lambda: sp)

        sp.search.return_value = {
            "tracks": {
                "items": [
                    {
                        "uri": "spotify:track:123",
                        "name": "Royals",
                        "artists": [{"name": "Lorde"}],
                    }
                ]
            }
        }

        music_tools.play_track.invoke(
            {"title": "Royals", "artist": "Lorde", "device": "Echo Dot"}
        )

        called_query = sp.search.call_args[1]["q"]
        assert called_query == "track:Royals artist:Lorde"

    def test_play_playlist_uses_playlist_context_not_track_search(self, monkeypatch):
        import core.subagents.music.tools as music_tools

        sp = self._mock_spotify(
            [{"id": "pixel", "name": "Pixel 9 Pro", "is_active": True}]
        )
        sp.current_user_playlists.return_value = {
            "items": [{"name": "Funn", "uri": "spotify:playlist:funn"}]
        }
        monkeypatch.setattr(music_tools, "_get_spotify_client", lambda: sp)

        response = music_tools.play_playlist.invoke(
            {"query": "Funn", "device": "Pixel 9 Pro"}
        )

        assert response == "Now playing playlist: Funn"
        sp.search.assert_not_called()
        sp.start_playback.assert_called_once_with(
            device_id="pixel", context_uri="spotify:playlist:funn"
        )

    def test_catalog_search_returns_public_tracks_and_playlists(self, monkeypatch):
        import core.subagents.music.tools as music_tools

        tool_names = {tool.name for tool in music_tools.get_spotify_tools()}
        assert {
            "search_spotify_catalog",
            "play_spotify_playlist",
            "set_volume",
        } <= tool_names

        sp = self._mock_spotify([])
        sp.search.return_value = {
            "playlists": {"items": [None, {"name": "Hindi Lo-Fi", "artists": []}]},
            "tracks": {
                "items": [
                    None,
                    {
                        "name": "Kesariya",
                        "artists": [{"name": "Pritam"}],
                    },
                ]
            },
            "albums": {"items": []},
            "artists": {"items": []},
        }
        monkeypatch.setattr(music_tools, "_get_spotify_client", lambda: sp)

        response = music_tools.search_spotify_catalog.invoke({"query": "Hindi lo-fi"})

        assert "Hindi Lo-Fi" in response
        assert "Kesariya — Pritam" in response
        sp.search.assert_called_once_with(
            q="Hindi lo-fi", type="track,playlist,album,artist", limit=5
        )
        sp.start_playback.assert_not_called()

    def test_play_public_playlist_and_set_requested_volume(self, monkeypatch):
        import core.subagents.music.tools as music_tools

        sp = self._mock_spotify(
            [{"id": "pixel", "name": "Pixel 9 Pro", "is_active": True}]
        )
        sp.search.return_value = {
            "playlists": {
                "items": [
                    None,
                    {"name": "Hindi Lo-Fi", "uri": "spotify:playlist:hindi-lofi"},
                ]
            }
        }
        monkeypatch.setattr(music_tools, "_get_spotify_client", lambda: sp)

        response = music_tools.play_spotify_playlist.invoke(
            {"query": "Hindi Lo-Fi", "device": "Pixel 9 Pro"}
        )
        volume_response = music_tools.set_volume.invoke(
            {"volume_percent": 30, "device": "Pixel 9 Pro"}
        )

        assert response == "Now playing Spotify playlist: Hindi Lo-Fi"
        assert volume_response == "Spotify volume set to 30%."
        sp.start_playback.assert_called_once_with(
            device_id="pixel", context_uri="spotify:playlist:hindi-lofi"
        )
        sp.volume.assert_called_once_with(volume_percent=30, device_id="pixel")

    def test_music_agent_never_falls_back_from_requested_device(self, monkeypatch):
        from langchain_core.messages import AIMessage

        import core.subagents.music.agent as music_agent
        import core.subagents.music.tools as music_tools

        sp = self._mock_spotify(
            [{"id": "echo", "name": "Aditya's Echo Dot", "is_active": False}]
        )
        sp.search.return_value = {
            "playlists": {
                "items": [
                    {
                        "name": "Good Vibes Hindi",
                        "uri": "spotify:playlist:good-vibes-hindi",
                    }
                ]
            }
        }
        monkeypatch.setattr(music_tools, "_get_spotify_client", lambda: sp)

        class ExplicitDeviceAgent:
            def invoke(self, _input, _config=None):
                result = music_tools.play_spotify_playlist.invoke(
                    {"query": "Good Vibes Hindi", "device": "Pixel 9 Pro"}
                )
                return {"messages": [AIMessage(content=result)]}

        monkeypatch.setattr(music_agent, "_agent", ExplicitDeviceAgent())

        response = music_agent.music_agent_tool.invoke(
            {
                "request": (
                    "Play a good Hindi playlist on the Pixel 9 Pro at 30% volume."
                )
            }
        )

        assert "couldn't find a device matching 'Pixel 9 Pro'" in response
        assert "Aditya's Echo Dot" in response
        sp.start_playback.assert_not_called()

    def test_playback_tools_require_explicit_device_argument(self, monkeypatch):
        import core.subagents.music.tools as music_tools

        sp = self._mock_spotify([])
        monkeypatch.setattr(music_tools, "_get_spotify_client", lambda: sp)

        with pytest.raises(ValueError):
            music_tools.play_spotify_playlist.invoke({"query": "Good Vibes Hindi"})
        sp.start_playback.assert_not_called()

    def test_resume_playback_resumes_queue_without_search(self, monkeypatch):
        import core.subagents.music.tools as music_tools

        sp = self._mock_spotify(
            [{"id": "echo", "name": "Aditya's Echo Dot", "is_active": False}]
        )
        monkeypatch.setattr(music_tools, "_get_spotify_client", lambda: sp)

        response = music_tools.resume_playback.invoke({"device": "Aditya's Echo Dot"})

        assert response == "Spotify playback resumed."
        sp.search.assert_not_called()
        sp.start_playback.assert_called_once_with(device_id="echo")

    def test_unrelated_track_result_is_never_started(self, monkeypatch):
        import core.subagents.music.tools as music_tools

        sp = self._mock_spotify(
            [{"id": "echo", "name": "Aditya's Echo Dot", "is_active": False}]
        )
        sp.search.return_value = {
            "tracks": {
                "items": [
                    {
                        "uri": "spotify:track:victory-anthem",
                        "name": "Victory Anthem",
                        "artists": [{"name": "Khushi TDT"}],
                    }
                ]
            }
        }
        monkeypatch.setattr(music_tools, "_get_spotify_client", lambda: sp)

        response = music_tools.play_track.invoke(
            {
                "title": "pick up from the previous queue",
                "artist": "",
                "device": "Aditya's Echo Dot",
            }
        )

        assert "not confident" in response
        sp.start_playback.assert_not_called()

    def test_current_playback_snapshot_returns_verified_media(self, monkeypatch):
        import core.subagents.music.tools as music_tools

        sp = self._mock_spotify([])
        sp.current_playback.return_value = {
            "is_playing": True,
            "device": {"name": "Aditya's Echo Dot"},
            "item": {
                "name": "Victory Anthem",
                "uri": "spotify:track:victory-anthem",
                "artists": [{"name": "Khushi TDT"}],
                "album": {
                    "name": "Victory Anthem",
                    "images": [{"url": "https://example.com/cover.jpg"}],
                },
            },
        }
        monkeypatch.setattr(music_tools, "_get_spotify_client", lambda: sp)

        assert music_tools.get_current_playback_snapshot() == {
            "status": "Playing",
            "name": "Victory Anthem",
            "artist": "Khushi TDT",
            "album": "Victory Anthem",
            "cover_url": "https://example.com/cover.jpg",
            "device": "Aditya's Echo Dot",
            "uri": "spotify:track:victory-anthem",
        }

    def test_set_volume_handles_unsupported_device(self, monkeypatch):
        import core.subagents.music.tools as music_tools

        sp = self._mock_spotify(
            [{"id": "pixel", "name": "Pixel 9 Pro", "is_active": True}]
        )
        sp.volume.side_effect = RuntimeError("VOLUME_CONTROL_DISALLOW")
        monkeypatch.setattr(music_tools, "_get_spotify_client", lambda: sp)

        response = music_tools.set_volume.invoke(
            {"volume_percent": 30, "device": "Pixel 9 Pro"}
        )

        assert response == (
            "Spotify cannot control volume on that device. "
            "Please adjust it directly on the device."
        )

    def test_set_volume_uses_matching_current_playback_device(self, monkeypatch):
        import core.subagents.music.tools as music_tools

        sp = self._mock_spotify([])
        sp.current_playback.return_value = {
            "device": {"id": "pixel", "name": "Pixel 9 Pro"}
        }
        monkeypatch.setattr(music_tools, "_get_spotify_client", lambda: sp)

        response = music_tools.set_volume.invoke(
            {"volume_percent": 30, "device": "Pixel 9 Pro"}
        )

        assert response == "Spotify volume set to 30%."
        sp.volume.assert_called_once_with(volume_percent=30, device_id="pixel")


# ---------------------------------------------------------------------------
# Graceful degradation (agent builds fail at import time)
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    """Tests that sub-agents return meaningful errors when their tools are unavailable."""

    def test_info_agent_degrades_when_tools_unavailable(self):
        import core.subagents.info.agent as info_mod

        original = info_mod._agent
        info_mod._agent = None

        with patch("core.subagents.info.agent._build", return_value=None):
            result = info_mod.info_agent_tool.invoke({"request": "weather"})
        assert "unavailable" in result.lower()

        info_mod._agent = original

    def test_music_agent_degrades_when_tools_unavailable(self):
        import core.subagents.music.agent as music_mod

        original = music_mod._agent
        music_mod._agent = None

        with (
            patch("core.subagents.music.agent.spotify_is_connected", return_value=True),
            patch("core.subagents.music.agent._build", return_value=None),
        ):
            result = music_mod.music_agent_tool.invoke({"request": "play something"})
        assert "unavailable" in result.lower()

        music_mod._agent = original

    def test_email_agent_degrades_when_tools_unavailable(self):
        import core.subagents.email.agent as email_mod

        original = email_mod._agent
        email_mod._agent = None

        with patch("core.subagents.email.agent._build", return_value=None):
            result = email_mod.email_agent_tool.invoke({"request": "send email"})
        assert "unavailable" in result.lower()

        email_mod._agent = original

    def test_calendar_agent_degrades_when_tools_unavailable(self):
        import core.subagents.calendar.agent as cal_mod

        original = cal_mod._agent
        cal_mod._agent = None

        with patch("core.subagents.calendar.agent._build", return_value=None):
            result = cal_mod.calendar_agent_tool.invoke({"request": "check calendar"})
        assert "unavailable" in result.lower()

        cal_mod._agent = original

    def test_contacts_agent_degrades_when_tools_unavailable(self):
        import core.subagents.contacts.agent as contacts_mod

        original = contacts_mod._agent
        contacts_mod._agent = None

        with patch("core.subagents.contacts.agent._build", return_value=None):
            result = contacts_mod.contacts_agent_tool.invoke({"request": "find Philip"})
        assert "unavailable" in result.lower()

        contacts_mod._agent = original

    def test_comms_agent_degrades_when_tools_unavailable(self):
        import core.subagents.comms.agent as comms_mod

        original = comms_mod._agent
        comms_mod._agent = None

        with patch("core.subagents.comms.agent._build", return_value=None):
            result = comms_mod.comms_agent_tool.invoke({"request": "send telegram"})
        assert "unavailable" in result.lower()

        comms_mod._agent = original


# ---------------------------------------------------------------------------
# Tool invocation with mocked underlying agent
# ---------------------------------------------------------------------------


class TestToolInvocationWithMockedAgent:
    """Tests that each tool correctly proxies requests to its underlying agent."""

    def _make_mock_agent(self, response_text: str):
        mock_agent = Mock()
        mock_msg = Mock()
        mock_msg.content = response_text
        mock_agent.invoke = Mock(return_value={"messages": [mock_msg]})
        return mock_agent

    def test_info_tool_returns_agent_response(self):
        import core.subagents.info.agent as info_mod

        mock_agent = self._make_mock_agent("Jersey City temp is 12°C, overcast.")
        info_mod._agent = mock_agent
        result = info_mod.info_agent_tool.invoke({"request": "weather in Jersey City"})
        assert "Jersey City" in result or "12" in result
        info_mod._agent = None

    def test_music_tool_returns_agent_response(self):
        import core.subagents.music.agent as music_mod

        mock_agent = self._make_mock_agent("Now playing: Taylor Swift - Love Story")
        music_mod._agent = mock_agent
        result = music_mod.music_agent_tool.invoke(
            {"request": "play Taylor Swift - Love Story"}
        )
        assert "Taylor Swift" in result
        music_mod._agent = None

    def test_email_tool_returns_agent_response(self):
        import core.subagents.email.agent as email_mod

        mock_agent = self._make_mock_agent("Email sent to philip@example.com.")
        email_mod._agent = mock_agent
        result = email_mod.email_agent_tool.invoke(
            {"request": "send email to philip@example.com about the sea level research"}
        )
        assert "philip" in result.lower() or "sent" in result.lower()
        email_mod._agent = None

    def test_calendar_tool_returns_agent_response(self):
        import core.subagents.calendar.agent as cal_mod

        mock_agent = self._make_mock_agent(
            "Event 'Design Review' created for tomorrow at 2pm."
        )
        cal_mod._agent = mock_agent
        result = cal_mod.calendar_agent_tool.invoke(
            {"request": "create Design Review tomorrow 2pm"}
        )
        assert "Design Review" in result or "created" in result.lower()
        cal_mod._agent = None

    def test_contacts_tool_returns_agent_response_when_found(self):
        import core.subagents.contacts.agent as contacts_mod

        mock_agent = self._make_mock_agent("Philip Thorne — philip.thorne@example.com")
        contacts_mod._agent = mock_agent
        result = contacts_mod.contacts_agent_tool.invoke(
            {"request": "Find Philip's email"}
        )
        assert "philip" in result.lower()
        contacts_mod._agent = None

    def test_contacts_tool_returns_not_found_message(self):
        import core.subagents.contacts.agent as contacts_mod

        mock_agent = self._make_mock_agent(
            "No contact found for 'Philip'. Please provide the email address directly."
        )
        contacts_mod._agent = mock_agent
        result = contacts_mod.contacts_agent_tool.invoke(
            {"request": "Find Philip's email"}
        )
        assert "no contact found" in result.lower() or "not found" in result.lower()
        contacts_mod._agent = None

    def test_agent_invocation_passes_request_in_messages(self):
        """Verify the request string is forwarded as a user message."""
        import core.subagents.info.agent as info_mod

        mock_agent = self._make_mock_agent("Sunny.")
        info_mod._agent = mock_agent
        info_mod.info_agent_tool.invoke({"request": "What is the weather in Denver?"})
        call_args = mock_agent.invoke.call_args[0][0]
        messages = call_args.get("messages", [])
        assert any("Denver" in str(m) for m in messages)
        info_mod._agent = None

    def test_comms_tool_returns_agent_response(self):
        import core.subagents.comms.agent as comms_mod

        mock_agent = self._make_mock_agent("Telegram message sent: Build finished.")
        comms_mod._agent = mock_agent
        result = comms_mod.comms_agent_tool.invoke(
            {"request": "Send telegram: Build finished"}
        )
        assert (
            "telegram" in result.lower()
            or "sent" in result.lower()
            or "build" in result.lower()
        )
        comms_mod._agent = None


# ---------------------------------------------------------------------------
# Multi-step chaining scenario (integration-style, all mocked)
# ---------------------------------------------------------------------------


class TestMultiStepChaining:
    """
    Tests the multi-step pattern: info → contacts → email.
    All agents mocked — no LLM calls.
    """

    def test_research_then_email_pattern(self):
        """
        Simulates: 'research rising sea levels then email Philip a summary'
        Step 1: info_agent_tool → returns research
        Step 2: contacts_agent_tool → returns Philip's email
        Step 3: email_agent_tool → sends email
        """
        import core.subagents.contacts.agent as contacts_mod
        import core.subagents.email.agent as email_mod
        import core.subagents.info.agent as info_mod

        def _mock_agent(text):
            m = Mock()
            m.content = text
            a = Mock()
            a.invoke = Mock(return_value={"messages": [m]})
            return a

        # Wire mocks
        info_mod._agent = _mock_agent(
            "Rising sea levels: oceans rising ~3.7mm/year, projections show +1m by 2100."
        )
        contacts_mod._agent = _mock_agent("Philip Thorne — philip.thorne@example.com")
        email_mod._agent = _mock_agent("Email sent to philip.thorne@example.com.")

        # Execute the chain
        research = info_mod.info_agent_tool.invoke(
            {"request": "research rising sea levels 2025"}
        )
        contact = contacts_mod.contacts_agent_tool.invoke(
            {"request": "find Philip's email"}
        )
        email_result = email_mod.email_agent_tool.invoke(
            {
                "request": f"Send email to philip.thorne@example.com: subject 'Rising Sea Levels Summary', body '{research}'"
            }
        )

        # Assertions across the chain
        assert "sea level" in research.lower() or "ocean" in research.lower()
        assert "philip" in contact.lower()
        assert "sent" in email_result.lower() or "philip" in email_result.lower()

        # Cleanup
        info_mod._agent = None
        contacts_mod._agent = None
        email_mod._agent = None

    def test_contacts_fallback_when_email_unknown(self):
        """
        Simulates: email requested, contacts not found, supervisor MUST ask user.
        Verifies contacts returns the canonical not-found message.
        """
        import core.subagents.contacts.agent as contacts_mod

        m = Mock()
        m.content = (
            "No contact found for 'Philip'. Please provide the email address directly."
        )
        mock_agent = Mock()
        mock_agent.invoke = Mock(return_value={"messages": [m]})
        contacts_mod._agent = mock_agent

        result = contacts_mod.contacts_agent_tool.invoke(
            {"request": "find Philip's email"}
        )

        # Supervisor should detect this and ask user
        assert (
            "no contact found" in result.lower() or "please provide" in result.lower()
        )

        contacts_mod._agent = None

    def test_music_then_info_independent_agents(self):
        """
        Simulates: 'play Taylor Swift and tell me the weather'
        Two independent agent calls — both should succeed.
        """
        import core.subagents.info.agent as info_mod
        import core.subagents.music.agent as music_mod

        def _mk(text):
            m = Mock()
            m.content = text
            a = Mock()
            a.invoke = Mock(return_value={"messages": [m]})
            return a

        music_mod._agent = _mk("Now playing: Taylor Swift - Shake It Off")
        info_mod._agent = _mk("Jersey City: 14°C, partly cloudy")

        music_result = music_mod.music_agent_tool.invoke(
            {"request": "play Taylor Swift - Shake It Off"}
        )
        info_result = info_mod.info_agent_tool.invoke(
            {"request": "weather in Jersey City"}
        )

        assert "taylor swift" in music_result.lower()
        assert "jersey city" in info_result.lower() or "14" in info_result

        music_mod._agent = None
        info_mod._agent = None


class TestSpotifyAuthentication:
    def test_missing_token_never_prompts_for_terminal_input(self, monkeypatch):
        import core.subagents.music.tools as music_tools

        auth_manager = Mock()
        auth_manager.cache_handler.get_cached_token.return_value = None
        auth_manager.validate_token.return_value = None
        terminal_input = Mock(side_effect=AssertionError("terminal input is forbidden"))
        monkeypatch.setattr(music_tools, "_spotify_client", None)
        monkeypatch.setattr(music_tools, "_spotify_auth_manager", lambda: auth_manager)
        monkeypatch.setattr("builtins.input", terminal_input)

        assert not music_tools.spotify_is_connected()
        with pytest.raises(music_tools.SpotifyAuthenticationRequired):
            music_tools._get_spotify_client()
        terminal_input.assert_not_called()

    def test_control_panel_callback_connects_without_terminal_input(self, monkeypatch):
        import core.subagents.music.tools as music_tools

        auth_manager = Mock()
        auth_manager.parse_response_code.return_value = "authorization-code"
        spotify_client = Mock()
        monkeypatch.setattr(music_tools, "_spotify_client", None)
        monkeypatch.setattr(music_tools, "_spotify_auth_manager", lambda: auth_manager)
        monkeypatch.setattr(music_tools.spotipy, "Spotify", spotify_client)

        music_tools.complete_spotify_authorization(
            "http://127.0.0.1:8100/callback?code=authorization-code"
        )

        auth_manager.get_access_token.assert_called_once_with(
            "authorization-code", as_dict=False
        )
        spotify_client.assert_called_once_with(auth_manager=auth_manager)

    def test_music_agent_returns_setup_message_before_building(self, monkeypatch):
        import core.subagents.music.agent as music_agent

        monkeypatch.setattr(music_agent, "_agent", None)
        monkeypatch.setattr(music_agent, "_agents", {})
        monkeypatch.setattr(music_agent, "spotify_is_connected", lambda: False)
        build = Mock()
        monkeypatch.setattr(music_agent, "_build", build)

        response = music_agent.music_agent_tool.invoke({"request": "play music"})

        assert "agent controls" in response.lower()
        build.assert_not_called()


# ---------------------------------------------------------------------------
# info_agent_tool query/request parameter compatibility
# ---------------------------------------------------------------------------


class TestInfoAgentParamCompat:
    """info_agent_tool must accept both `request` and `query` params (LLM compat)."""

    def _make_mock_agent(self, response_text: str):
        mock_agent = Mock()
        mock_msg = Mock()
        mock_msg.content = response_text
        mock_agent.invoke = Mock(return_value={"messages": [mock_msg]})
        return mock_agent

    def test_invoke_with_request_param(self):
        """Standard invocation with `request` param works."""
        import core.subagents.info.agent as info_mod

        info_mod._agent = self._make_mock_agent("Sunny and warm.")
        result = info_mod.info_agent_tool.invoke({"request": "weather in NYC"})
        assert "Sunny" in result
        info_mod._agent = None

    def test_invoke_with_query_param(self):
        """Backward-compat: `query` param is accepted as alias for `request`."""
        import core.subagents.info.agent as info_mod

        info_mod._agent = self._make_mock_agent("Rainy in London.")
        result = info_mod.info_agent_tool.invoke({"query": "weather in London"})
        assert "London" in result or "Rainy" in result
        info_mod._agent = None

    def test_query_forwarded_to_agent(self):
        """When `query` is used, the value is forwarded to the underlying agent."""
        import core.subagents.info.agent as info_mod

        mock_agent = self._make_mock_agent("Result.")
        info_mod._agent = mock_agent
        info_mod.info_agent_tool.invoke({"query": "Mount Everest height"})

        call_args = mock_agent.invoke.call_args[0][0]
        messages = call_args.get("messages", [])
        assert any("Mount Everest" in str(m) for m in messages)
        info_mod._agent = None

    def test_request_takes_precedence_over_query(self):
        """When both `request` and `query` are provided, `request` wins."""
        import core.subagents.info.agent as info_mod

        mock_agent = self._make_mock_agent("Response.")
        info_mod._agent = mock_agent
        info_mod.info_agent_tool.invoke(
            {"request": "weather in Paris", "query": "weather in Tokyo"}
        )

        call_args = mock_agent.invoke.call_args[0][0]
        messages = call_args.get("messages", [])
        assert any("Paris" in str(m) for m in messages)
        info_mod._agent = None

    def test_empty_params_returns_helpful_error(self):
        """Invoking with both params empty returns a helpful error, not a crash."""
        import core.subagents.info.agent as info_mod

        original = info_mod._agent
        result = info_mod.info_agent_tool.invoke({"request": "", "query": ""})
        assert "provide" in result.lower() or "request" in result.lower()
        info_mod._agent = original

    def test_missing_params_returns_helpful_error(self):
        """Invoking with no params at all returns a helpful error."""
        import core.subagents.info.agent as info_mod

        original = info_mod._agent
        result = info_mod.info_agent_tool.invoke({})
        assert "provide" in result.lower() or "request" in result.lower()
        info_mod._agent = original


class TestInfoAgentExecution:
    """One semantically-directed info agent with bounded execution."""

    def _mk_agent(self, response_text: str):
        mock_agent = Mock()
        mock_msg = Mock()
        mock_msg.content = response_text
        mock_agent.invoke = Mock(return_value={"messages": [mock_msg]})
        return mock_agent

    def test_prompt_requires_semantic_source_judgment(self):
        import core.subagents.info.agent as info_mod

        prompt = info_mod._BASE_PROMPT.lower()

        assert "meaning of the user's request" in prompt
        assert "not from\nword matching" in prompt
        assert "two substantive\nnon-wikipedia pages" in prompt
        assert "tavily_search" in prompt
        assert "tavily_extract" in prompt

    def test_info_agent_exposes_all_relevant_sources(self):
        from core.subagents.info.tools import get_info_tools

        tool_names = {tool.name for tool in get_info_tools()}

        assert {"search_web", "read_website", "wikipedia_search"} <= tool_names

    def test_info_agent_uses_fixed_recursion_limit(self, monkeypatch):
        import core.subagents.info.agent as info_mod

        original_agent = info_mod._agent
        agent = self._mk_agent("Answer")

        info_mod._agent = agent
        monkeypatch.setattr(info_mod.Config, "INFO_RECURSION_LIMIT", 30)

        result = info_mod.info_agent_tool.invoke({"request": "quick fact"})

        assert "Answer" in result
        call_args = agent.invoke.call_args[0]
        assert call_args[1]["recursion_limit"] == 30

        info_mod._agent = original_agent

    def test_recursion_returns_graceful_fallback(self):
        import core.subagents.info.agent as info_mod

        original_agent = info_mod._agent
        original_recent_context = info_mod.recent_context

        deep_agent = Mock()
        deep_agent.invoke = Mock(side_effect=GraphRecursionError("limit"))
        mock_recent_context = Mock()
        mock_recent_context.invoke = Mock(
            return_value="No recent snippets available. Run a search tool first."
        )

        info_mod._agent = None
        info_mod._agent = deep_agent
        info_mod.recent_context = mock_recent_context

        result = info_mod.info_agent_tool.invoke({"request": "analyze this in depth"})

        assert "research loop" in result.lower()
        assert "recursion limit" not in result.lower()

        info_mod._agent = original_agent
        info_mod.recent_context = original_recent_context
