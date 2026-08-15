# ABOUTME: Small regression checks for Heathcliff's dashboard greeting.

from utils import heathcliff_greetings as greetings


def test_weather_greeting_keeps_butler_tone(monkeypatch):
    monkeypatch.setattr(greetings, "get_time_of_day", lambda: "morning")
    monkeypatch.setattr(greetings, "get_weather_description", lambda: "rain, 12°C")

    greeting = greetings.generate_greeting("Bruce")

    assert "Bruce" in greeting
    assert "morning" in greeting.lower()
    assert "rain, 12°C" in greeting


def test_greeting_can_skip_weather(monkeypatch):
    monkeypatch.setattr(greetings, "get_time_of_day", lambda: "evening")
    monkeypatch.setattr(
        greetings,
        "get_weather_description",
        lambda: (_ for _ in ()).throw(AssertionError("weather should not be fetched")),
    )

    assert "Alfred" in greetings.generate_greeting("Alfred", include_weather=False)
