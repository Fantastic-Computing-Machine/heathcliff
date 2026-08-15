# ABOUTME: Tests canonical Google AI key selection
# ABOUTME: Keeps old environment names working during migration

from config.config import PlatformConf, _ai_api_key


def test_ai_key_takes_precedence(monkeypatch):
    monkeypatch.setenv("AI_KEY", "canonical-ai")
    monkeypatch.setenv("GEMINI_API_KEY", "legacy-gemini")
    monkeypatch.setenv("GOOGLE_API_KEY", "legacy-google")

    assert _ai_api_key() == "canonical-ai"


def test_ai_key_accessor_returns_platform_key(monkeypatch):
    monkeypatch.setattr(PlatformConf, "AI_KEY", "provider-key")

    assert PlatformConf.get_ai_api_key() == "provider-key"
