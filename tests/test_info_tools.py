# ABOUTME: Regression tests for reliable Wikipedia search results
# ABOUTME: Keeps research independent from the brittle wikipedia package client

from types import SimpleNamespace
from unittest.mock import Mock, patch

from core.subagents.info import tools


def test_wikipedia_search_uses_rest_api(monkeypatch):
    response = Mock()
    response.json.return_value = {
        "pages": [
            {
                "title": "Mount Fuji",
                "description": "Mountain in Japan",
                "excerpt": "Japan's <span>highest</span> mountain",
            }
        ]
    }
    monkeypatch.setattr(tools.requests, "get", lambda *args, **kwargs: response)

    result = tools.wikipedia_search.invoke({"query": "Mount Fuji"})

    assert result == "Mount Fuji — Mountain in Japan — Japan's highest mountain"
    response.raise_for_status.assert_called_once()


def test_google_search_uses_its_own_credentials(monkeypatch):
    captured = {}
    monkeypatch.setattr(tools, "_google_tool", None)

    class FakeWrapper:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(tools, "GoogleSearchAPIWrapper", FakeWrapper)
    monkeypatch.setattr(tools, "GoogleSearchResults", lambda **kwargs: Mock())

    tools._init_google_tool(
        SimpleNamespace(GOOGLE_CSE_API_KEY="search-key", GOOGLE_CSE_ID="search-id")
    )

    assert captured == {"google_api_key": "search-key", "google_cse_id": "search-id"}


def test_tavily_tools_are_opt_in(monkeypatch):
    monkeypatch.setattr(tools.Config, "TAVILY_API_KEY", None)

    assert tools.tavily_tools() == []


def test_tavily_tools_use_official_langchain_integration(monkeypatch):
    monkeypatch.setattr(tools.Config, "TAVILY_API_KEY", "tvly-test")
    search, extract = Mock(name="search"), Mock(name="extract")
    monkeypatch.setattr(tools, "_tavily_search_tool", None)
    monkeypatch.setattr(tools, "_tavily_extract_tool", None)

    with patch.dict(
        "sys.modules",
        {
            "langchain_tavily": SimpleNamespace(
                TavilySearch=lambda **kwargs: search,
                TavilyExtract=lambda **kwargs: extract,
            )
        },
    ):
        assert [tool.name for tool in tools.tavily_tools()] == [
            "tavily_search",
            "tavily_extract",
        ]
        assert tools._init_tavily_tools() == (search, extract)


def test_search_provider_error_falls_back_without_wikipedia(monkeypatch):
    monkeypatch.setattr(
        tools,
        "_dispatch_search",
        Mock(
            side_effect=[
                tools.SearchOutcome("duckduckgo", error="rate limited"),
                tools.SearchOutcome("google", content="Independent result"),
            ]
        ),
    )
    wikipedia = Mock(side_effect=AssertionError("Wikipedia must remain explicit"))
    monkeypatch.setattr(tools, "_search_wikipedia", wikipedia)

    result = tools.search_web.invoke({"query": "evidence workflow"})

    assert result == "Independent result"
    wikipedia.assert_not_called()


def test_all_search_provider_errors_are_reported_as_failure(monkeypatch):
    monkeypatch.setattr(
        tools,
        "_dispatch_search",
        Mock(
            side_effect=[
                tools.SearchOutcome("duckduckgo", error="rate limited"),
                tools.SearchOutcome("google", error="not configured"),
            ]
        ),
    )

    result = tools.search_web.invoke({"query": "evidence workflow"})

    assert result.startswith("Web search failed")
    assert "rate limited" in result
