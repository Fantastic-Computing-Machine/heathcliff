# ABOUTME: Tests for the stealth HTTP client and improved search_graph_tool
# ABOUTME: Validates both math_evaluator and search_and_scrape with mocked externals

from unittest.mock import MagicMock, patch

import pytest

from tools.math_tools import math_evaluator
from tools.search_graph_tool import search_and_scrape
from utils.http_client import StealthHttpClient


class TestMathEvaluator:
    def test_basic_addition(self):
        assert math_evaluator.run("2 + 2") == "4"

    def test_sqrt(self):
        assert math_evaluator.run("sqrt(16)") == "4"


class TestStealthHttpClient:
    def test_random_user_agent_varies(self):
        client = StealthHttpClient()
        agents = {client.random_user_agent() for _ in range(20)}
        # Should produce more than one unique UA over 20 draws
        assert len(agents) > 1

    def test_random_headers_has_required_keys(self):
        client = StealthHttpClient()
        headers = client.random_headers()
        assert "User-Agent" in headers
        assert "Accept" in headers
        assert "Accept-Language" in headers

    def test_extra_headers_merged(self):
        client = StealthHttpClient()
        headers = client.random_headers(extra_headers={"X-Custom": "test"})
        assert headers["X-Custom"] == "test"


class TestSearchAndScrape:
    @patch("tools.search_graph_tool._SEARCH_SCRAPE_GRAPH")
    def test_returns_joined_snippets(self, mock_graph):
        mock_graph.invoke.return_value = {"snippets": ["Snippet 1", "Snippet 2"]}
        result = search_and_scrape.run({"query": "test query"})
        assert "Snippet 1" in result
        assert "Snippet 2" in result
        mock_graph.invoke.assert_called_once()

    @patch("tools.search_graph_tool._SEARCH_SCRAPE_GRAPH")
    def test_empty_snippets(self, mock_graph):
        mock_graph.invoke.return_value = {"snippets": []}
        result = search_and_scrape.run({"query": "test query"})
        assert "No content" in result
