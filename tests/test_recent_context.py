# ABOUTME: Tests for core/subagents/info/recent_context.py — JSON-backed recency store
# ABOUTME: Covers persistence, TTL expiry, max-items pruning, corruption recovery, ordering

import json
import os
import sys
import time
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Helpers — isolate each test with a fresh temp file + reset module cache
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    """Point the recent-context store at a per-test temp file and reset cache."""
    store_file = str(tmp_path / "recent_memory.json")

    monkeypatch.setattr(
        "config.config.RecentContextConfig.RECENT_CONTEXT_STORE_PATH", store_file
    )
    monkeypatch.setattr(
        "config.config.RecentContextConfig.RECENT_CONTEXT_TTL_SECONDS", 7200
    )
    monkeypatch.setattr(
        "config.config.RecentContextConfig.RECENT_CONTEXT_MAX_ITEMS", 100
    )
    monkeypatch.setattr(
        "config.config.RecentContextConfig.RECENT_CONTEXT_MAX_SNIPPET_CHARS", 1200
    )
    monkeypatch.setattr(
        "config.config.RecentContextConfig.RECENT_CONTEXT_MAX_RETURN", 5
    )

    import core.subagents.info.recent_context as rc_mod

    with rc_mod._lock:
        rc_mod._cache.clear()
        rc_mod._cache_loaded = False

    yield store_file


def _read_store(path: str):
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        content = f.read().strip()
        if not content:
            return []
        return json.loads(content)


# ---------------------------------------------------------------------------
# File persistence roundtrip
# ---------------------------------------------------------------------------


class TestPersistenceRoundtrip:
    def test_capture_creates_file_and_persists(self, _isolated_store):
        from core.subagents.info.recent_context import _capture_recent_result

        _capture_recent_result("search_web", "Some search results here")
        data = _read_store(_isolated_store)
        assert len(data) == 1
        assert data[0]["tool_name"] == "search_web"
        assert data[0]["snippet"] == "Some search results here"
        assert "created_at" in data[0]

    def test_multiple_captures_persist(self, _isolated_store):
        from core.subagents.info.recent_context import _capture_recent_result

        _capture_recent_result("get_weather", "Sunny 25C")
        _capture_recent_result("get_news", "Breaking: AI advances")
        _capture_recent_result("wikipedia_search", "Python is a language")

        data = _read_store(_isolated_store)
        assert len(data) == 3
        tool_names = [e["tool_name"] for e in data]
        assert "get_weather" in tool_names
        assert "get_news" in tool_names
        assert "wikipedia_search" in tool_names

    def test_cache_reloads_from_disk(self, _isolated_store):
        from core.subagents.info.recent_context import _capture_recent_result

        _capture_recent_result("search_web", "Result A")

        # Reset in-memory cache to simulate fresh process
        import core.subagents.info.recent_context as rc_mod

        with rc_mod._lock:
            rc_mod._cache.clear()
            rc_mod._cache_loaded = False

        # Capture again — should reload from disk first
        _capture_recent_result("get_news", "Result B")

        data = _read_store(_isolated_store)
        assert len(data) == 2

    def test_recent_context_tool_reads_persisted_data(self, _isolated_store):
        from core.subagents.info.recent_context import (
            _capture_recent_result,
            recent_context,
        )

        _capture_recent_result("search_web", "Persisted result")
        result = recent_context.invoke({"n": 3})
        assert "search_web" in result
        assert "Persisted result" in result


# ---------------------------------------------------------------------------
# TTL expiry
# ---------------------------------------------------------------------------


class TestTTLExpiry:
    def test_expired_entries_pruned_on_read(self, _isolated_store):
        from core.subagents.info.recent_context import recent_context

        # Write an entry with a timestamp 3 hours ago (beyond 2h TTL)
        old_entry = {
            "tool_name": "search_web",
            "snippet": "Stale data",
            "created_at": time.time() - 10800,  # 3h ago
        }
        with open(_isolated_store, "w") as f:
            json.dump([old_entry], f)

        # Reset cache so it re-reads from disk
        import core.subagents.info.recent_context as rc_mod

        with rc_mod._lock:
            rc_mod._cache.clear()
            rc_mod._cache_loaded = False

        result = recent_context.invoke({"n": 3})
        assert "No recent snippets available" in result

    def test_fresh_entries_survive_prune(self, _isolated_store):
        from core.subagents.info.recent_context import (
            _capture_recent_result,
            recent_context,
        )

        _capture_recent_result("get_weather", "Fresh weather data")
        result = recent_context.invoke({"n": 3})
        assert "Fresh weather data" in result

    def test_mixed_fresh_and_stale(self, _isolated_store):
        # Write mix of stale and fresh entries
        entries = [
            {
                "tool_name": "search_web",
                "snippet": "Stale",
                "created_at": time.time() - 10800,
            },
            {
                "tool_name": "get_news",
                "snippet": "Fresh news",
                "created_at": time.time(),
            },
        ]
        with open(_isolated_store, "w") as f:
            json.dump(entries, f)

        import core.subagents.info.recent_context as rc_mod

        with rc_mod._lock:
            rc_mod._cache.clear()
            rc_mod._cache_loaded = False

        result = rc_mod.recent_context.invoke({"n": 5})
        assert "Fresh news" in result
        assert "Stale" not in result

    def test_stale_entries_pruned_on_write(self, _isolated_store):
        from core.subagents.info.recent_context import _capture_recent_result

        # Seed a stale entry on disk
        old_entry = {
            "tool_name": "old_tool",
            "snippet": "Old data",
            "created_at": time.time() - 10800,
        }
        with open(_isolated_store, "w") as f:
            json.dump([old_entry], f)

        import core.subagents.info.recent_context as rc_mod

        with rc_mod._lock:
            rc_mod._cache.clear()
            rc_mod._cache_loaded = False

        # New capture should trigger prune of old entry
        _capture_recent_result("new_tool", "New data")
        data = _read_store(_isolated_store)
        assert len(data) == 1
        assert data[0]["tool_name"] == "new_tool"


# ---------------------------------------------------------------------------
# Max-items pruning
# ---------------------------------------------------------------------------


class TestMaxItemsPruning:
    def test_excess_entries_pruned_to_max(self, _isolated_store, monkeypatch):
        monkeypatch.setattr(
            "config.config.RecentContextConfig.RECENT_CONTEXT_MAX_ITEMS", 5
        )
        from core.subagents.info.recent_context import _capture_recent_result

        for i in range(10):
            _capture_recent_result("tool", f"Snippet {i}")

        data = _read_store(_isolated_store)
        assert len(data) == 5

    def test_newest_entries_kept_after_prune(self, _isolated_store, monkeypatch):
        monkeypatch.setattr(
            "config.config.RecentContextConfig.RECENT_CONTEXT_MAX_ITEMS", 3
        )
        from core.subagents.info.recent_context import _capture_recent_result

        for i in range(6):
            _capture_recent_result("tool", f"Item {i}")

        data = _read_store(_isolated_store)
        snippets = [e["snippet"] for e in data]
        # Should keep the 3 newest: Item 3, Item 4, Item 5
        assert "Item 5" in snippets
        assert "Item 4" in snippets
        assert "Item 3" in snippets
        assert "Item 0" not in snippets


# ---------------------------------------------------------------------------
# Corrupted JSON fallback
# ---------------------------------------------------------------------------


class TestCorruptedFallback:
    def test_corrupt_json_resets_gracefully(self, _isolated_store):
        # Write garbage to the store
        with open(_isolated_store, "w") as f:
            f.write("{{{not valid json!!!")

        import core.subagents.info.recent_context as rc_mod

        with rc_mod._lock:
            rc_mod._cache.clear()
            rc_mod._cache_loaded = False

        # Should not crash — returns empty
        result = rc_mod.recent_context.invoke({"n": 3})
        assert "No recent snippets available" in result

    def test_corrupt_json_allows_new_writes(self, _isolated_store):
        with open(_isolated_store, "w") as f:
            f.write("CORRUPT DATA")

        import core.subagents.info.recent_context as rc_mod

        with rc_mod._lock:
            rc_mod._cache.clear()
            rc_mod._cache_loaded = False

        rc_mod._capture_recent_result("get_news", "New after corrupt")
        data = _read_store(_isolated_store)
        assert len(data) == 1
        assert data[0]["snippet"] == "New after corrupt"

    def test_non_list_json_resets(self, _isolated_store):
        with open(_isolated_store, "w") as f:
            json.dump({"not": "a list"}, f)

        import core.subagents.info.recent_context as rc_mod

        with rc_mod._lock:
            rc_mod._cache.clear()
            rc_mod._cache_loaded = False

        result = rc_mod.recent_context.invoke({"n": 3})
        assert "No recent snippets available" in result


# ---------------------------------------------------------------------------
# n clamping and return ordering
# ---------------------------------------------------------------------------


class TestReturnBehavior:
    def test_n_clamped_to_max_return(self, _isolated_store, monkeypatch):
        monkeypatch.setattr(
            "config.config.RecentContextConfig.RECENT_CONTEXT_MAX_RETURN", 3
        )
        from core.subagents.info.recent_context import _capture_recent_result

        for i in range(10):
            _capture_recent_result("tool", f"S{i}")

        import core.subagents.info.recent_context as rc_mod

        result = rc_mod.recent_context.invoke({"n": 99})
        # Should only contain 3 entries despite requesting 99
        assert result.count("---") == 2  # 3 items = 2 separators

    def test_n_clamped_to_minimum_1(self, _isolated_store):
        from core.subagents.info.recent_context import _capture_recent_result

        _capture_recent_result("tool", "Only one")

        import core.subagents.info.recent_context as rc_mod

        result = rc_mod.recent_context.invoke({"n": 0})
        assert "Only one" in result
        assert "---" not in result

    def test_newest_first_ordering(self, _isolated_store):
        from core.subagents.info.recent_context import _capture_recent_result

        _capture_recent_result("tool", "First")
        time.sleep(0.01)
        _capture_recent_result("tool", "Second")
        time.sleep(0.01)
        _capture_recent_result("tool", "Third")

        import core.subagents.info.recent_context as rc_mod

        result = rc_mod.recent_context.invoke({"n": 3})
        # "Third" should appear before "First" in the output
        third_pos = result.index("Third")
        first_pos = result.index("First")
        assert third_pos < first_pos

    def test_empty_store_returns_guidance(self, _isolated_store):
        from core.subagents.info.recent_context import recent_context

        result = recent_context.invoke({"n": 3})
        assert "No recent snippets available" in result
        assert "Run a search tool first" in result


# ---------------------------------------------------------------------------
# Snippet content filtering
# ---------------------------------------------------------------------------


class TestContentFiltering:
    def test_empty_content_not_stored(self, _isolated_store):
        from core.subagents.info.recent_context import _capture_recent_result

        _capture_recent_result("tool", "")
        _capture_recent_result("tool", "   ")
        data = _read_store(_isolated_store)
        assert len(data) == 0

    def test_error_responses_not_stored(self, _isolated_store):
        from core.subagents.info.recent_context import _capture_recent_result

        _capture_recent_result("tool", "Error fetching data: timeout")
        _capture_recent_result("tool", "error: connection refused")
        data = _read_store(_isolated_store)
        assert len(data) == 0

    def test_long_snippets_truncated(self, _isolated_store, monkeypatch):
        monkeypatch.setattr(
            "config.config.RecentContextConfig.RECENT_CONTEXT_MAX_SNIPPET_CHARS", 50
        )
        from core.subagents.info.recent_context import _capture_recent_result

        _capture_recent_result("tool", "A" * 200)
        data = _read_store(_isolated_store)
        assert len(data) == 1
        assert len(data[0]["snippet"]) == 53  # 50 + "..."
        assert data[0]["snippet"].endswith("...")


# ---------------------------------------------------------------------------
# Auto path setup
# ---------------------------------------------------------------------------


class TestAutoPathSetup:
    def test_creates_parent_directory(self, tmp_path, monkeypatch):
        nested = str(tmp_path / "deep" / "nested" / "store.json")
        monkeypatch.setattr(
            "config.config.RecentContextConfig.RECENT_CONTEXT_STORE_PATH", nested
        )

        import core.subagents.info.recent_context as rc_mod

        with rc_mod._lock:
            rc_mod._cache.clear()
            rc_mod._cache_loaded = False

        rc_mod._ensure_store_path()
        assert os.path.exists(nested)

    def test_creates_empty_file_if_missing(self, tmp_path, monkeypatch):
        store = str(tmp_path / "new_store.json")
        monkeypatch.setattr(
            "config.config.RecentContextConfig.RECENT_CONTEXT_STORE_PATH", store
        )

        import core.subagents.info.recent_context as rc_mod

        rc_mod._ensure_store_path()
        data = _read_store(store)
        assert data == []
