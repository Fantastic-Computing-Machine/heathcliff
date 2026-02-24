# ABOUTME: JSON-backed recency store for recent tool-call snippets
# ABOUTME: TTL + max-items pruning, atomic writes, thread-safe, auto-path setup

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict, List, Optional

from langchain.tools import tool
from pydantic import BaseModel, ConfigDict, Field

from logger import logger

# ---------------------------------------------------------------------------
# In-memory cache + thread lock
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_cache: List[Dict[str, Any]] = []
_cache_loaded = False


# ---------------------------------------------------------------------------
# Config helpers (lazy import to avoid circular import at module level)
# ---------------------------------------------------------------------------


def _cfg() -> Any:
    """Return the Config singleton (lazy to avoid circular import)."""
    from config import Config

    return Config


def _store_path() -> str:
    return _cfg().RECENT_CONTEXT_STORE_PATH


def _ttl() -> int:
    return _cfg().RECENT_CONTEXT_TTL_SECONDS


def _max_items() -> int:
    return _cfg().RECENT_CONTEXT_MAX_ITEMS


def _max_snippet_chars() -> int:
    return _cfg().RECENT_CONTEXT_MAX_SNIPPET_CHARS


def _max_return() -> int:
    return _cfg().RECENT_CONTEXT_MAX_RETURN


# ---------------------------------------------------------------------------
# Path setup (auto-create on first access)
# ---------------------------------------------------------------------------


def _ensure_store_path() -> str:
    """Ensure parent directory and file exist. Returns the store path."""
    path = _store_path()
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if not os.path.exists(path):
        _atomic_write(path, [])
    return path


# ---------------------------------------------------------------------------
# File I/O — atomic write + corrupt-safe read
# ---------------------------------------------------------------------------


def _atomic_write(path: str, data: List[Dict[str, Any]]) -> None:
    """Write JSON atomically via tmp + rename."""
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp_path, path)
    except Exception as exc:
        logger.warning(f"[recent_context] atomic write failed: {exc}")
        # Clean up stale tmp if rename failed
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _read_file(path: str) -> List[Dict[str, Any]]:
    """Load JSON from file. Returns empty list on corrupt/missing file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        logger.warning("[recent_context] store is not a list — resetting")
        return []
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning(f"[recent_context] corrupt store — resetting: {exc}")
        return []
    except FileNotFoundError:
        return []
    except Exception as exc:
        logger.warning(f"[recent_context] read failed — resetting: {exc}")
        return []


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------


def _prune(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove expired entries and enforce max-items cap."""
    now = time.time()
    ttl = _ttl()
    max_items = _max_items()

    # TTL filter
    alive = [e for e in entries if (now - e.get("created_at", 0)) < ttl]

    # Count cap — keep newest
    if len(alive) > max_items:
        alive.sort(key=lambda e: e.get("created_at", 0))
        alive = alive[-max_items:]

    return alive


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------


def _load_cache() -> None:
    """Load from disk into in-memory cache (once, then kept in sync)."""
    global _cache, _cache_loaded
    path = _ensure_store_path()
    raw = _read_file(path)
    _cache = _prune(raw)
    _cache_loaded = True


def _ensure_cache() -> None:
    """Lazy-load cache on first access."""
    global _cache_loaded
    if not _cache_loaded:
        _load_cache()


def _persist() -> None:
    """Write current in-memory cache to disk."""
    path = _ensure_store_path()
    _atomic_write(path, _cache)


# ---------------------------------------------------------------------------
# Public write API (used by tools.py via _capture_recent_result)
# ---------------------------------------------------------------------------


def _capture_recent_result(tool_name: str, content: str) -> None:
    """Store a compact, recency-biased snippet from a successful tool result."""
    if not content:
        return

    text = content.strip()
    if not text:
        return

    # Skip error responses
    if text.lower().startswith("error"):
        return

    max_chars = _max_snippet_chars()
    if len(text) > max_chars:
        text = text[:max_chars] + "..."

    entry: Dict[str, Any] = {
        "tool_name": tool_name,
        "snippet": text,
        "created_at": time.time(),
    }

    with _lock:
        _ensure_cache()
        _cache.append(entry)
        # Prune after append
        pruned = _prune(_cache)
        _cache.clear()
        _cache.extend(pruned)
        _persist()


# ---------------------------------------------------------------------------
# LangChain tool
# ---------------------------------------------------------------------------


class RecentContextArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n: int = Field(3, description="Number of recent snippets to return. Default 3.")


@tool(args_schema=RecentContextArgs)
def recent_context(n: int = 3) -> str:
    """Return the most recent snippets captured from prior tool calls.

    Use for: grounding answers with fresh data from earlier tool results.
    Provide: n (number of snippets, default 3).
    Returns: Newest snippets separated by --- dividers, or guidance to run a search first.
    """
    with _lock:
        _ensure_cache()
        # Re-prune on read to evict stale entries
        pruned = _prune(_cache)
        _cache.clear()
        _cache.extend(pruned)

        if not _cache:
            return "No recent snippets available. Run a search tool first."

        max_n = _max_return()
        n = max(1, min(n, max_n))

        # Sort newest-first
        sorted_entries = sorted(
            _cache, key=lambda e: e.get("created_at", 0), reverse=True
        )
        latest = sorted_entries[:n]

    parts = []
    for entry in latest:
        tool_name = entry.get("tool_name", "unknown")
        snippet = entry.get("snippet", "")
        parts.append(f"{tool_name}: {snippet}")

    return "\n---\n".join(parts)


# ---------------------------------------------------------------------------
# Module-level auto-setup: ensure store path exists on import
# ---------------------------------------------------------------------------

try:
    _ensure_store_path()
except Exception as exc:
    logger.warning(f"[recent_context] auto-setup failed (non-fatal): {exc}")
