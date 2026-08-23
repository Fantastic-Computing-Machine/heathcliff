# ABOUTME: Focused control-panel tests without live Streamlit, Mem0, or Langfuse services.

from pathlib import Path
from types import SimpleNamespace

from streamlit.testing.v1 import AppTest


class _MemoryList:
    def get(self, *, limit, offset=0):
        return {
            "documents": [f"memory {index}" for index in range(offset, offset + limit)],
            "metadatas": [{"category": "fact"} for _ in range(limit)],
            "ids": [f"m{index}" for index in range(offset, offset + limit)],
        }


def test_memory_rows_default_to_a_50_item_page():
    from ui.views.memories import PAGE_SIZE, _memory_rows

    memory = SimpleNamespace(memories=_MemoryList())
    rows = _memory_rows(memory, query="", page=2)

    assert len(rows) == PAGE_SIZE
    assert rows[0]["id"] == "m100"


def test_analytics_includes_persisted_execution_events():
    from ui.views.analytics import _event_rows

    memory = SimpleNamespace(
        get_conversation_history=lambda _conversation_id: [
            {
                "created_at": "2026-08-15T12:00:00",
                "execution_events": [
                    {"type": "plan", "message": "One task", "data": {"count": 1}}
                ],
            }
        ]
    )

    rows = _event_rows(memory, [{"conversation_id": "conversation-123"}])

    assert rows == [
        {
            "Time": "2026-08-15T12:00:00",
            "Conversation": "conversa",
            "Event": "plan",
            "Detail": "One task",
            "Data": {"count": 1},
        }
    ]


def test_command_center_renders_without_custom_html_or_services():
    app = AppTest.from_string(
        """
import streamlit as st
from types import SimpleNamespace
from ui.views.command_center import render

profile = SimpleNamespace(enabled_agents=("info_agent_tool",))
runtime = SimpleNamespace(snapshot=lambda: (profile, 1))
st.session_state["app_runtime"] = runtime
render()
"""
    ).run()

    assert not app.exception
    assert app.title[0].value == "Command Center"


def test_home_router_uses_view_render_callables():
    source = Path("ui/Home.py").read_text(encoding="utf-8")

    for view in (
        "command_center",
        "observability",
        "analytics",
        "memories",
        "conversations",
        "agent_controls",
    ):
        assert f"{view}.render" in source

    for url_path in (
        'url_path="traces"',
        'url_path="analytics"',
        'url_path="memories"',
        'url_path="conversations"',
        'url_path="agent-controls"',
    ):
        assert url_path in source
