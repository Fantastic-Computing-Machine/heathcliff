# ABOUTME: Shared router and application shell for the Heathcliff control panel.

from __future__ import annotations

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui.runtime import AppRuntime
from ui.views import (
    agent_controls,
    analytics,
    command_center,
    conversations,
    memories,
    observability,
)

st.set_page_config(
    page_title="Heathcliff Control Panel",
    page_icon="🕴️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def get_runtime() -> AppRuntime:
    """Keep the shared agent runtime alive through Streamlit reruns."""
    return AppRuntime()


try:
    runtime = get_runtime()
except Exception as exc:
    st.error("Heathcliff could not initialise.")
    st.code(str(exc), language="text")
    st.stop()

st.session_state["app_runtime"] = runtime
profile, revision = runtime.snapshot()

with st.sidebar:
    st.markdown("## Heathcliff")
    st.caption("Autonomous intelligence control panel")
    st.divider()
    stats = runtime.memory.get_stats()
    st.metric("Memories", stats.get("memories", 0))
    st.metric("Conversation messages", stats.get("chats", 0))
    st.caption(f"Runtime profile r{revision}")
    st.caption(f"{len(profile.enabled_agents)} capabilities enabled")
    st.divider()
    st.caption(
        "Operator surface — deploy behind private access until authentication is added."
    )

pages = {
    "Operate": [
        st.Page(
            command_center.render,
            title="Command Center",
            icon=":material/terminal:",
            default=True,
        ),
    ],
    "Observe": [
        st.Page(
            observability.render,
            title="Runs & Traces",
            icon=":material/monitoring:",
            url_path="traces",
        ),
        st.Page(
            analytics.render,
            title="Analytics",
            icon=":material/analytics:",
            url_path="analytics",
        ),
    ],
    "Knowledge": [
        st.Page(
            memories.render,
            title="Memories",
            icon=":material/psychology:",
            url_path="memories",
        ),
        st.Page(
            conversations.render,
            title="Conversations",
            icon=":material/forum:",
            url_path="conversations",
        ),
    ],
    "Manage": [
        st.Page(
            agent_controls.render,
            title="Agent Controls",
            icon=":material/tune:",
            url_path="agent-controls",
        ),
    ],
}

st.navigation(pages, position="sidebar", expanded=True).run()
