# ABOUTME: Langfuse health and current-session trace access for Heathcliff operators.

from __future__ import annotations

import streamlit as st

from config import Config
from ui.components import page_heading, status_line
from utils.langfuse_client import get_langfuse_client


def render() -> None:
    page_heading(
        "Observe",
        "Runs & Traces",
        "Langfuse is the durable record for plans, tool calls, timings, and outputs.",
    )
    configured = bool(Config.LANGFUSE_PUBLIC_KEY and Config.LANGFUSE_SECRET_KEY)
    status_line("Tracing", "configured" if configured else "not configured")

    host = Config.LANGFUSE_BASE_URL or Config.LANGFUSE_HOST
    if host:
        st.link_button("Open Langfuse dashboard", host, type="primary")
    elif not configured:
        st.info(
            "Set Langfuse credentials and a host in deployment configuration to enable trace links."
        )

    if configured and st.button("Check tracing connection"):
        try:
            client = get_langfuse_client()
            if client and client.auth_check():
                st.success("Langfuse authenticated successfully.")
            else:
                st.error(
                    "Langfuse could not authenticate with the configured credentials."
                )
        except Exception as exc:
            st.error(f"Tracing health check failed: {exc}")

    st.subheader("This browser session")
    runs = list(reversed(st.session_state.get("recent_runs", [])))
    if not runs:
        st.caption("Start a command to create a trace link here.")
        return

    for run in runs:
        columns = st.columns([2, 1, 1])
        columns[0].code(run.get("run_id", "unknown")[:12], language=None)
        columns[1].caption(f"Profile r{run.get('profile_revision', 0)}")
        if run.get("trace_url"):
            columns[2].link_button("Open trace", run["trace_url"], width="stretch")
        else:
            columns[2].caption("Trace unavailable")
