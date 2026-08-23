# ABOUTME: Small, default-Streamlit presentation helpers for the control panel.

from __future__ import annotations

from typing import Any

import streamlit as st


def page_heading(eyebrow: str, title: str, detail: str) -> None:
    st.title(title)
    st.caption(f"{eyebrow} · {detail}")


def status_line(label: str, value: str) -> None:
    st.info(f"{label}: {value}")


def event_row(event: dict[str, Any]) -> str:
    """Return a compact, readable timeline label from a coordinator event."""
    kind = event.get("type", "event").replace("_", " ").title()
    message = event.get("message") or ""
    if not message and event.get("type") == "response":
        message = "Response received"
    return f"**{kind}** — {message}".strip(" —")
