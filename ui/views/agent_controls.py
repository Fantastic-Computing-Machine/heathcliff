# ABOUTME: Shared process-wide, non-secret controls for Heathcliff's next runs.

from __future__ import annotations

import streamlit as st

from core.runtime_profile import DEFAULT_AGENT_NAMES
from core.subagents.music.tools import (
    complete_spotify_authorization,
    spotify_authorization_url,
    spotify_is_connected,
)
from ui.components import page_heading, status_line


def _set_form_defaults(profile, revision: int) -> None:
    if st.session_state.get("profile_form_revision") == revision:
        return
    st.session_state.update(
        {
            "profile_form_revision": revision,
            "supervisor_model": profile.supervisor_model,
            "tool_model": profile.tool_model,
            "temperature": profile.temperature,
            "max_tokens": profile.max_tokens,
            "max_tasks": profile.max_tasks,
            "per_task_timeout": profile.per_task_timeout_ms // 1000,
            "total_runtime": profile.max_total_runtime_ms // 1000,
            "enabled_agents": list(profile.enabled_agents),
        }
    )


def _render_spotify_connection() -> None:
    st.subheader("Spotify connection")
    if spotify_is_connected():
        st.success("Spotify is connected.")
        return

    try:
        authorization_url = spotify_authorization_url()
    except ValueError as exc:
        st.info(str(exc))
        return

    st.caption(
        "Connect Spotify here before asking Heathcliff to play music. "
        "The callback page may not load; copy its address and paste it below."
    )
    st.link_button("Authorize Spotify", authorization_url)
    with st.form("spotify_connection", clear_on_submit=True):
        redirect_url = st.text_input(
            "Spotify callback URL", type="password", autocomplete="off"
        )
        submitted = st.form_submit_button("Save Spotify connection")
    if submitted:
        try:
            complete_spotify_authorization(redirect_url)
            st.success("Spotify connected. Music requests are ready.")
        except ValueError as exc:
            st.error(str(exc))
        except Exception:
            st.error("Spotify could not complete authorization. Please try again.")


def render() -> None:
    runtime = st.session_state["app_runtime"]
    profile, revision = runtime.snapshot()
    _set_form_defaults(profile, revision)

    page_heading(
        "Manage",
        "Agent Controls",
        "Shared process settings apply to new runs only and reset when Heathcliff restarts.",
    )
    status_line(
        "Active profile",
        f"revision {revision}; existing runs retain their original settings",
    )

    with st.form("runtime_profile"):
        models, limits = st.columns(2, gap="large")
        with models:
            st.subheader("Models")
            st.text_input("Supervisor model", key="supervisor_model")
            st.text_input("Tool model", key="tool_model")
            st.slider("Temperature", 0.0, 2.0, step=0.05, key="temperature")
            st.number_input(
                "Max response tokens", 256, 32768, step=256, key="max_tokens"
            )
        with limits:
            st.subheader("Execution limits")
            st.number_input("Maximum subtasks", 1, 20, key="max_tasks")
            st.number_input(
                "Per-task timeout (seconds)", 5, 600, key="per_task_timeout"
            )
            st.number_input("Total runtime (seconds)", 30, 900, key="total_runtime")

        st.subheader("Enabled capabilities")
        st.multiselect(
            "The planner can use only these registered capabilities",
            DEFAULT_AGENT_NAMES,
            key="enabled_agents",
        )
        submitted = st.form_submit_button(
            "Apply runtime profile", type="primary", width="content"
        )

    if submitted:
        try:
            _, applied_revision = runtime.apply(
                {
                    "supervisor_model": st.session_state.supervisor_model,
                    "tool_model": st.session_state.tool_model,
                    "temperature": st.session_state.temperature,
                    "max_tokens": st.session_state.max_tokens,
                    "max_tasks": st.session_state.max_tasks,
                    "per_task_timeout_ms": st.session_state.per_task_timeout * 1000,
                    "max_total_runtime_ms": st.session_state.total_runtime * 1000,
                    "enabled_agents": st.session_state.enabled_agents,
                }
            )
            st.success(
                f"Profile revision {applied_revision} is ready for the next run."
            )
        except ValueError as exc:
            st.error(str(exc))

    if st.button("Reset to deployment defaults"):
        _, reset_revision = runtime.reset_profile()
        st.session_state.pop("profile_form_revision", None)
        st.success(f"Profile reset as revision {reset_revision}.")
        st.rerun()

    _render_spotify_connection()
