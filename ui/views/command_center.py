# ABOUTME: Primary Heathcliff operator workspace for chat, live runs, and approvals.

from __future__ import annotations

import uuid
from typing import Any

import streamlit as st

from config import Config
from core.subagents.music.tools import get_current_playback_snapshot
from ui.components import event_row, page_heading, status_line
from utils.heathcliff_greetings import generate_greeting


def _ensure_session() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("conversation_id", str(uuid.uuid4()))
    st.session_state.setdefault("run_timeline", [])
    st.session_state.setdefault("recent_runs", [])
    st.session_state.setdefault(
        "heathcliff_greeting",
        generate_greeting(user_name=Config.MASTER_INFO.get("name", "Sir")),
    )


def _append_timeline(event: dict[str, Any]) -> None:
    if event.get("type") != "response":
        st.session_state.run_timeline.append(event)
        st.session_state.run_timeline = st.session_state.run_timeline[-30:]


def _resume_approval(
    runtime: Any, approved: bool, modified_input: str | None = None
) -> None:
    approval = st.session_state["pending_approval"]
    try:
        handle = runtime.agent_for_revision(approval["profile_revision"])
        response = handle.agent.resume_approval(
            conversation_id=approval["session_id"],
            user_input=approval["user_input"],
            approved=approved,
            modified_input=modified_input,
            execution_events=approval.get("execution_events", []),
        )
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.pop("pending_approval", None)
        st.success("Action resumed." if approved else "Action rejected.")
        st.rerun()
    except Exception as exc:
        st.error(f"Unable to resume the action: {exc}")


def _render_approval(runtime: Any) -> None:
    approval = st.session_state.get("pending_approval")
    if not approval:
        return

    st.warning("Approval required before Heathcliff can continue.")
    actions = approval.get("actions", [])
    if actions:
        st.dataframe(actions, hide_index=True, width="stretch")
    else:
        st.code(approval.get("tool_input", ""), language="text")

    with st.expander("Review or modify the action", expanded=True):
        revised = st.text_area(
            "Action instructions",
            value=approval.get("tool_input", ""),
            key="approval_modified_input",
            height=120,
        )
        approve, reject = st.columns(2)
        with approve:
            if st.button("Approve and continue", type="primary", width="stretch"):
                _resume_approval(runtime, approved=True, modified_input=revised)
        with reject:
            if st.button("Reject action", width="stretch"):
                _resume_approval(runtime, approved=False)


def _render_run_timeline() -> None:
    timeline = st.session_state.get("run_timeline", [])
    if not timeline:
        st.caption("No activity in this browser session yet.")
        return
    for event in reversed(timeline[-8:]):
        st.markdown(event_row(event))


def _render_spotify_playback(playback: dict[str, Any]) -> None:
    """Render compact, verified Spotify state beneath its assistant turn."""
    cover, details = st.columns([1, 6], vertical_alignment="center")
    with cover:
        if playback.get("cover_url"):
            st.image(playback["cover_url"], width=88)
    with details:
        st.caption(f"Spotify · {playback['status']} on {playback['device']}")
        st.markdown(f"**{playback['name']}**")
        st.caption(f"{playback['artist']} · {playback['album']}")


def render() -> None:
    runtime = st.session_state["app_runtime"]
    _ensure_session()
    profile, revision = runtime.snapshot()

    page_heading(
        "Operate",
        "Command Center",
        "Run requests, review agent work, and approve consequential actions.",
    )
    status_line(
        "Active profile",
        f"revision {revision} · {len(profile.enabled_agents)} capabilities enabled",
    )

    controls = st.columns([1, 4])
    if controls[0].button("Start fresh conversation"):
        st.session_state.messages = []
        st.session_state.run_timeline = []
        st.session_state.conversation_id = str(uuid.uuid4())
        st.rerun()
    latest = st.session_state.get("recent_runs", [])
    if latest and latest[-1].get("trace_url"):
        controls[1].link_button("Open latest Langfuse trace", latest[-1]["trace_url"])

    with st.expander("Current run activity", expanded=True):
        _render_run_timeline()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("spotify_playback"):
                _render_spotify_playback(message["spotify_playback"])

    if not st.session_state.messages:
        with st.chat_message("assistant"):
            st.markdown(st.session_state.heathcliff_greeting)

    if st.session_state.get("pending_approval"):
        _render_approval(runtime)
        return

    prompt = st.chat_input("Give Heathcliff an instruction")
    if not prompt:
        return

    handle = runtime.current_agent()
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_box = st.empty()
        final_response = ""
        music_used = False
        with st.status("Heathcliff is working", expanded=True) as status:
            try:
                for event in handle.agent.stream_invoke(
                    prompt, conversation_id=st.session_state.conversation_id
                ):
                    _append_timeline(event)
                    event_type = event.get("type")
                    if event_type == "run_started":
                        st.session_state.recent_runs.append(event["data"])
                        st.session_state.recent_runs = st.session_state.recent_runs[
                            -20:
                        ]
                    elif event_type in {"plan", "dispatch"}:
                        status.update(
                            label=event.get("message", "Working"), state="running"
                        )
                    elif event_type == "subtask_complete":
                        st.write(event.get("message", "Subtask complete"))
                        music_used = (
                            music_used
                            or event.get("data", {}).get("agent") == "music_agent_tool"
                        )
                    elif event_type == "approval_required":
                        pending = dict(event.get("data", {}))
                        pending["user_input"] = prompt
                        pending["profile_revision"] = handle.revision
                        pending["execution_events"] = list(
                            st.session_state.run_timeline
                        )
                        st.session_state.pending_approval = pending
                        status.update(label="Awaiting approval", state="complete")
                    elif event_type == "response":
                        final_response = str(event.get("data", ""))
                    elif event_type == "error":
                        final_response = str(
                            event.get("data", event.get("message", ""))
                        )
                        status.update(label="Run failed", state="error")

                if st.session_state.get("pending_approval"):
                    response_box.info("The proposed action is ready for your review.")
                elif final_response:
                    response_box.markdown(final_response)
                    assistant_message = {
                        "role": "assistant",
                        "content": final_response,
                    }
                    if music_used:
                        playback = get_current_playback_snapshot()
                        if playback:
                            assistant_message["spotify_playback"] = playback
                            _render_spotify_playback(playback)
                    st.session_state.messages.append(assistant_message)
                    status.update(label="Complete", state="complete")
                else:
                    response_box.error("Heathcliff did not return a response.")
            except Exception as exc:
                status.update(label="Run failed", state="error")
                response_box.error(f"Run failed: {exc}")
