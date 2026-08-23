# ABOUTME: Conversation browser with persistent per-turn execution timelines.

from __future__ import annotations

from typing import Any

import streamlit as st

from ui.components import event_row, page_heading


def _conversation_label(item: dict[str, Any]) -> str:
    return (
        f"{item.get('start_time', '')[:19]} · {item.get('msg_count', 0)} messages · "
        f"{item['conversation_id'][:8]}"
    )


def _render_execution_events(events: list[dict[str, Any]]) -> None:
    if not events:
        return
    with st.expander(f"Execution details ({len(events)} events)"):
        for event in events:
            st.markdown(event_row(event))
            data = event.get("data")
            if data and data not in ({}, ""):
                st.json(data, expanded=False)


def render() -> None:
    runtime = st.session_state["app_runtime"]
    memory = runtime.memory
    page_heading(
        "Knowledge",
        "Conversations",
        "Read complete transcripts, including the execution details recorded for each turn.",
    )

    conversations = sorted(
        memory.get_all_conversations(),
        key=lambda item: item.get("start_time", ""),
        reverse=True,
    )
    query = st.text_input(
        "Filter conversations", placeholder="Date, ID, or message count"
    )
    if query:
        needle = query.lower()
        conversations = [
            item
            for item in conversations
            if needle in item.get("conversation_id", "").lower()
            or needle in item.get("start_time", "").lower()
            or needle in str(item.get("msg_count", ""))
        ]

    if not conversations:
        st.info("No conversations match this filter.")
        return

    st.dataframe(
        [
            {
                "Started": item.get("start_time", "")[:19],
                "Messages": item.get("msg_count", 0),
                "Conversation ID": item["conversation_id"],
            }
            for item in conversations
        ],
        hide_index=True,
        width="stretch",
        height=min(420, 44 * (len(conversations) + 1)),
    )
    selected_id = st.selectbox(
        "Open conversation",
        options=[item["conversation_id"] for item in conversations],
        format_func=lambda conversation_id: _conversation_label(
            next(
                item
                for item in conversations
                if item["conversation_id"] == conversation_id
            )
        ),
    )
    messages = memory.get_conversation_history(selected_id)
    controls = st.columns([1, 1, 3])
    if controls[0].button("Continue in Command Center", type="primary"):
        st.session_state.conversation_id = selected_id
        st.session_state.messages = [
            {"role": item.get("role", "assistant"), "content": item.get("content", "")}
            for item in messages
        ]
        st.switch_page("Home.py")
    if controls[1].button("Prepare deletion"):
        st.session_state["delete_conversation_id"] = selected_id
    if st.session_state.get("delete_conversation_id") == selected_id:
        st.warning("Deleting this conversation cannot be undone.")
        confirm, cancel = st.columns(2)
        if confirm.button("Delete conversation", type="primary"):
            memory.clear_conversation(selected_id)
            st.session_state.pop("delete_conversation_id", None)
            st.rerun()
        if cancel.button("Cancel"):
            st.session_state.pop("delete_conversation_id", None)
            st.rerun()

    st.divider()
    for message in messages:
        with st.chat_message(message.get("role", "assistant")):
            st.markdown(message.get("content", ""))
            st.caption(str(message.get("created_at", ""))[:19])
        _render_execution_events(message.get("execution_events", []))
