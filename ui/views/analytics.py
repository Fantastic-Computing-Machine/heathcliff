# ABOUTME: Operational analytics assembled from local conversation history and session runs.

from __future__ import annotations

from collections import Counter
from typing import Any

import streamlit as st

from ui.components import page_heading


def _event_rows(
    memory: Any, conversations: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for conversation in conversations:
        conversation_id = conversation["conversation_id"]
        for message in memory.get_conversation_history(conversation_id):
            for event in message.get("execution_events", []):
                rows.append(
                    {
                        "Time": str(message.get("created_at", ""))[:19],
                        "Conversation": conversation_id[:8],
                        "Event": event.get("type", "event"),
                        "Detail": event.get("message", ""),
                        "Data": event.get("data", {}),
                    }
                )
    return sorted(rows, key=lambda row: row["Time"], reverse=True)


def render() -> None:
    runtime = st.session_state["app_runtime"]
    memory = runtime.memory
    profile, revision = runtime.snapshot()
    page_heading(
        "Observe",
        "Analytics",
        "Detailed local operational activity. Langfuse remains the authoritative trace system.",
    )

    conversations = memory.get_all_conversations()
    event_rows = _event_rows(memory, conversations)
    event_counts = Counter(row["Event"] for row in event_rows)
    metrics = st.columns(4)
    metrics[0].metric("Conversations", len(conversations))
    metrics[1].metric(
        "Messages", sum(item.get("msg_count", 0) for item in conversations)
    )
    metrics[2].metric("Recorded execution events", len(event_rows))
    metrics[3].metric("Active profile revision", revision)

    st.subheader("Event breakdown")
    if event_counts:
        st.dataframe(
            [
                {"Event": event, "Count": count}
                for event, count in event_counts.most_common()
            ],
            hide_index=True,
            width="stretch",
        )
    else:
        st.info(
            "Execution analytics will appear after new completed turns are recorded."
        )

    st.subheader("Recent activity")
    if not event_rows:
        st.caption("No recorded execution activity yet.")
        return
    st.dataframe(event_rows[:200], hide_index=True, width="stretch", height=600)
    st.caption(
        f"Showing the newest {min(len(event_rows), 200)} events across all conversations. "
        f"Profile r{revision} has {len(profile.enabled_agents)} enabled capabilities."
    )
