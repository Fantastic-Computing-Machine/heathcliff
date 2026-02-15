# ABOUTME: Streamlit multipage app - Analytics and insights page
# ABOUTME: View usage statistics and conversation patterns

import os
import sys
from datetime import datetime

import pandas as pd
import streamlit as st

# Add parent directory to path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from ui.shared import get_memory_manager

st.set_page_config(page_title="Analytics", page_icon="📊", layout="wide")


memory = get_memory_manager()

# Header
st.title("📊 Usage Analytics")
st.markdown("Insights into your interactions with Heathcliff")

# Overview metrics
stats = memory.get_stats()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "💾 Total Memories",
        stats["memories"],
        help="Long-term facts and preferences stored",
    )

with col2:
    st.metric("💬 Chat Messages", stats["chats"], help="Total conversation turns")

with col3:
    st.metric(
        "📄 Documents Indexed",
        stats["documents"],
        help="Emails, files, and other documents",
    )

st.markdown("---")

# Two column layout
col_left, col_right = st.columns(2)

# Recent Activity
with col_left:
    st.subheader("🕒 Recent Activity")

    recent_chats = memory.chats.get(limit=20)

    if recent_chats and recent_chats.get("metadatas"):
        activity_data = []
        for doc, meta in zip(recent_chats["documents"], recent_chats["metadatas"]):
            activity_data.append(
                {
                    "Role": meta.get("role", "unknown").capitalize(),
                    "Message": doc[:50] + ("..." if len(doc) > 50 else ""),
                    "Session": meta.get("session", "unknown")[:8],
                    "Time": (
                        meta.get("timestamp", "unknown")[:19]
                        if meta.get("timestamp")
                        else "N/A"
                    ),
                }
            )

        if activity_data:
            df = pd.DataFrame(activity_data)
            st.dataframe(df, width="content", hide_index=True)
    else:
        st.info("No recent activity to display")

# Memory Distribution
with col_right:
    st.subheader("🗂️ Memory Distribution")

    all_memories = memory.memories.get(limit=300)

    if all_memories and all_memories.get("metadatas"):
        categories = {}
        for meta in all_memories["metadatas"]:
            cat = meta.get("category", "general")
            categories[cat] = categories.get(cat, 0) + 1

        if categories:
            df = pd.DataFrame(
                list(categories.items()), columns=["Category", "Count"]
            ).sort_values("Count", ascending=False)

            st.bar_chart(df.set_index("Category"))

            st.caption(f"**Total Categories**: {len(categories)}")
        else:
            st.info("No memories categorized yet")
    else:
        st.info("No memories to analyze")

st.markdown("---")

# Session Analytics
st.subheader("📈 Session Analytics")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Unique Sessions**")

    all_chats = memory.chats.get(limit=300)

    if all_chats and all_chats.get("metadatas"):
        sessions = set(
            meta.get("session", "unknown") for meta in all_chats["metadatas"]
        )
        st.metric("Total Sessions", len(sessions))

        st.caption("A session is a continuous conversation period")
    else:
        st.info("No session data available")

with col2:
    st.markdown("**Message Distribution**")

    if all_chats and all_chats.get("metadatas"):
        roles = {}
        for meta in all_chats["metadatas"]:
            role = meta.get("role", "unknown")
            roles[role] = roles.get(role, 0) + 1

        role_df = pd.DataFrame(list(roles.items()), columns=["Role", "Messages"])

        st.dataframe(role_df, width="content", hide_index=True)
    else:
        st.info("No message data available")

# Footer
st.markdown("---")
st.caption(f"📅 Dashboard generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
