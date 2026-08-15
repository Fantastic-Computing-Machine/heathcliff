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

from db.memory_manager import MemoryManager

st.set_page_config(page_title="Analytics", page_icon="📊", layout="wide")


# Initialize memory
@st.cache_resource
def get_memory():
    return MemoryManager()


memory = get_memory()

# Header
st.title("📊 Usage Analytics")
st.markdown("Insights into your interactions with Heathcliff")

# Overview metrics
stats = memory.get_stats()

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "💾 Total Memories",
        stats["memories"],
        help="Long-term facts and preferences stored",
    )

with col2:
    st.metric("💬 Chat Messages", stats["chats"], help="Total conversation turns")

st.markdown("---")

# Two column layout
col_left, col_right = st.columns(2)

# Recent Activity
with col_left:
    st.subheader("🕒 Recent Activity")

    conversations = memory.get_all_conversations()
    recent_conversations = sorted(
        conversations, key=lambda x: x.get("start_time", ""), reverse=True
    )[:20]

    if recent_conversations:
        activity_data = [
            {
                "Conversation": c["conversation_id"][:8],
                "Messages": c["msg_count"],
                "Started": c.get("start_time", "")[:19] or "N/A",
            }
            for c in recent_conversations
        ]
        st.dataframe(pd.DataFrame(activity_data), width="content", hide_index=True)
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

# Conversation Analytics
st.subheader("📈 Conversation Analytics")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Unique Conversations**")

    all_conversations = memory.get_all_conversations()

    if all_conversations:
        st.metric("Total Conversations", len(all_conversations))
        st.caption("A conversation is a continuous interaction period")
    else:
        st.info("No conversation data available")

with col2:
    st.markdown("**Message Distribution**")

    if all_conversations:
        total_msgs = sum(c.get("msg_count", 0) for c in all_conversations)
        avg_msgs = total_msgs / len(all_conversations) if all_conversations else 0
        role_df = pd.DataFrame(
            [
                {"Metric": "Total messages", "Value": total_msgs},
                {"Metric": "Avg per conversation", "Value": round(avg_msgs, 1)},
            ]
        )
        st.dataframe(role_df, width="content", hide_index=True)
    else:
        st.info("No message data available")

# Footer
st.markdown("---")
st.caption(f"📅 Dashboard generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
