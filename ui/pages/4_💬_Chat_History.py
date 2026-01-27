# ABOUTME: Streamlit multipage app - Chat History page
# ABOUTME: View and manage past conversations

import os
import sys

import streamlit as st

# Add parent directory to path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from config import Config
from ui.shared import get_memory_manager

st.set_page_config(page_title="Chat History", page_icon="💬", layout="wide")

memory = get_memory_manager()

st.title("💬 Chat History")

# Initialize session state for confirmations
if "confirm_delete_all" not in st.session_state:
    st.session_state["confirm_delete_all"] = False

# Sidebar for Filtering
with st.sidebar:
    st.header("Filters")

    # Refresh button
    if st.button("🔄 Refresh Data"):
        st.cache_resource.clear()
        st.rerun()

    st.divider()

    # Get sessions
    sessions = memory.get_all_sessions()

    selected_session_id = None
    if not sessions:
        st.warning("No chat history found.")
    else:
        # Format session options
        # Sort by start time descending
        sessions.sort(key=lambda x: x["start_time"], reverse=True)

        session_options = {
            f"{s['start_time'][:19]} ({s['msg_count']} msgs)": s["session_id"]
            for s in sessions
        }

        selected_option = st.selectbox(
            "Select Session", options=list(session_options.keys())
        )
        selected_session_id = (
            session_options[selected_option] if selected_option else None
        )

    # Global Actions
    st.divider()
    st.subheader("⚠️ Danger Zone")

    if st.button("🧨 Delete ALL History", type="primary", width="content"):
        st.session_state["confirm_delete_all"] = True

    if st.session_state["confirm_delete_all"]:
        st.warning("Are you sure? This cannot be undone.")
        col_yes, col_no = st.columns(2)
        with col_yes:
            if st.button("Yes, Delete", width="content"):
                if memory.delete_all_chats():
                    st.success("All chat history deleted!")
                    st.session_state["confirm_delete_all"] = False
                    st.rerun()
                else:
                    st.error("Failed to delete history.")
        with col_no:
            if st.button("Cancel", width="content"):
                st.session_state["confirm_delete_all"] = False
                st.rerun()

# Main Content
if selected_session_id:
    # Display Session Info
    col_header, col_action = st.columns([3, 1])
    with col_header:
        st.subheader(f"Session ID: `{selected_session_id}`")
    with col_action:
        if st.button("🗑️ Delete This Session", type="primary", width="content"):
            if memory.clear_session(selected_session_id):
                st.success("Session deleted successfully!")
                st.rerun()
            else:
                st.error("Failed to delete session.")

    st.divider()

    # Get messages
    messages = memory.get_session_history(selected_session_id)

    if messages:
        for msg in messages:
            role = msg.get("role", "unknown")
            timestamp = msg.get("timestamp", "")
            content = msg.get("content", "")

            # Map role to avatar/name
            if role == "user":
                avatar = "👤"
            else:
                avatar = "🐦‍🔥"  # Heathcliff icon

            with st.chat_message(role, avatar=avatar):
                st.markdown(content)
                st.caption(f"_{timestamp[:19]}_")
    else:
        st.info("No messages found in this session.")

elif not sessions:
    st.info("No chat history available. Start a conversation in the Home page!")
else:
    st.info("👈 Select a session from the sidebar to view conversation history.")
