# ABOUTME: Streamlit multipage app - Chat History page
# ABOUTME: View and manage past conversations

import os
import sys

import streamlit as st

# Add parent directory to path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from db.memory_manager import MemoryManager

st.set_page_config(page_title="Chat History", page_icon="💬", layout="wide")


# Initialize memory
def get_memory():
    return MemoryManager()


memory = get_memory()

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

    # Get conversations
    conversations = memory.get_all_conversations()

    selected_conversation_id = None
    if not conversations:
        st.warning("No chat history found.")
    else:
        conversations.sort(key=lambda x: x["start_time"], reverse=True)

        conversation_options = {
            f"{c['start_time'][:19]} ({c['msg_count']} msgs)": c["conversation_id"]
            for c in conversations
        }

        selected_option = st.selectbox(
            "Select Conversation", options=list(conversation_options.keys())
        )
        selected_conversation_id = (
            conversation_options[selected_option] if selected_option else None
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
if selected_conversation_id:
    col_header, col_action = st.columns([3, 1])
    with col_header:
        st.subheader(f"Conversation ID: `{selected_conversation_id}`")
    with col_action:
        if st.button("🗑️ Delete This Conversation", type="primary", width="content"):
            if memory.clear_conversation(selected_conversation_id):
                st.success("Conversation deleted successfully!")
                st.rerun()
            else:
                st.error("Failed to delete conversation.")

    st.divider()

    messages = memory.get_conversation_history(selected_conversation_id)

    if messages:
        for msg in messages:
            role = msg.get("role", "unknown")
            timestamp = msg.get("created_at", "")
            content = msg.get("content", "")

            avatar = "👤" if role == "user" else "🐦‍🔥"

            with st.chat_message(role, avatar=avatar):
                st.markdown(content)
                st.caption(f"_{timestamp[:19]}_")
    else:
        st.info("No messages found in this conversation.")

elif not conversations:
    st.info("No chat history available. Start a conversation in the Home page!")
else:
    st.info("👈 Select a conversation from the sidebar to view history.")
