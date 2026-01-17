# ABOUTME: Streamlit multipage app - Home page with chat interface
# ABOUTME: Main entry point for the Heathcliff dashboard

import os
import sys

import streamlit as st

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from core.approval_handler import (
    StreamlitApprovalHandler,
    approve_request,
    clear_approval,
    is_approval_pending,
    reject_request,
)
from logger import logger
from ui.shared import (
    clear_chat_session,
    get_agent,
    get_memory_manager,
    init_session_state,
)
from ui.streamlit_callback import StatusCallbackHandler
from utils.heathcliff_greetings import generate_greeting

# Page config
st.set_page_config(
    page_title="Heathcliff Dashboard",
    page_icon="🐦‍🔥",
    layout="centered",
    initial_sidebar_state="expanded",
    # my email in menu with github repo link.
    menu_items={
        "About": "Heathcliff - Your voice-activated AI assistant powered by Gemini",
    },
)

# Initialize Session State
init_session_state()

# Initialize components
try:
    memory = get_memory_manager()
    agent = get_agent(memory)
    initialization_success = True
except Exception as e:
    initialization_success = False
    error_msg = str(e)


if initialization_success:
    # Stats
    stats = memory.get_stats()
    st.sidebar.metric("💾 Memories", stats["memories"])
    st.sidebar.metric("💬 Conversations", stats["chats"])
    st.sidebar.metric("📄 Documents", stats["documents"])

    st.sidebar.markdown("---")
    st.sidebar.info(f"**Status**: ✅ Ready\n\n**Model**: {Config.MODEL}")
else:
    st.sidebar.error(f"⚠️ Initialization failed")
    st.sidebar.code(error_msg)


# Main content
st.title("Heathcliff")
st.info("Your voice-activated AI assistant powered by Gemini", icon="🧑‍🏫")

if not initialization_success:
    st.error(f"Failed to initialize Heathcliff: {error_msg}")
    st.info("Please check your configuration and API keys in `.env` file.")
    st.stop()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Check for pending approval requests
if is_approval_pending(st.session_state):
    approval = st.session_state["pending_approval"]
    tool_name = approval.get("tool_name", "Unknown Tool")
    tool_input = approval.get("tool_input", "")

    st.warning(f"### 🔒 Approval Required: **{tool_name}**", icon="⚠️")

    with st.expander("Tool Execution Details", expanded=False, icon="💡"):
        st.markdown(f"**Tool**: `{tool_name}`")
        st.markdown(f"**Arguments**:")
        st.code(tool_input, language="text")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("✅ Approve", key="approve_btn", width="content"):
                approve_request(st.session_state)
                # Do NOT clear approval here; let agent consume it
                st.success("Tool execution approved!")
                st.rerun()

        with col2:
            if st.button("✏️ Modify", key="modify_btn", width="content"):
                # Show modification form
                st.session_state["show_modify_form"] = True

        with col3:
            if st.button("❌ Reject", key="reject_btn", width="content"):
                reject_request(st.session_state)
                # Do NOT clear approval here; let agent consume it (or just clear on reject)
                # Actually for reject, we can clear it immediately or let re-run handle it
                # Logic in approval_handler handles rejection consumption too
                st.error("Tool execution rejected!")
                # Add rejection message to chat
                rejection_msg = (
                    f"I won't execute {tool_name} as you rejected the operation."
                )
                st.session_state.messages.append(
                    {"role": "assistant", "content": rejection_msg}
                )
                st.rerun()

    # Show modification form if requested
    if st.session_state.get("show_modify_form"):
        with st.form("modify_tool_args"):
            st.markdown("### Edit Tool Arguments")
            modified_input = st.text_area(
                "Edit the arguments below:",
                value=tool_input,
                height=150,
            )

            col_submit, col_cancel = st.columns(2)
            with col_submit:
                if st.form_submit_button("✅ Submit", width="content"):
                    approve_request(st.session_state, modified_input=modified_input)
                    st.session_state["show_modify_form"] = False
                    st.success("Tool execution approved with modifications!")
                    st.rerun()

            with col_cancel:
                if st.form_submit_button("❌ Cancel", width="content"):
                    st.session_state["show_modify_form"] = False
                    st.rerun()

# Chat input logic with Resume Capability
chat_input_message = st.chat_input(
    generate_greeting(user_name="Adi", include_weather=True),
)

prompt = None
is_resuming = False

if chat_input_message:
    prompt = str(chat_input_message).strip()
    st.session_state["last_prompt"] = prompt  # Save for resumption
elif "last_prompt" in st.session_state and "pending_approval" in st.session_state:
    # Check if we should resume after approval
    if st.session_state["pending_approval"].get("status") == "approved":
        prompt = st.session_state["last_prompt"]
        is_resuming = True
        logger.info("Resuming execution with approved prompt")

if prompt:
    logger.info(f"Processing prompt: {prompt[:50]}...")

    # Add user message to session state ONLY if not resuming (it's already there)
    if not is_resuming:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

    # Get agent response
    with st.chat_message("assistant"):
        # Placeholder for final response
        response_placeholder = st.empty()

        try:
            logger.info("Starting agent invoke...")

            # Create approval handler for this request
            approval_handler = StreamlitApprovalHandler(st.session_state)

            # Use status container for tool outputs
            with st.status("Heathcliff is thinking...", expanded=True) as status:
                st_callback = StatusCallbackHandler(status_container=status)

                full_response = agent.invoke(
                    prompt,
                    session_id=st.session_state.session_id,
                    additional_callbacks=[approval_handler, st_callback],
                )
                status.update(label="Finished!", state="complete", expanded=False)

            logger.info(f"Agent response received: {full_response[:50]}...")

            # Display response and save to session state
            if full_response:
                response_placeholder.markdown(full_response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )
            else:
                error_msg = "I encountered an error processing your request."
                response_placeholder.markdown(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )

            # Clear last prompt after successful execution
            if "last_prompt" in st.session_state:
                del st.session_state["last_prompt"]

        except Exception as e:
            # Check if this was just an approval interruption
            if is_approval_pending(st.session_state):
                st.rerun()

            logger.error(f"Error during agent invocation: {e}", exc_info=True)
            error_msg = f"Error: {str(e)}"
            response_placeholder.markdown(error_msg)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_msg}
            )

    # Rerun to display the updated chat from session state
    st.rerun()


# --------------------------------------------
# Sidebar controls
# --------------------------------------------

with st.sidebar:
    st.markdown("---")
    st.subheader("💬 Chat Controls")

    if st.button("📋 Copy Conversation", width="content"):
        # Format conversation as text
        conversation_text = ""
        for msg in st.session_state.messages:
            role = "You" if msg["role"] == "user" else "Heathcliff"
            conversation_text += f"{role}: {msg['content']}\n\n"

        # Copy to clipboard using Streamlit's built-in clipboard
        if conversation_text:
            st.code(conversation_text, language="text")
            st.success("Conversation formatted above - copy manually")
        else:
            st.info("No conversation to copy yet")

    if st.button("🗑️ Clear Chat History", width="content"):
        clear_chat_session()
        st.rerun()

    if st.button("🔄 New Session", width="content"):
        clear_chat_session()
        st.success("New session started!")
        st.rerun()

    if st.button("♻️ Reload Agent", width="content"):
        # Clear cache and reload agent
        st.cache_resource.clear()
        st.success("Agent reloaded!")
        st.rerun()
