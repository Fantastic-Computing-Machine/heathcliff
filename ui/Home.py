# ABOUTME: Streamlit multipage app - Home page with chat interface
# ABOUTME: Main entry point for the Heathcliff dashboard

import os
import sys

import streamlit as st

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
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


# Chat input logic
# chat_greeting: str = generate_greeting(user_name="Adi", include_weather=True)
chat_greeting: str = "Hi, what are we doing today?"
prompt = st.chat_input(placeholder=chat_greeting)

if prompt:
    prompt = prompt.strip()
    logger.info(f"Processing prompt: {prompt[:50]}...")

    # Add user message to session state and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get agent response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()

        try:
            logger.info("Starting agent invoke...")

            # Use status container for live thinking steps
            with st.status("🚀 Processing...", expanded=True) as status:
                st_callback = StatusCallbackHandler(status_container=status)

                full_response = agent.invoke(
                    prompt,
                    session_id=st.session_state.session_id,
                    additional_callbacks=[st_callback],
                )

                status.update(label="✅ Complete", state="complete", expanded=False)

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

        except Exception as e:
            logger.error(f"Error during agent invocation: {e}", exc_info=True)
            error_msg = f"Error: {str(e)}"
            response_placeholder.markdown(error_msg)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_msg}
            )

    # Rerun to clear the input and prepare for next message
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
