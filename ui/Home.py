# ABOUTME: Streamlit multipage app - Home page with chat interface
# ABOUTME: Main entry point for the Heathcliff dashboard

import streamlit as st
import sys
import os
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config
from core.memory_manager import MemoryManager
from core.agent_core import HeathcliffAgent
from tools import get_all_tools
from utils.heathcliff_greetings import generate_greeting
from datetime import datetime
from logger import logger


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


# Initialize components
@st.cache_resource
def init_components():
    """Initialize config, memory, and agent."""
    config = get_config()
    memory = MemoryManager()
    tools = get_all_tools()
    agent = HeathcliffAgent(config=config, memory_manager=memory, tools=tools)
    return config, memory, agent


try:
    config, memory, agent = init_components()
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
    st.sidebar.info(
        f"**Status**: ✅ Ready\n\n**Model**: {config.get('llm.model', 'N/A')}"
    )
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

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize session start time
if "session_start_time" not in st.session_state:
    st.session_state.session_start_time = datetime.now()

# Initialize greeting shown flag
if "greeting_shown" not in st.session_state:
    st.session_state.greeting_shown = False

# Initialize persistent session_id for maintaining conversation context
if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())
    logger.info(f"Created new session: {st.session_state.session_id}")

# Display Heathcliff's greeting ONCE at the start of the session
if not st.session_state.greeting_shown and len(st.session_state.messages) == 0:
    greeting = generate_greeting(user_name="Adi", include_weather=True)
    with st.chat_message("assistant"):
        st.markdown(f"*{greeting}*")
    st.session_state.greeting_shown = True
    # Don't add to messages - this is ephemeral greeting

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if chat_input_message := st.chat_input(
    "Hi, what's on your mind...",
    accept_file=True,
    accept_audio=True,
    file_type=["jpg", "jpeg", "png"],
):
    prompt = chat_input_message.get("text", "")
    file = chat_input_message.get("file", None)
    audio = chat_input_message.get("audio", None)

    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get agent response with streaming
    with st.chat_message("assistant"):
        # Container for streaming response
        response_placeholder = st.empty()
        full_response = ""

        # Status container for showing agent progress
        with st.status("Processing request...", expanded=True) as status:
            try:
                # Use persistent session_id to maintain conversation context
                for event in agent.stream_invoke(prompt, session_id=st.session_state.session_id):
                    if event["type"] == "status":
                        # Update status label with current phase
                        status.update(label=event["message"], state="running")

                    elif event["type"] == "tool":
                        # Show tool execution details
                        with status:
                            st.write(f"🛠️ {event['message']}")
                            if "data" in event and event["data"].get("args"):
                                with st.expander("Tool Details", expanded=False):
                                    st.json(event["data"])

                    elif event["type"] == "response":
                        # Capture the final response
                        status.update(label="Generating response...", state="running")
                        full_response = event["data"]

                    elif event["type"] == "complete":
                        # Mark as complete
                        status.update(label="✅ Complete", state="complete")

                        # Show tool usage summary
                        if event.get("data", {}).get("tools_used"):
                            tools = event["data"]["tools_used"]
                            st.caption(f"🛠️ Tools used: {', '.join(tools)}")

                    elif event["type"] == "error":
                        # Handle errors
                        status.update(label="❌ Error", state="error")
                        st.error(event["message"])
                        full_response = event["data"]
                        break

            except Exception as e:
                status.update(label="❌ Error", state="error")
                full_response = f"Error: {str(e)}"
                st.error(full_response)

        # Stream the response with typing effect (only for responses > 100 chars)
        if full_response:
            if len(full_response) > 100:
                # Word-by-word streaming
                words = full_response.split()
                displayed = []
                for word in words:
                    displayed.append(word)
                    response_placeholder.markdown(" ".join(displayed) + "▌")
                    time.sleep(0.02)  # Typing effect delay
                # Final display without cursor
                response_placeholder.markdown(full_response)
            else:
                # Short responses - display immediately
                response_placeholder.markdown(full_response)

            # Save to session state
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )
        else:
            # Fallback if no response
            error_msg = "I encountered an error processing your request."
            response_placeholder.markdown(error_msg)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_msg}
            )

# Sidebar controls
with st.sidebar:
    st.markdown("---")
    st.subheader("💬 Chat Controls")

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.greeting_shown = False  # Reset greeting for new session
        # Generate new session_id to clear conversation context
        import uuid
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    if st.button("🔄 New Session", use_container_width=True):
        # Create new session
        st.session_state.messages = []
        st.session_state.greeting_shown = False  # Reset greeting for new session
        st.session_state.session_start_time = datetime.now()  # Reset session time
        # Generate new session_id to clear conversation context
        import uuid
        st.session_state.session_id = str(uuid.uuid4())
        st.success("New session started!")
        st.rerun()

    if st.button("♻️ Reload Agent", use_container_width=True):
        # Clear cache and reload agent
        st.cache_resource.clear()
        st.success("Agent reloaded!")
        st.rerun()


# Footer
st.markdown("---")
st.caption(
    "💡 **Tip**: Use the sidebar to navigate to Memories, Analytics, and Settings pages"
)
