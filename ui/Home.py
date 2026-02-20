# ABOUTME: Streamlit multipage app - Home page with blob-centered chat interface
# ABOUTME: Main entry point for the Heathcliff dashboard

import os
import sys
import time
from typing import Any, List

import streamlit as st

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime

from config import Config
from core.agent_core import HeathcliffAgent
from core.approval_handler import (
    StreamlitApprovalHandler,
    approve_request,
    clear_approval,
    is_approval_pending,
    reject_request,
)
from core.memory_manager import MemoryManager
from logger import logger
from ui.blob import blob
from utils.errors import AgentInitializationError
from utils.heathcliff_greetings import generate_greeting

# Page config -- wide layout so the blob gets maximum space
st.set_page_config(
    page_title="Heathcliff",
    page_icon="🐦‍🔥",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        "About": "Heathcliff - Your voice-activated AI assistant powered by Gemini",
    },
)

# ── Hide Streamlit chrome for immersive blob experience ──────────────
st.markdown(
    """
    <style>
    /* Hide header, footer, and hamburger menu for clean look */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* Remove default padding so blob fills the viewport */
    .stMainBlockContainer {
        padding-top: 0 !important;
        padding-bottom: 0 !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
        max-width: 100% !important;
    }
    section[data-testid="stSidebar"] {
        background: rgba(10, 10, 15, 0.95);
    }

    /* Make the blob iframe fill the space */
    iframe[title="ui.blob.heathcliff_blob"] {
        border: none !important;
        width: 100vw !important;
        height: 100vh !important;
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        z-index: 999 !important;
    }

    /* Hide block container outlines */
    .stElementContainer {
        margin: 0 !important;
        padding: 0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Initialize components ────────────────────────────────────────────
@st.cache_resource
def init_components():
    """Initialize memory and agent."""
    try:
        memory = MemoryManager()
        agent = HeathcliffAgent(memory_manager=memory)
    except Exception as e:
        logger.error(f"Error initializing components: {e}")
        raise AgentInitializationError(
            f"Failed to initialize agent components: {e}"
        ) from e
    return memory, agent


try:
    memory, agent = init_components()
    initialization_success = True
except Exception as e:
    initialization_success = False
    error_msg = str(e)

# ── Sidebar (collapsed by default, houses stats & controls) ──────────
if initialization_success:
    stats = memory.get_stats()
    st.sidebar.metric("Memories", stats["memories"])
    st.sidebar.metric("Conversations", stats["chats"])
    st.sidebar.metric("Documents", stats["documents"])
    st.sidebar.markdown("---")
    st.sidebar.info(f"**Status**: Ready\n\n**Model**: {Config.MODEL}")
else:
    st.sidebar.error("Initialization failed")
    st.sidebar.code(error_msg)

# ── Session state init ───────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_start_time" not in st.session_state:
    st.session_state.session_start_time = datetime.now()

if "session_id" not in st.session_state:
    import uuid

    st.session_state.session_id = str(uuid.uuid4())
    logger.info(f"Created new session: {st.session_state.session_id}")

if "blob_state" not in st.session_state:
    st.session_state.blob_state = "idle"

if "my_greeting" not in st.session_state:
    st.session_state.my_greeting = generate_greeting(
        user_name="Adi", include_weather=True
    )

# ── Error guard ──────────────────────────────────────────────────────
if not initialization_success:
    st.error(f"Failed to initialize Heathcliff: {error_msg}")
    st.info("Please check your configuration and API keys in `.env` file.")
    st.stop()

# ── Render the 3D Blob (takes the full viewport) ────────────────────
blob(state=st.session_state.blob_state, height=800, key="main_blob")

# ── Sidebar controls ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("---")
    st.subheader("Controls")

    if st.button("New Session"):
        st.session_state.messages = []
        st.session_state.session_start_time = datetime.now()
        import uuid

        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.blob_state = "idle"
        st.rerun()

    if st.button("Reload Agent"):
        st.cache_resource.clear()
        st.rerun()
