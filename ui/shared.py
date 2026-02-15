# ABOUTME: Shared utilities for Streamlit UI state and resource management
# ABOUTME: Centralizes initialization of MemoryManager and HeathcliffAgent to ensure singletons

import os
import sys
import uuid
from datetime import datetime
from typing import Any, List, Optional, Tuple

import streamlit as st

from config import Config
from core.agent_core import HeathcliffAgent
from core.memory_manager import MemoryManager
from logger import logger
from utils.errors import AgentInitializationError


@st.cache_resource(show_spinner="Initializing Memory Manager...")
def get_memory_manager() -> MemoryManager:
    """
    Get or initialize the MemoryManager singleton.
    Cached resource shared across all pages.

    Note: In most cases, you can just use get_agent() which auto-initializes memory.
    This function is kept for pages that only need memory access (e.g., Analytics).
    """
    try:
        return MemoryManager()
    except Exception as e:
        logger.error(f"Error initializing MemoryManager: {e}")
        st.error(f"Failed to initialize Memory Database: {e}")
        raise


@st.cache_resource(show_spinner="Initializing Heathcliff...")
def get_agent(_memory_manager: Optional[MemoryManager] = None) -> HeathcliffAgent:
    """
    Get or initialize the HeathcliffAgent singleton.
    Cached resource shared across all pages.

    Args:
        _memory_manager: Optional memory manager (for backward compatibility).
                         If not provided, agent will auto-initialize its own.
    """
    try:
        if _memory_manager is not None:
            return HeathcliffAgent(memory_manager=_memory_manager)
        return HeathcliffAgent.create()
    except Exception as e:
        logger.error(f"Error initializing HeathcliffAgent: {e}")
        st.error(f"Failed to initialize Agent: {e}")
        raise


def init_session_state():
    """Initialize common session state variables if they don't exist."""

    # Session ID for the conversation
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        logger.info(f"Created new session: {st.session_state.session_id}")

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Session start time
    if "session_start_time" not in st.session_state:
        st.session_state.session_start_time = datetime.now()

    # Greeting flag
    if "greeting_shown" not in st.session_state:
        st.session_state.greeting_shown = False

    # Approval state
    if "pending_approval" not in st.session_state:
        st.session_state.pending_approval = None


def clear_chat_session():
    """Reset the current chat session and generate a new ID."""
    st.session_state.messages = []
    st.session_state.greeting_shown = False
    st.session_state.session_start_time = datetime.now()
    st.session_state.session_id = str(uuid.uuid4())
    logger.info(f"Session reset. New ID: {st.session_state.session_id}")
