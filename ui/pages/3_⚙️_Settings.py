# ABOUTME: Streamlit multipage app - Settings and configuration page
# ABOUTME: View system configuration and API status

import os
import sys

import streamlit as st

# Add parent directory to path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from config import Config

st.set_page_config(page_title="Settings", page_icon="⚙️", layout="wide")


# Header
st.title("⚙️ Configuration & Settings")
st.markdown("View and verify your Heathcliff configuration")

# Tabs for different settings categories
tab1, tab2, tab3, tab4 = st.tabs(
    ["🔑 API Status", "🎤 Audio Settings", "🤖 LLM Settings", "📁 Database"]
)

# Tab 1: API Status
with tab1:
    st.subheader("API Connection Status")

    # Check API keys
    api_checks = {
        "Gemini API": {
            "configured": bool(Config.AI_KEY),
            "key_preview": (
                f"{Config.AI_KEY[:10]}..." if Config.AI_KEY else "Not configured"
            ),
            "docs": "https://makersuite.google.com/app/apikey",
        },
        "Google OAuth (Gmail/Calendar/Drive)": {
            "configured": bool(Config.GOOGLE_APPLICATION_CREDENTIALS),
            "key_preview": (
                Config.GOOGLE_APPLICATION_CREDENTIALS
                if Config.GOOGLE_APPLICATION_CREDENTIALS
                else "Not configured"
            ),
            "docs": "https://console.cloud.google.com/",
        },
        "OpenWeatherMap": {
            "configured": bool(Config.OPENWEATHERMAP_API_KEY),
            "key_preview": (
                f"{Config.OPENWEATHERMAP_API_KEY[:10]}..."
                if Config.OPENWEATHERMAP_API_KEY
                else "Not configured"
            ),
            "docs": "https://openweathermap.org/api",
        },
        "NewsAPI": {
            "configured": bool(Config.NEWS_API_KEY),
            "key_preview": (
                f"{Config.NEWS_API_KEY[:10]}..."
                if Config.NEWS_API_KEY
                else "Not configured"
            ),
            "docs": "https://newsapi.org/",
        },
        "Spotify Client": {
            "configured": bool(
                Config.SPOTIFY_CLIENT_ID and Config.SPOTIFY_CLIENT_SECRET
            ),
            "key_preview": (
                f"{Config.SPOTIFY_CLIENT_ID[:10]}..."
                if Config.SPOTIFY_CLIENT_ID
                else "Not configured"
            ),
            "docs": "https://developer.spotify.com/dashboard",
        },
        "Telegram Bot": {
            "configured": bool(Config.TELEGRAM_BOT_TOKEN),
            "key_preview": (
                f"{Config.TELEGRAM_BOT_TOKEN[:10]}..."
                if Config.TELEGRAM_BOT_TOKEN
                else "Not configured"
            ),
            "docs": "https://t.me/botfather",
        },
    }

    for api_name, info in api_checks.items():
        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            status_icon = "✅" if info["configured"] else "❌"
            st.write(f"{status_icon} **{api_name}**")

        with col2:
            if info["configured"]:
                st.code(info["key_preview"], language=None)
            else:
                st.warning("Not configured")

        with col3:
            st.markdown(f"[Docs]({info['docs']})")

    st.markdown("---")
    st.info("💡 **Tip**: Edit `.env` file in the project root to update API keys")

# Tab 2: Audio Settings
with tab2:
    st.subheader("Voice & Audio Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Wake Word**")
        wake_word = Config.WAKE_WORD
        st.code(wake_word, language=None)
        st.caption("The word used to activate voice mode")

        st.markdown("**Audio Processing**")
        st.code(
            f"""
Sample Rate: {Config.SAMPLE_RATE} Hz
Chunk Size: {Config.CHUNK_SIZE} samples
        """,
            language=None,
        )

    with col2:
        st.markdown("**Text-to-Speech**")
        st.code(
            f"""
Rate: {Config.TTS_RATE} words/minute
Volume: {Config.TTS_VOLUME}
Voice: {Config.TTS_VOICE or "default"}
        """,
            language=None,
        )

    st.markdown("---")
    st.info("💡 **Tip**: Edit `config/Config.py` to change audio settings")

# Tab 3: LLM Settings
with tab3:
    st.subheader("Language Model Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**SUPERVISOR_MODEL**")
        model = Config.SUPERVISOR_MODEL
        st.code(model, language=None)

        st.markdown("**TOOL_MODEL**")
        model = Config.TOOL_MODEL
        st.code(model, language=None)

        st.markdown("**Parameters**")
        st.code(
            f"""
Temperature: {Config.TEMPERATURE}
Max Tokens: {Config.MAX_TOKENS}
        """,
            language=None,
        )

    with col2:
        st.markdown("**Memory Settings**")
        st.code(
            f"""
Chat Context: {Config.MEMORY_CHAT_CONTEXT} messages
Long-term Memories: {Config.MEMORY_MAX_MEMORIES} items
        """,
            language=None,
        )

        st.markdown("**Session**")
        st.code(
            f"""
Timeout: {Config.TIMEOUT_SECONDS} seconds
        """,
            language=None,
        )

    st.markdown("---")
    st.info("💡 **Tip**: Higher temperature = more creative, Lower = more focused")

# Tab 4: Database
with tab4:
    st.subheader("ChromaDB Configuration")

    persist_dir = Config.CHROMA_PERSIST_DIRECTORY

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Storage Location**")
        st.code(persist_dir, language="bash")

        st.markdown("**Collections**")
        st.code(
            """
- memories (long-term facts)
- chat_messages (conversations)
        """,
            language=None,
        )

    with col2:
        st.markdown("**Status**")

        import os as os_check

        if os_check.path.exists(persist_dir):
            st.success("✅ Database directory exists")

            # Get size
            try:
                total_size = sum(
                    os_check.path.getsize(os_check.path.join(dirpath, filename))
                    for dirpath, dirnames, filenames in os_check.walk(persist_dir)
                    for filename in filenames
                )
                size_mb = total_size / (1024 * 1024)
                st.metric("Database Size", f"{size_mb:.2f} MB")
            except Exception as e:
                st.error(f"Error calculating size: {e}")
        else:
            st.warning("⚠️ Database directory not found (will be created on first use)")

    st.markdown("---")

    # Database actions
    st.subheader("⚠️ Danger Zone")

    with st.expander("🗑️ Clear All Data"):
        st.warning("This will delete all conversations and memories!")
        st.warning("This action CANNOT be undone!")

        if st.button("I understand, clear everything"):
            st.error("❌ Clear functionality disabled in UI for safety")
            st.info(
                "To clear data, delete the ChromaDB directory manually:\n\n`rm -rf ./chroma_db`"
            )

# Footer
st.markdown("---")
st.caption("⚙️ All settings are loaded from `.env` and `config/Config.py` files")
