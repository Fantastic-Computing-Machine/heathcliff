# ABOUTME: Streamlit multipage app - Settings and configuration page
# ABOUTME: View system configuration and API status

import streamlit as st
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import get_config


st.set_page_config(page_title="Settings", page_icon="⚙️", layout="wide")

# Initialize config
config = get_config()

# Header
st.title("⚙️ Configuration & Settings")
st.markdown("View and verify your Heathcliff configuration")

# Tabs for different settings categories
tab1, tab2, tab3, tab4 = st.tabs(["🔑 API Status", "🎤 Audio Settings", "🤖 LLM Settings", "📁 Database"])

# Tab 1: API Status
with tab1:
    st.subheader("API Connection Status")

    # Check API keys
    api_checks = {
        "Gemini API": {
            "configured": bool(config.gemini_key),
            "key_preview": f"{config.gemini_key[:10]}..." if config.gemini_key else "Not configured",
            "docs": "https://makersuite.google.com/app/apikey"
        },
        "Google OAuth (Gmail/Calendar/Drive)": {
            "configured": bool(config.google_credentials),
            "key_preview": config.google_credentials if config.google_credentials else "Not configured",
            "docs": "https://console.cloud.google.com/"
        },
        "OpenWeatherMap": {
            "configured": bool(config.openweathermap_key),
            "key_preview": f"{config.openweathermap_key[:10]}..." if config.openweathermap_key else "Not configured",
            "docs": "https://openweathermap.org/api"
        },
        "NewsAPI": {
            "configured": bool(config.newsapi_key),
            "key_preview": f"{config.newsapi_key[:10]}..." if config.newsapi_key else "Not configured",
            "docs": "https://newsapi.org/"
        },
        "Spotify Client": {
            "configured": bool(config.spotify_client_id and config.spotify_client_secret),
            "key_preview": f"{config.spotify_client_id[:10]}..." if config.spotify_client_id else "Not configured",
            "docs": "https://developer.spotify.com/dashboard"
        },
        "Telegram Bot": {
            "configured": bool(config.telegram_token),
            "key_preview": f"{config.telegram_token[:10]}..." if config.telegram_token else "Not configured",
            "docs": "https://t.me/botfather"
        },
    }

    for api_name, info in api_checks.items():
        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            status_icon = "✅" if info['configured'] else "❌"
            st.write(f"{status_icon} **{api_name}**")

        with col2:
            if info['configured']:
                st.code(info['key_preview'], language=None)
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
        wake_word = config.get('wake_word', 'heathcliff')
        st.code(wake_word, language=None)
        st.caption("The word used to activate voice mode")

        st.markdown("**Audio Processing**")
        st.code(f"""
Sample Rate: {config.get('audio.sample_rate', 16000)} Hz
Chunk Size: {config.get('audio.chunk_size', 512)} samples
        """, language=None)

    with col2:
        st.markdown("**Text-to-Speech**")
        st.code(f"""
Rate: {config.get('tts.rate', 175)} words/minute
Volume: {config.get('tts.volume', 0.9)}
Voice: {config.get('tts.voice', 'default')}
        """, language=None)

    st.markdown("---")
    st.info("💡 **Tip**: Edit `config.yaml` to change audio settings")

# Tab 3: LLM Settings
with tab3:
    st.subheader("Language Model Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Model**")
        model = config.get('llm.model', 'gemini-2.0-flash-exp')
        st.code(model, language=None)

        st.markdown("**Parameters**")
        st.code(f"""
Temperature: {config.get('llm.temperature', 0.7)}
Max Tokens: {config.get('llm.max_tokens', 1024)}
        """, language=None)

    with col2:
        st.markdown("**Memory Settings**")
        st.code(f"""
Chat Context: {config.get('memory.max_chat_context', 10)} messages
Long-term Memories: {config.get('memory.max_memories', 5)} items
        """, language=None)

        st.markdown("**Session**")
        st.code(f"""
Timeout: {config.get('session.timeout_seconds', 300)} seconds
        """, language=None)

    st.markdown("---")
    st.info("💡 **Tip**: Higher temperature = more creative, Lower = more focused")

# Tab 4: Database
with tab4:
    st.subheader("ChromaDB Configuration")

    persist_dir = config.get('chroma.persist_directory', './chroma_db')

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Storage Location**")
        st.code(persist_dir, language="bash")

        st.markdown("**Collections**")
        st.code("""
- memories (long-term facts)
- chat_messages (conversations)
- my_data (documents)
        """, language=None)

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
        st.warning("This will delete ALL conversations, memories, and indexed documents!")
        st.warning("This action CANNOT be undone!")

        if st.button("I understand, clear everything"):
            st.error("❌ Clear functionality disabled in UI for safety")
            st.info("To clear data, delete the ChromaDB directory manually:\n\n`rm -rf ./chroma_db`")

# Footer
st.markdown("---")
st.caption("⚙️ All settings are loaded from `.env` and `config.yaml` files")
