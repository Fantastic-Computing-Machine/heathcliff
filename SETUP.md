# Heathcliff Setup Guide

Complete setup instructions for the Heathcliff voice-activated AI assistant.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [System Dependencies](#system-dependencies)
3. [Python Environment](#python-environment)
4. [API Credentials Setup](#api-credentials-setup)
5. [Configuration](#configuration)
6. [Running Heathcliff](#running-heathcliff)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

- **Python**: 3.10 or higher
- **Operating System**: Linux, macOS, or Windows with WSL
- **Microphone**: For voice input
- **Speakers**: For audio output

---

## System Dependencies

### Linux (Ubuntu/Debian)

```bash
# Audio support for PyAudio
sudo apt update
sudo apt install python3-pyaudio portaudio19-dev

# Optional: For better audio quality
sudo apt install espeak ffmpeg
```

### macOS

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install port audio
brew install portaudio
```

### Windows (WSL)

Follow Linux instructions above in your WSL environment.

---

## Python Environment

### 1. Create Virtual Environment

```bash
cd heathcliff
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## API Credentials Setup

### 1. Google Cloud Platform (Gmail, Calendar, Drive)

#### Enable APIs

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable the following APIs:
   - Gmail API
   - Google Calendar API
   - Google Drive API (optional)

#### Create OAuth 2.0 Credentials

1. Navigate to **APIs & Services** > **Credentials**
2. Click **Create Credentials** > **OAuth 2.0 Client ID**
3. Choose **Desktop app** as application type
4. Download the credentials as `credentials.json`
5. Place `credentials.json` in the project root directory

**Required Scopes:**

```
https://www.googleapis.com/auth/gmail.readonly
https://www.googleapis.com/auth/gmail.send
https://www.googleapis.com/auth/gmail.compose
https://www.googleapis.com/auth/calendar.readonly
https://www.googleapis.com/auth/calendar.events
https://www.googleapis.com/auth/drive.readonly
```

### 2. Google AI Studio (Gemini API)

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Click **Get API Key**
3. Create a new API key
4. Copy the key (starts with `AI...`)

### 3. OpenWeatherMap

1. Sign up at [OpenWeatherMap](https://openweathermap.org/api)
2. Get your free API key from the account dashboard
3. Free tier: 60 calls/minute, 1,000,000 calls/month

### 4. NewsAPI

1. Sign up at [NewsAPI](https://newsapi.org/)
2. Get your free API key
3. Free tier: 100 requests/day

### 5. Spotify

1. Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
2. Create an app
3. Note your **Client ID** and **Client Secret**
4. Add `http://localhost:8888/callback` to **Redirect URIs**

### 6. Telegram (Optional)

1. Open Telegram and search for [@BotFather](https://t.me/botfather)
2. Send `/newbot` and follow instructions
3. Save the bot token
4. Get your chat ID:
   - Send a message to your bot
   - Visit: `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates`
   - Find your `chat_id` in the response

### 7. Langfuse (Optional but recommended)

1. Create a project at [Langfuse](https://cloud.langfuse.com/) or use your self-hosted deployment.
2. Generate a **Public key** and **Secret key** via **Project Settings → API Keys**.
3. Copy the base URL for your region (EU: `https://cloud.langfuse.com`, US: `https://us.cloud.langfuse.com`, custom if self-hosting).
4. Set `LANGFUSE_BASE_URL`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, optional `LANGFUSE_HOST`/`LANGFUSE_RELEASE` inside `.env`.
5. Run Heathcliff and inspect real-time traces, prompts, and tool usage inside Langfuse. Use `python -m utils.langfuse_client` to verify credentials if nothing appears. Traces default to `user_id=adiagarwal`; update `observability.langfuse.user_id` in `config/config.py` if you want a different alias.
6. The Langfuse LangChain callback reads credentials from environment variables—do not pass them directly into the handler, otherwise new SDK versions throw `unexpected keyword argument 'secret_key'`.

---

## Configuration

### 1. Environment Variables

Copy the example file:

```bash
cp .env.example .env
```

Edit `.env` and add your credentials:

```bash
# Gemini API
GEMINI_API_KEY=AIza...your_key_here

# Google Services (path to credentials.json)
GOOGLE_APPLICATION_CREDENTIALS=./credentials.json

# Spotify
SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_client_secret

# Telegram (Optional)
TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
TELEGRAM_CHAT_ID=your_chat_id

# Weather
OPENWEATHERMAP_API_KEY=your_key_here

# News
NEWSAPI_KEY=your_key_here

# Langfuse Observability (optional)
LANGFUSE_BASE_URL=https://cloud.langfuse.com
LANGFUSE_PUBLIC_KEY=pk_live_...
LANGFUSE_SECRET_KEY=sk_live_...
LANGFUSE_RELEASE=local-dev
```

### 2. Configuration File

Edit `config/config.py` to customize settings:

```yaml
# Wake word (must be supported by OpenWakeWord)
wake_word: "hey_jarvis"  # or "alexa", "hey_mycroft", etc.

# TTS settings
tts:
  rate: 175        # Words per minute
  volume: 0.9      # 0.0 to 1.0

# News preferences
news:
  sources:
    - bbc-news
    - techcrunch
  topics:
    - technology
    - artificial-intelligence
  max_articles: 5

# Weather
weather:
  default_city: "London"
  units: "metric"  # or "imperial"

# Observability
observability:
  langfuse:
    enabled: true
    base_url: null
    environment: "local-dev"
    trace_name: "heathcliff.agent"

# LLM settings
llm:
  model: "gemini-2.0-flash-exp"
  temperature: 0.7
  max_tokens: 1024
```

---

## Running Heathcliff

### First-Time Setup

On first run, you'll be prompted to authorize Google APIs:

```bash
python main.py
```

A browser window will open for OAuth consent. Grant the requested permissions.

### Voice Mode (Default)

```bash
python main.py
```

- Say "heathcliff" to activate
- Speak your command
- Heathcliff will respond via audio

### Text Mode (Testing)

```bash
python main.py --text
```

- Type commands in the terminal
- Responses are printed
- No audio hardware required

### Streamlit Dashboard

```bash
streamlit run ui/streamlit_app.py
```

Access at `http://localhost:8501`

---

## Testing Your Setup

### 1. Test Configuration

```python
from config import Config

Config.validate()
```

### 2. Test Memory

```python
from core.memory_manager import MemoryManager
from config import Config

memory = MemoryManager(config=Config)
memory_id = memory.add_memory("Test memory", category="test")
results = memory.recall("test")
print(results)
```

### 3. Test Agent (Text Mode)

```bash
python main.py --text
```

Try:

- "What's the weather?"
- "Search for artificial intelligence"
- "Tell me the time"

### 4. Test Tools Individually

```python
from tools import get_weather, get_news

# Test weather
print(get_weather.invoke("London"))

# Test news
print(get_news.invoke("technology"))
```

---

## Troubleshooting

### PyAudio Installation Issues

**Linux:**

```bash
sudo apt install python3-dev portaudio19-dev
pip install --force-reinstall pyaudio
```

**macOS:**

```bash
brew install portaudio
pip install --global-option='build_ext' --global-option='-I/opt/homebrew/include' --global-option='-L/opt/homebrew/lib' pyaudio
```

### OpenWakeWord Wake Word Issues

- Pre-trained models include: "hey_jarvis", "alexa", "hey_mycroft", and more
- Threshold can be adjusted in `core/audio_handler.py` (default: 0.5)
- For custom wake words, you can train your own models using the [OpenWakeWord training guide](https://github.com/dscripka/openWakeWord#training-new-models)
- Models are automatically downloaded on first use

### Google OAuth Issues

1. Ensure `credentials.json` is in project root
2. Delete existing token files (`*_token.pickle`) and re-authenticate
3. Check that all required APIs are enabled in Google Cloud Console

### Gemini API Errors

- Verify API key is correct
- Check quota limits in [Google AI Studio](https://makersuite.google.com/)
- Ensure you're using the correct model name: `gemini-2.0-flash-exp`

### ChromaDB Issues

```bash
# Clear database and restart
rm -rf ./chroma_db
python main.py
```

### Audio Device Not Found

```python
# List available audio devices
python -m sounddevice
```

Adjust device index in `core/audio_handler.py` if needed.

---

## Next Steps

1. **Add Memories**: Use the Streamlit dashboard to add facts about yourself
2. **Test Tools**: Try different commands to test each integration
3. **Customize**: Modify `config/config.py` for your preferences
4. **Extend**: Add custom tools in `tools/` directory

---

## Getting Help

- Check logs in console output
- Review `logger.py` for debug settings
- See [CLAUDE.md](.claude/CLAUDE.md) for development info
- Check [plan/TODO.md](plan/TODO.md) for known issues

---

## Security Notes

- Never commit `.env` or `credentials.json` to version control
- Keep API keys secure
- OAuth tokens are stored in `*_token.pickle` files (also git-ignored)
- Rotate API keys regularly

---

## Performance Tips

1. **First Run**: Initial model downloads may take time
2. **Memory**: ChromaDB builds embeddings on first use
3. **Audio**: Adjust chunk size in `config/config.py` for latency/quality trade-off
4. **Rate Limits**: Be mindful of API quotas in production use

Enjoy using Heathcliff! 🎤
