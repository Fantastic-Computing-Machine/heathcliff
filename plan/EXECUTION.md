# Heathcliff Implementation Plan

## Task 1: Project Setup & Config Manager

**Assign to**: Assistant A

**Deliverables**

```
heathcliff/
├── .env.example
├── config/config.py
├── config/
│   └── __init__.py
└── requirements.txt
```

**Implementation**

```python
# config/__init__.py
from config.config import Conf

Config = Conf()
```

**Docs**

- dotenv: <https://pypi.org/project/python-dotenv/>
- PyYAML: <https://pyyaml.org/wiki/PyYAMLDocumentation>

**Requirements**

- Create `.env.example` with all key names
- Create `config/config.py` with wake_word, news sources
- Expose singleton `Config` in `config/__init__.py`
- Generate `requirements.txt`

---

## Task 2: Memory Manager (ChromaDB)

**Assign to**: Assistant B

**Deliverable**: `core/memory_manager.py`

**Implementation**

```python
import chromadb
import uuid
from datetime import datetime

class MemoryManager:
    def __init__(self, persist_dir="./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.memories = self.client.get_or_create_collection("memories")
        self.chats = self.client.get_or_create_collection("chat_messages")
        self.my_data = self.client.get_or_create_collection("my_data")

    def add_memory(self, text, category="general"):
        """Store long-term fact"""
        pass

    def recall(self, query, n=3):
        """Search memories"""
        pass

    def save_chat(self, user_msg, asst_msg, session_id):
        """Store conversation turn"""
        pass

    def get_chat_context(self, query, n=5):
        """Retrieve relevant chat history"""
        pass

    def index_document(self, content, source, doc_type):
        """Store document in my_data"""
        pass
```

**Docs**

- ChromaDB: <https://docs.trychroma.com/getting-started>

**Requirements**

- 3 collections: memories, chat_messages, my_data
- Metadata: timestamp, type, category, session_id
- Query methods with n_results parameter
- Auto-generate UUIDs for IDs

---

## Task 3: Audio Handler (STT/TTS/Wake Word)

**Assign to**: Assistant C

**Deliverable**: `core/audio_handler.py`

**Implementation**

```python
import speech_recognition as sr
import pyttsx3
import pvporcupine

class AudioHandler:
    def __init__(self, wake_word="heathcliff"):
        self.recognizer = sr.Recognizer()
        self.tts_engine = pyttsx3.init()
        self.porcupine = pvporcupine.create(keywords=[wake_word])

    def listen_for_wake_word(self):
        """Return True when wake word detected"""
        pass

    def speech_to_text(self):
        """Capture audio, return text"""
        pass

    def text_to_speech(self, text):
        """Speak text"""
        pass

    def listen_loop(self, callback):
        """Main loop: wake word → STT → callback → TTS"""
        pass
```

**Docs**

- SpeechRecognition: <https://pypi.org/project/SpeechRecognition/>
- pyttsx3: <https://pyttsx3.readthedocs.io/>
- Porcupine: <https://picovoice.ai/docs/porcupine/>

**Requirements**

- Wake word detection with Porcupine
- Google Speech Recognition for STT
- pyttsx3 for TTS
- `listen_loop()` orchestrates: wake → listen → process → speak

---

## Task 4: Gemini Agent Core

**Assign to**: Assistant D

**Deliverable**: `core/agent_core.py`

**Implementation**

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from typing import TypedDict

class AgentState(TypedDict):
    messages: list
    context: str
    session_id: str

class HeathcliffAgent:
    def __init__(self, config, memory_manager):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=config.GEMINI_API_KEY
        )
        self.memory = memory_manager
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        # Add nodes: retrieval, reasoning, tool_calling
        # Add edges
        return workflow.compile()

    def invoke(self, user_input, session_id):
        """Process input, return response"""
        pass
```

**Docs**

- LangChain Google GenAI: <https://python.langchain.com/docs/integrations/chat/google_generative_ai>
- LangGraph: <https://langchain-ai.github.io/langgraph/>

**Requirements**

- Use Gemini Flash 2.5 via LangChain
- LangGraph StateGraph with nodes
- Integrate MemoryManager for context retrieval
- Return response + save to memory

---

## Task 5: Tools - Email

**Assign to**: Assistant E

**Deliverable**: `tools/email_tool.py`

**Implementation**

```python
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from langchain.tools import tool

@tool
def read_emails(query: str, max_results: int = 5):
    """Read recent emails matching query"""
    pass

@tool
def send_email(to: str, subject: str, body: str):
    """Send email via Gmail API"""
    pass
```

**Docs**

- Gmail API Python: <https://developers.google.com/gmail/api/quickstart/python>

**Requirements**

- OAuth2 with credentials.json
- Read emails (list + get)
- Send email (messages.send)
- Return as LangChain tools

---

## Task 6: Tools - Calendar

**Assign to**: Assistant F

**Deliverable**: `tools/calendar_tool.py`

**Implementation**

```python
from langchain.tools import tool

@tool
def read_my_calendar(days_ahead: int = 7):
    """Read user's calendar (read-only)"""
    pass

@tool
def add_heathcliff_event(summary: str, start_time: str, duration_mins: int):
    """Add event to Heathcliff's calendar"""
    pass

@tool
def list_heathcliff_events():
    """List Heathcliff's calendar events"""
    pass
```

**Docs**

- Calendar API: <https://developers.google.com/calendar/api/quickstart/python>

**Requirements**

- 2 calendars: user (read), heathcliff (CRUD)
- OAuth2 same as Gmail
- Return as LangChain tools

---

## Task 7: Tools - Spotify

**Assign to**: Assistant G

**Deliverable**: `tools/spotify_tool.py`

**Implementation**

```python
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from langchain.tools import tool

@tool
def play_track(query: str):
    """Search and play track"""
    pass

@tool
def pause_playback():
    """Pause current playback"""
    pass

@tool
def current_track():
    """Get currently playing track"""
    pass
```

**Docs**

- Spotipy: <https://spotipy.readthedocs.io/>

**Requirements**

- SpotifyOAuth with client_id/secret
- Player control: play, pause, skip, volume
- Return as LangChain tools

---

## Task 8: Tools - News/Weather/Web

**Assign to**: Assistant H

**Deliverable**: `tools/info_tools.py`

**Implementation**

```python
from langchain.tools import tool
import requests

@tool
def get_weather(city: str):
    """Get current weather"""
    pass

@tool
def get_news(category: str = "technology"):
    """Get latest news"""
    pass

@tool
def search_web(query: str):
    """Search web via SerpAPI/Google"""
    pass

@tool
def wikipedia_search(query: str):
    """Search Wikipedia"""
    pass
```

**Docs**

- OpenWeatherMap: <https://openweathermap.org/api>
- NewsAPI: <https://newsapi.org/docs>
- Wikipedia: <https://pypi.org/project/wikipedia/>

**Requirements**

- Weather from OpenWeatherMap
- News from NewsAPI (filter by sources in config)
- Web search (use requests or SerpAPI)
- Return as LangChain tools

---

## Task 9: Tools - Telegram/GDrive

**Assign to**: Assistant I

**Deliverable**: `tools/comm_tools.py`

**Implementation**

```python
from telegram import Bot
from langchain.tools import tool

@tool
def send_to_telegram(message: str):
    """Send message to Telegram"""
    pass

@tool
def read_gdrive_file(file_id: str):
    """Read file from Google Drive"""
    pass
```

**Docs**

- python-telegram-bot: <https://docs.python-telegram-bot.org/>
- Drive API: <https://developers.google.com/drive/api/quickstart/python>

**Requirements**

- Telegram Bot API (get chat_id from config)
- GDrive OAuth2 (same as Gmail)
- Return as LangChain tools

---

## Task 10: Streamlit UI

**Assign to**: Assistant J

**Deliverable**: `ui/streamlit_app.py`

**Implementation**

```python
import streamlit as st
from core.memory_manager import MemoryManager

st.title("Heathcliff Dashboard")

tab1, tab2, tab3 = st.tabs(["Chat History", "Memories", "Settings"])

with tab1:
    # Display chat_messages collection
    pass

with tab2:
    # Display memories collection
    # Add/delete memory form
    pass

with tab3:
    # Config editor (news sources, wake word)
    pass
```

**Docs**

- Streamlit: <https://docs.streamlit.io/>

**Requirements**

- 3 tabs: Chat History, Memories, Settings
- Read from ChromaDB collections
- Display with timestamps
- Settings: edit config/config.py values

---

## Task 11: Main Orchestrator

**Assign to**: Assistant K

**Deliverable**: `main.py`

**Implementation**

```python
from core.audio_handler import AudioHandler
from core.agent_core import HeathcliffAgent
from core.memory_manager import MemoryManager
from config import Config

def main():
    config = Config
    memory = MemoryManager(config=config)
    agent = HeathcliffAgent(config, memory)
    audio = AudioHandler(config.yaml_config['wake_word'])

    def process_input(text):
        response = agent.invoke(text, session_id)
        return response

    audio.listen_loop(process_input)

if __name__ == "__main__":
    main()
```

**Requirements**

- Initialize all components
- Connect audio → agent → memory
- Handle graceful shutdown
- Session management (new UUID per wake)

---

## Dependency Graph

```
Task 1 (Config) → All tasks
Task 2 (Memory) → Task 4, 10, 11
Task 3 (Audio) → Task 11
Task 4 (Agent) → Task 11
Tasks 5-9 (Tools) → Task 4
Task 10 (UI) → Independent
Task 11 (Main) → All
```

## Execution Order

**Parallel** (can run simultaneously):

- Tasks 1, 2, 3
- Tasks 5, 6, 7, 8, 9 (after Task 1)

**Sequential**:

1. Task 1
2. Tasks 2, 3, 5-9 (parallel)
3. Task 4 (needs 2, 5-9)
4. Task 10 (needs 2)
5. Task 11 (needs all)

**Total**: 11 discrete tasks, 5-6 can run in parallel.
