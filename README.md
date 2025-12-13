# heathcliff
Assistant that just does stuff -- no BS.

## What is Heathcliff?

Heathcliff is a voice-enabled personal AI assistant that integrates with your daily services. Wake it up with "Heathcliff", give it commands, and watch it orchestrate tasks across Gmail, Calendar, Spotify, and more using LLM-powered decision making.

## Key Features

- Voice activation with wake word detection
- Gmail integration (read, search, send, draft)
- Google Calendar management
- Spotify playback control
- LLM-powered task orchestration (Gemini 2.0 Flash)
- Conversational context awareness with ChromaDB memory
- LangGraph-based agent orchestration

## Tech Stack

- **LLM Framework**: LangChain + LangGraph with Gemini 2.0 Flash
- **Memory**: ChromaDB for persistent vector storage
- **Voice**: pvporcupine (wake word), Google STT
- **Integrations**: Gmail, Google Calendar, Spotify APIs
- **Audio**: PyAudio

## Architecture

Heathcliff uses a LangGraph-based agent architecture with 4 nodes:

1. **Retrieval Node**: Fetches relevant context and memories from ChromaDB
2. **Reasoning Node**: Processes input with Gemini LLM, determines actions
3. **Tool Calling Node**: Executes requested tools (weather, time, etc.)
4. **Output Node**: Saves conversation to memory, returns response

## Setup

### Prerequisites

Install system dependencies:
```bash
sudo apt install python3-pyaudio portaudio19-dev
```

### Installation

```bash
pip install -r requirements.txt
```

### Configuration

1. Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

2. Add your Gemini API key to `.env`:
```
GEMINI_API_KEY=your-api-key-here
```

3. (Optional) Configure LLM settings in `config.yaml`:
```yaml
llm:
  model: "gemini-2.0-flash-exp"
  temperature: 0.7
  max_tokens: 1024
```

## Usage

### Using the Agent Core (Programmatic)

```python
from core import MemoryManager, HeathcliffAgent
from config.config_loader import get_config

# Initialize components
config = get_config()
memory = MemoryManager(persist_dir="./chroma_db")
agent = HeathcliffAgent(config=config, memory_manager=memory)

# Single turn conversation
response = agent.invoke("Hello! What can you do?")
print(response)

# Multi-turn conversation (same session maintains context)
session_id = "my-session-123"
response1 = agent.invoke("My name is Adi", session_id=session_id)
response2 = agent.invoke("What's my name?", session_id=session_id)
# response2 will know your name is Adi
```

### Using the Memory Manager

```python
from core import MemoryManager

memory = MemoryManager(persist_dir="./chroma_db")

# Store a long-term memory
memory_id = memory.add_memory("User prefers dark mode", category="preferences")

# Recall relevant memories
results = memory.recall("what are user preferences?", n=3)
print(results["documents"])

# Save chat conversation
memory.save_chat(
    user_msg="What's the weather?",
    assistant_msg="It's sunny and 72F",
    session_id="session-123"
)

# Retrieve chat context
context = memory.get_chat_context("weather", session_id="session-123")
```

### Voice Mode

```bash
python app.py
```

Say "Heathcliff" to activate, then give your command.

## Example Conversations

```
User: Hello!
Heathcliff: Hello! I'm Heathcliff, your personal AI assistant. How can I help you today?

User: What's the weather in London?
Heathcliff: The weather in London is 72F and sunny.

User: My name is Adi and I work as a software engineer.
Heathcliff: Nice to meet you, Adi! I'll remember that you're a software engineer.

User: What do you know about me?
Heathcliff: Based on what I know, your name is Adi and you work as a software engineer.
```

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test files
python -m pytest tests/test_memory_manager.py -v
python -m pytest tests/test_agent_core.py -v
python -m pytest tests/test_agent_integration.py -v
python -m pytest tests/test_agent_e2e.py -v
```

## Project Structure

```
heathcliff/
├── core/
│   ├── __init__.py
│   ├── memory_manager.py    # ChromaDB-backed memory storage
│   ├── agent_core.py        # LangGraph agent orchestrator
│   └── audio_handler.py     # Voice I/O handling
├── config/
│   └── config_loader.py     # Configuration management
├── tools/                   # Tool integrations
├── tests/                   # Test suite
│   ├── test_memory_manager.py
│   ├── test_agent_core.py
│   ├── test_agent_integration.py
│   └── test_agent_e2e.py
├── config.yaml              # Runtime configuration
├── requirements.txt         # Python dependencies
└── README.md
```

## Status

Development Phase 2 Complete (v0.2.0):
- Memory Manager with ChromaDB persistence
- LangGraph-based Agent Core with Gemini 2.0 Flash
- Multi-turn conversation support
- Tool calling framework
- 67 tests passing
