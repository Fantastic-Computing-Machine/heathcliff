# ChromaDB Multi-Collection Setup

> **Note**: Heathcliff uses **Mem0** for long-term memory (add/search) and **ChromaDB** as the backend for Mem0, chat history, and document indexing. The `MemoryManager` constructor takes no arguments — it reads all configuration from the `Config` singleton. See `config/config.py` for Mem0 and ChromaDB configuration.

## Structure

```python
# core/memory_manager.py (simplified)
import chromadb

class MemoryManager:
    def __init__(self):  # No args — reads from Config singleton
        # ChromaDB client (local or cloud, depending on Config)
        client = chroma_client(...)  # Uses Config.CHROMA_* settings

        # Collections
        self.chats = client.get_or_create_collection("chat_messages")
        self.my_data = client.get_or_create_collection("my_data")

        # Mem0 handles long-term memories (separate from ChromaDB collections above)
        self._mem0 = self._get_mem0_client(client)
```

## Usage Patterns

**Memories** (facts to remember)

```python
def add_memory(self, text, metadata={}):
    self.memories.add(
        documents=[text],
        metadatas=[{"type": "fact", "timestamp": time.time(), **metadata}],
        ids=[f"mem_{uuid.uuid4()}"]
    )

def recall(self, query, n=3):
    return self.memories.query(query_texts=[query], n_results=n)
```

**Chat Messages** (conversation history)

```python
def save_chat(self, user_msg, assistant_msg, session_id):
    id_base = f"{session_id}_{int(time.time())}"
    self.chats.add(
        documents=[user_msg, assistant_msg],
        metadatas=[
            {"role": "user", "session": session_id},
            {"role": "assistant", "session": session_id}
        ],
        ids=[f"{id_base}_user", f"{id_base}_asst"]
    )

def get_context(self, query, n=5):
    return self.chats.query(query_texts=[query], n_results=n)
```

**My Data** (documents/emails/files)

```python
def index_document(self, content, source, doc_type):
    self.my_data.add(
        documents=[content],
        metadatas=[{"source": source, "type": doc_type}],
        ids=[f"{doc_type}_{source}_{uuid.uuid4()}"]
    )

def search_my_data(self, query, doc_type=None, n=3):
    filters = {"type": doc_type} if doc_type else None
    return self.my_data.query(query_texts=[query], n_results=n, where=filters)
```

## Agent Integration

```python
# core/agent_core.py
from core.memory_manager import MemoryManager

class HeathcliffAgent:
    def __init__(self, memory_manager=None):
        self.memory = memory_manager or MemoryManager()
        self.session_id = str(uuid.uuid4())

    def invoke(self, user_input, session_id=None):
        # 1. Get pair-based chat context (semantic + chronological)
        message_history = self.memory.build_message_history(
            query=user_input, session_id=session_id
        )

        # 2. Get Mem0 long-term memories
        memories = self.memory.recall(user_input)

        # 3. Build memories block for USER_PROMPT_TEMPLATE
        memories_block = self._build_memories_block(memories)

        # 4. Format user prompt with XML delimiters
        #    <USER_MEMORY_CONTEXT>...memories...</USER_MEMORY_CONTEXT>
        #    <USER_QUERY>...user_input...</USER_QUERY>
        user_prompt = USER_PROMPT_TEMPLATE.format(
            memories_block=memories_block,
            temporal_context=get_current_temporal_context(),
            user_query=user_input,
        )

        # 5. Invoke LangGraph agent with message history + user prompt
        result = self.agent.invoke({"messages": history + [HumanMessage(user_prompt)]})
        response = result["messages"][-1].content

        # 6. Save to chat history
        self.memory.save_chat(user_input, response, session_id)

        return response
```

## Query Examples

```python
# Remember something
memory.add_memory("User likes jazz music", {"category": "preference"})

# Search memories
results = memory.recall("what music does user like?")

# Filter by metadata
memory.chats.query(
    query_texts=["email discussion"],
    where={"session": session_id}
)

# Search specific doc types
memory.search_my_data("budget report", doc_type="email")
```

## Key Points

- **Memories (via Mem0)**: Long-term facts (name, preferences, important info) — managed by Mem0 SDK
- **Chats (ChromaDB)**: Short-term conversation context (pair-based history)
- **My_data (ChromaDB)**: Indexed documents (emails, files from GDrive)

Embed once, query fast. Use metadata filters for precision.
