# ChromaDB Multi-Collection Setup

## Structure

```python
# core/memory_manager.py
import chromadb
from chromadb.config import Settings

class MemoryManager:
    def __init__(self, persist_dir="./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_dir)

        # Collections
        self.memories = self.client.get_or_create_collection("memories")
        self.chats = self.client.get_or_create_collection("chat_messages")
        self.my_data = self.client.get_or_create_collection("my_data")
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
    def __init__(self):
        self.memory = MemoryManager()
        self.session_id = str(uuid.uuid4())

    def process(self, user_input):
        # Get relevant context
        context = self.memory.get_context(user_input, n=3)
        memories = self.memory.recall(user_input, n=2)

        # Build prompt with context
        prompt = f"""
        Relevant memories: {memories['documents']}
        Recent context: {context['documents']}
        User: {user_input}
        """

        # Get response
        response = self.llm.invoke(prompt)

        # Save chat
        self.memory.save_chat(user_input, response, self.session_id)

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

- **Memories**: Long-term facts (name, preferences, important info)
- **Chats**: Short-term conversation context (last N messages)
- **My_data**: Indexed documents (emails, files from GDrive)

Embed once, query fast. Use metadata filters for precision.
