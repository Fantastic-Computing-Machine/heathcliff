# Persistence setup

Heathcliff uses ChromaDB for conversation history and as the vector-store
backend for Mem0 long-term memories. Configure the Chroma and Mem0 values in
`.env`, then create `MemoryManager()`; it reads the shared `Config`.

```python
from db.memory_manager import MemoryManager

memory = MemoryManager()
memory.add_memory("User likes jazz", category="preference")
memory.save_turn("Play music", "What would you like to hear?", "conversation-1")
results = memory.recall("music preferences")
```

Conversation records and long-term memories are separate. There is no general
document-index collection until an actual document writer and search tool need
one.
