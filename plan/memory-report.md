# Memory / Qdrant wiring contract

Implementation in progress. No live calls, migrations, commits, or service restarts.

## Controller construction

```python
from core.runtime.gemini_memory import GeminiEmbedder, GeminiMemoryClassifier
from core.runtime.memory import MemoryJobWorker, SemanticMemoryPipeline
from core.runtime.recall import QdrantIndex, RecallService
from db.recall_store import PostgresRecallStore

repository = PostgresRecallStore(pool)  # borrow the connected runtime asyncpg pool
index = QdrantIndex(qdrant_url, api_key=qdrant_api_key)
embedder = GeminiEmbedder(gemini_api_key, model=configured_embedding_model)
recall = RecallService(repository, embedder, index)
pipeline = SemanticMemoryPipeline(
    runtime_store, GeminiMemoryClassifier(gemini_api_key, configured_model),
    embedder, recall=recall,
)
memory_worker = MemoryJobWorker(runtime_store, pipeline)
```

Controller applies ordered migration `db/migrations/003_runtime_recall.sql` with its
normal migration mechanism. Repository construction performs no I/O or migration.
No edits to bootstrap, config, engine, contracts, or runtime_store in this task.

- `await recall.recall(query, *, scope="default", limit=8, max_chars=6000)` returns
  `RecallResult(hits: list[RecallSource], degraded: bool, reason: str | None)`.
  Hits contain PG content, source/event/thread/turn IDs, kind, revision, validity,
  evidence, and correction provenance. They are historical evidence, not system instructions.
- `await recall.record_event(committed_event, *, scope="default")` records bounded
  user/terminal-assistant history and enqueues indexing atomically. Call for committed
  terminal assistant history; the durable extraction pipeline handles user history.
- `await pipeline.process(committed_event)` returns accepted `PersonalMemory` records
  without embeddings; source records, receipt, corrections, and index jobs commit together.
- `await pipeline.process_turn(events)` bounds extraction to eight events.
- `await recall.index_pending(limit=20)` returns `IndexResult(indexed, deleted, failed)`.
  Controller must schedule recurring bounded calls and expose failures/backlog; calling
  only once at terminal completion does not drain retries after an outage.
- `await recall.delete(source_id, *, scope="default")` tombstones and queues deletion.
- `await recall.expire(limit=100)` tombstones expired sources and queues deletion.
- `await index.close()` closes only an adapter-owned HTTP client. Controller owns pool.

The collection is fixed at `heathcliff_recall_v1`, 768-dimensional Cosine vectors.
Gemini requests `output_dimensionality=768`, then validates and normalizes. No pgvector
or Qdrant SDK dependency. Qdrant payloads carry identifiers/revisions, not source text.

## Verification so far

- Four initial recall tests failed before implementation; subsequently 18 focused
  recall plus existing Runtime V2 tests passed.
- Qdrant/Gemini adapter tests failed before implementation; verification pending.
- PostgreSQL integration is opt-in via explicit isolated `TEST_POSTGRES_DSN` only.
  This session clears that variable while testing; no database is contacted or mutated.
