"""Offline recall checks; PostgreSQL tests require an explicit isolated DSN."""

import asyncio
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from datetime import timedelta
from uuid import UUID, uuid4

import pytest

from core.runtime.contracts import EventKind, RuntimeEvent, utcnow


class Embedder:
    async def embed(self, text):
        return [1.0] + [0.0] * 767


class Index:
    def __init__(self):
        self.points = {}
        self.unavailable = False

    async def upsert(self, source, vector):
        from core.runtime.recall import point_id

        if self.unavailable:
            raise OSError("offline")
        self.points[point_id(source.id, source.revision)] = {
            "id": str(point_id(source.id, source.revision)),
            "score": 0.9,
            "payload": {"source_id": str(source.id), "revision": source.revision},
        }

    async def delete(self, source_id, revision):
        from core.runtime.recall import point_id

        if self.unavailable:
            raise OSError("offline")
        self.points.pop(point_id(source_id, revision), None)

    async def search(self, vector, *, scope, limit):
        if self.unavailable:
            raise OSError("offline")
        return list(self.points.values())[:limit]


def setup_recall():
    from core.runtime.recall import RecallService
    from db.recall_store import InMemoryRecallStore

    repo, index = InMemoryRecallStore(), Index()
    return repo, index, RecallService(repo, Embedder(), index)


def event(text="I prefer tea", **kwargs):
    return RuntimeEvent(
        thread_id=uuid4(), kind=EventKind.INPUT_ADMITTED,
        payload={"content": text}, **kwargs,
    )


def test_source_survives_index_outage_and_retries_without_duplicate_points():
    async def run():
        repo, index, service = setup_recall()
        source = await service.record_event(event())
        index.unavailable = True
        result = await service.index_pending()
        assert result.failed == 1
        assert (await repo.get(source.id)).content == "I prefer tea"
        recalled = await service.recall("What do I drink?")
        assert recalled.degraded and recalled.reason == "semantic_index_unavailable"
        assert recalled.hits == []
        index.unavailable = False
        # The clock is injected into the repository, not slept in tests.
        repo.clock = lambda: utcnow() + timedelta(hours=1)
        assert (await service.index_pending()).indexed == 1
        await service.record_event(RuntimeEvent(
            id=source.source_event_id, thread_id=source.thread_id,
            kind=EventKind.INPUT_ADMITTED, payload={"content": "I prefer tea"},
        ))
        assert (await service.index_pending()).indexed == 0
        assert len(index.points) == 1
        hit = (await service.recall("beverage")).hits[0]
        assert hit.content == "I prefer tea"
        assert hit.source_event_id == source.source_event_id
        assert hit.thread_id == source.thread_id

    asyncio.run(run())


def test_pg_authority_filters_stale_deleted_expired_foreign_and_forged_hits():
    async def run():
        repo, index, service = setup_recall()
        source = await service.record_event(event())
        await service.index_pending()
        stale = dict(next(iter(index.points.values())))
        changed = await repo.save(source.model_copy(update={"content": "I prefer coffee"}),
                                  expected_revision=1)
        assert changed.revision == 2
        assert (await service.recall("drink")).hits == []
        await service.index_pending()
        index.points[uuid4()] = stale
        index.points[uuid4()] = {"id": str(uuid4()), "score": 1,
                                "payload": {"source_id": str(source.id), "revision": 2}}
        assert [h.content for h in (await service.recall("drink")).hits] == ["I prefer coffee"]
        await service.delete(source.id)
        assert (await service.recall("drink")).hits == []
        assert await repo.save(source) is None
        assert (await service.index_pending()).deleted >= 1
        other = await service.record_event(event("another user"), scope="other")
        expired = await service.record_event(event("old"))
        await repo.save(expired.model_copy(update={"valid_until": utcnow() - timedelta(seconds=1)}),
                        expected_revision=1)
        await service.index_pending()
        assert other is not None
        assert (await service.recall("anything")).hits == []

    asyncio.run(run())


def test_pipeline_requires_evidence_and_committed_user_sources_then_tombstones_corrections():
    async def run():
        from core.runtime.memory import EvidenceMemoryCandidate, SemanticMemoryPipeline
        from db.runtime_store import InMemoryRuntimeStore

        repo, index, recall = setup_recall()
        runtime = InMemoryRuntimeStore()
        thread = await runtime.create_thread()
        source = await runtime.append_event(RuntimeEvent(
            thread_id=thread.id, kind=EventKind.INPUT_ADMITTED,
            payload={"content": "I prefer tea"},
        ))

        class Classifier:
            candidates = []

            async def classify(self, source):
                return self.candidates

        classifier = Classifier()
        base = dict(kind="preference", subject="drink", content="prefers tea",
                    confidence=1, source_event_id=source.id, source_kind="user",
                    evidence_quote="I prefer tea")
        classifier.candidates = [EvidenceMemoryCandidate(**base)]
        pipeline = SemanticMemoryPipeline(runtime, classifier, Embedder(), recall=recall)
        accepted = await pipeline.process(source)
        assert len(accepted) == 1
        assert accepted[0].embedding is None
        assert await pipeline.process(source) == []
        await recall.index_pending()
        assert {h.kind for h in (await recall.recall("drink")).hits} == {"history", "fact"}
        correction = await runtime.append_event(RuntimeEvent(
            thread_id=thread.id, kind=EventKind.INPUT_ADMITTED,
            payload={"content": "I now prefer coffee"},
        ))
        classifier.candidates = [EvidenceMemoryCandidate(**{
            **base, "kind": "correction", "content": "prefers coffee",
            "source_event_id": correction.id, "evidence_quote": "I now prefer coffee",
            "supersedes_id": accepted[0].id,
        })]
        new = await pipeline.process(correction)
        assert len(new) == 1
        assert (await repo.get(accepted[0].id)).deleted_at is not None
        await recall.delete(new[0].id)
        assert await pipeline.process(correction) == []
        assert await pipeline.process(source) == []
        invalid = await runtime.append_event(RuntimeEvent(
            thread_id=thread.id, kind=EventKind.INPUT_ADMITTED,
            payload={"content": "hello"},
        ))
        classifier.candidates = [EvidenceMemoryCandidate(**{
            **base, "source_event_id": invalid.id,
        })]
        assert await pipeline.process(invalid) == []
        assert await pipeline.process(event()) == []  # not committed

    asyncio.run(run())


def test_revision_point_ids_are_uuid_and_distinct():
    from core.runtime.recall import point_id

    source_id = UUID("00000000-0000-0000-0000-000000000001")
    assert isinstance(point_id(source_id, 1), UUID)
    assert point_id(source_id, 1) == point_id(source_id, 1)
    assert point_id(source_id, 1) != point_id(source_id, 2)


def test_qdrant_rest_uses_dedicated_collection_and_checks_vector_configuration():
    import httpx
    from core.runtime.recall import QdrantIndex, point_id

    async def run():
        requests = []

        def respond(request):
            requests.append(request)
            path = request.url.path
            assert path.startswith("/collections/heathcliff_recall_v1")
            if request.method == "GET":
                return httpx.Response(404)
            if path.endswith("/query"):
                return httpx.Response(200, json={"result": {"points": []}})
            return httpx.Response(200, json={"result": True, "status": "ok"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            index = QdrantIndex("http://qdrant.invalid", api_key="test-only", client=client)
            repo, _, service = setup_recall()
            source = await service.record_event(event())
            await index.upsert(source, [1.0] + [0.0] * 767)
            await index.search([1.0] + [0.0] * 767, scope="default", limit=8)
            await index.delete(source.id, source.revision)
        create = json.loads(requests[1].content)
        assert create["vectors"] == {"size": 768, "distance": "Cosine"}
        point = json.loads(requests[2].content)["points"][0]
        assert UUID(point["id"]) == point_id(source.id, 1)
        assert "content" not in point["payload"]
        assert point["payload"]["scope"] == "default"
        query = json.loads(requests[3].content)
        assert query["filter"]["must"] == [{"key": "scope", "match": {"value": "default"}}]
        assert requests[2].url.params["wait"] == "true"

        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request:
            httpx.Response(200, json={"result": {"config": {"params": {
                "vectors": {"size": 3072, "distance": "Cosine"}
            }}}}))) as client:
            index = QdrantIndex("http://qdrant.invalid", client=client)
            with pytest.raises(ValueError, match="768"):
                await index.ensure_collection()

    asyncio.run(run())


def test_gemini_embedding_requests_768_normalizes_and_rejects_bad_vectors():
    from core.runtime.gemini_memory import GeminiEmbedder

    async def run():
        class Models:
            values = [3.0, 4.0] + [0.0] * 766

            async def embed_content(self, **kwargs):
                assert kwargs["model"] == "configured-embedding-model"
                assert kwargs["config"]["output_dimensionality"] == 768
                return SimpleNamespace(embeddings=[SimpleNamespace(values=self.values)])

        models = Models()
        client = SimpleNamespace(aio=SimpleNamespace(models=models))
        embedder = GeminiEmbedder("unused", model="configured-embedding-model", client=client)
        vector = await embedder.embed("a beverage")
        assert vector[:2] == [0.6, 0.8]
        assert math.isclose(sum(x * x for x in vector), 1)
        for values in ([0.0] * 768, [1.0], [float("nan")] * 768):
            models.values = values
            with pytest.raises(ValueError):
                await embedder.embed("invalid")

    asyncio.run(run())


def test_claim_fences_old_worker_and_expiration_is_durable():
    async def run():
        repo, index, service = setup_recall()
        source = await service.record_event(event())
        first = await repo.claim(lease_seconds=1)
        assert await repo.claim() is None
        repo.clock = lambda: utcnow() + timedelta(seconds=2)
        second = await repo.claim()
        assert second.id == first.id and second.token != first.token
        assert not await repo.finish(first, succeeded=True)
        assert await repo.finish(second, succeeded=False)
        changed = await repo.save(source.model_copy(update={"valid_until": utcnow() - timedelta(seconds=1)}), expected_revision=1)
        assert await service.expire() == 1
        assert (await repo.get(changed.id)).deleted_at is not None
        assert await service.expire() == 0

    asyncio.run(run())


def test_postgres_source_outbox_atomicity_and_restart():
    """Opt-in only: fresh schema on TEST_POSTGRES_DSN, never DATABASE_URL."""
    dsn = os.environ.get("TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("requires an explicit isolated TEST_POSTGRES_DSN")

    import asyncpg
    from db.recall_store import PostgresRecallStore

    async def run():
        schema = "recall_test_" + uuid4().hex
        conn = await asyncpg.connect(dsn)
        pool = None
        try:
            await conn.execute(f'CREATE SCHEMA "{schema}"')
            await conn.execute(f'SET search_path TO "{schema}"')
            # Only the journal boundary is needed; no live migrations or pgvector.
            await conn.execute("""CREATE TABLE runtime_events (
                id uuid PRIMARY KEY, thread_id uuid NOT NULL, turn_id uuid,
                kind text NOT NULL, payload jsonb NOT NULL, created_at timestamptz NOT NULL
            )""")
            sql = (Path(__file__).parents[1] / "db/migrations/003_runtime_recall.sql").read_text()
            await conn.execute(sql)
            pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2,
                                           server_settings={"search_path": schema})
            repo = PostgresRecallStore(pool)
            from core.runtime.recall import RecallService

            service = RecallService(repo, Embedder(), Index())
            ev = event()
            await conn.execute("INSERT INTO runtime_events VALUES ($1,$2,$3,$4,$5::jsonb,$6)",
                               ev.id, ev.thread_id, ev.turn_id, ev.kind.value,
                               json.dumps(ev.payload), ev.created_at)
            source = await service.record_event(ev)
            assert await conn.fetchval("SELECT count(*) FROM runtime_recall_jobs") == 1
            assert (await PostgresRecallStore(pool).get(source.id)).content == "I prefer tea"
            with pytest.raises(ValueError):
                await repo.accept([source.model_copy(update={"id": uuid4(), "supersedes_id": uuid4()})], ev.id)
            assert not await repo.extracted(ev.id)
            assert await conn.fetchval("SELECT count(*) FROM runtime_recall_sources") == 1
            jobs = await asyncio.gather(repo.claim(), repo.claim())
            assert sum(j is not None for j in jobs) == 1
            claimed = next(j for j in jobs if j is not None)
            assert await repo.finish(claimed, succeeded=True)
            revised = await repo.save(source.model_copy(update={"valid_until": utcnow() - timedelta(seconds=1)}), expected_revision=1)
            assert revised.revision == 2
            assert await service.expire() == 1
            assert await repo.save(source) is None
            assert await repo.extracted(ev.id)
            assert await conn.fetchval("SELECT count(*) FROM runtime_recall_revisions") == 3
            assert (await service.recall("beverage")).hits == []
        finally:
            if pool is not None:
                await pool.close()
            await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
            await conn.close()

    asyncio.run(run())
