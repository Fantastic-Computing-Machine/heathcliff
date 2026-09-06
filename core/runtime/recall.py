"""Bounded semantic recall. Qdrant proposes IDs; PostgreSQL supplies the truth."""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from uuid import NAMESPACE_URL, UUID, uuid5

import httpx

from core.runtime.contracts import EventKind, RuntimeEvent, utcnow
from db.recall_store import RecallSource

COLLECTION = "heathcliff_recall_v1"
DIMENSIONS = 768


def point_id(source_id: UUID, revision: int) -> UUID:
    """Revision-specific IDs fence late workers without Qdrant transactions."""
    return uuid5(NAMESPACE_URL, f"heathcliff:recall:v1:{source_id}:{revision}")


def normalize_embedding(values: list[float]) -> list[float]:
    if len(values) != DIMENSIONS or any(not math.isfinite(v) for v in values):
        raise ValueError("Recall requires 768 finite embedding values")
    norm = math.hypot(*values)
    if not math.isfinite(norm) or norm == 0:
        raise ValueError("Recall embedding must have a finite nonzero norm")
    return [v / norm for v in values]


@dataclass
class RecallResult:
    hits: list[RecallSource] = field(default_factory=list)
    degraded: bool = False
    reason: str | None = None


@dataclass
class IndexResult:
    indexed: int = 0
    deleted: int = 0
    failed: int = 0


class QdrantIndex:
    """Small async REST adapter; owns only the dedicated recall collection."""

    def __init__(self, url: str, *, api_key: str = "", client: httpx.AsyncClient | None = None,
                 timeout_seconds: float = 5):
        parsed = httpx.URL(url)
        if parsed.scheme not in {"http", "https"} or not parsed.host or parsed.userinfo or parsed.query:
            raise ValueError("Qdrant URL must be an HTTP(S) origin without credentials")
        self.url = url.rstrip("/") + f"/collections/{COLLECTION}"
        self._owned_client = client is None
        self.client = client or httpx.AsyncClient()
        self.headers = {"api-key": api_key} if api_key else {}
        self.timeout_seconds = timeout_seconds
        self._ready = False
        self._collection_lock = asyncio.Lock()

    async def _request(self, method: str, suffix: str = "", **kwargs):
        response = await self.client.request(method, self.url + suffix, headers=self.headers,
                                             timeout=self.timeout_seconds, **kwargs)
        if response.status_code == 404:
            self._ready = False
        return response

    async def ensure_collection(self) -> None:
        async with self._collection_lock:
            if self._ready:
                return
            response = await self._request("GET")
            if response.status_code == 404:
                response = await self._request("PUT", json={"vectors": {"size": DIMENSIONS, "distance": "Cosine"}})
                if response.status_code == 409:
                    response = await self._request("GET")
                else:
                    response.raise_for_status()
                    self._ready = True
                    return
            response.raise_for_status()
            vectors = response.json()["result"]["config"]["params"]["vectors"]
            if vectors.get("size") != DIMENSIONS or vectors.get("distance") != "Cosine":
                raise ValueError("Recall collection must use 768-dimensional Cosine vectors")
            self._ready = True

    async def upsert(self, source: RecallSource, vector: list[float]) -> None:
        vector = normalize_embedding(vector)
        await self.ensure_collection()
        response = await self._request("PUT", "/points", params={"wait": "true"}, json={"points": [{
            "id": str(point_id(source.id, source.revision)), "vector": vector,
            "payload": {"source_id": str(source.id), "revision": source.revision,
                        "scope": source.scope, "kind": source.kind},
        }]})
        response.raise_for_status()

    async def delete(self, source_id: UUID, revision: int) -> None:
        await self.ensure_collection()
        response = await self._request("POST", "/points/delete", params={"wait": "true"},
                                       json={"points": [str(point_id(source_id, revision))]})
        response.raise_for_status()

    async def search(self, vector: list[float], *, scope: str, limit: int) -> list[dict]:
        vector = normalize_embedding(vector)
        await self.ensure_collection()
        response = await self._request("POST", "/points/query", json={
            "query": vector, "limit": max(1, min(limit, 80)), "with_payload": True,
            "with_vector": False,
            "filter": {"must": [{"key": "scope", "match": {"value": scope}}]},
        })
        response.raise_for_status()
        points = response.json()["result"]["points"]
        if not isinstance(points, list):
            raise ValueError("Invalid Qdrant candidate response")
        return points

    async def close(self) -> None:
        if self._owned_client:
            await self.client.aclose()


def history_source(event: RuntimeEvent, scope: str = "default") -> RecallSource | None:
    if event.kind not in {EventKind.INPUT_ADMITTED, EventKind.TURN_COMPLETED}:
        return None
    if event.payload.get("contains_secret") or event.payload.get("privacy") == "secret":
        return None
    text = event.payload.get("content") or event.payload.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    return RecallSource(
        id=uuid5(NAMESPACE_URL, f"heathcliff:history:{scope}:{event.id}"),
        scope=scope, kind="history", source_event_id=event.id,
        thread_id=event.thread_id, turn_id=event.turn_id, content=text[:8000],
        source_kind="user" if event.kind == EventKind.INPUT_ADMITTED else "assistant",
        created_at=event.created_at,
    )


class RecallService:
    def __init__(self, repository, embedder, index, *, timeout_seconds: float = 10):
        self.repository = repository
        self.embedder = embedder
        self.index = index
        self.timeout_seconds = timeout_seconds

    async def record_event(self, event: RuntimeEvent, *, scope: str = "default") -> RecallSource | None:
        """Project a committed, non-secret history event; PG verifies provenance."""
        source = history_source(event, scope)
        return await self.repository.save(source) if source else None

    async def recall(self, query: str, *, scope: str = "default", limit: int = 8,
                     max_chars: int = 6000) -> RecallResult:
        if not query.strip():
            return RecallResult()
        limit, max_chars = max(1, min(limit, 20)), max(0, min(max_chars, 16000))
        try:
            async with asyncio.timeout(self.timeout_seconds):
                vector = normalize_embedding(await self.embedder.embed(query[:8000]))
        except Exception:
            return RecallResult(degraded=True, reason="embedding_unavailable")
        try:
            async with asyncio.timeout(self.timeout_seconds):
                points = await self.index.search(vector, scope=scope, limit=min(limit * 4, 80))
        except Exception:
            return RecallResult(degraded=True, reason="semantic_index_unavailable")
        candidates = []
        for point in points[:80]:
            try:
                payload = point["payload"]
                source_id = UUID(payload["source_id"])
                revision = payload["revision"]
                if type(revision) is not int or revision < 1:
                    continue
                if UUID(str(point["id"])) != point_id(source_id, revision):
                    continue
                score = float(point["score"])
                if math.isfinite(score):
                    candidates.append((score, source_id, revision))
            except (KeyError, TypeError, ValueError):
                continue
        try:
            async with asyncio.timeout(self.timeout_seconds):
                sources = await self.repository.get_many(list({c[1] for c in candidates}))
        except Exception:
            return RecallResult(degraded=True, reason="source_store_unavailable")
        hits, seen, remaining = [], set(), max_chars
        for _, source_id, revision in sorted(candidates, reverse=True):
            source = sources.get(source_id)
            if (source is None or source_id in seen or source.scope != scope
                    or source.revision != revision or not source.valid(utcnow())):
                continue
            if len(source.content) > remaining:
                continue
            hits.append(source)
            seen.add(source_id)
            remaining -= len(source.content)
            if len(hits) == limit:
                break
        return RecallResult(hits=hits)

    async def index_pending(self, limit: int = 20) -> IndexResult:
        """Process one bounded batch; the controller schedules subsequent batches."""
        result = IndexResult()
        for _ in range(max(0, min(limit, 100))):
            job = await self.repository.claim(lease_seconds=max(60, int(self.timeout_seconds * 3)))
            if job is None:
                break
            try:
                async with asyncio.timeout(self.timeout_seconds):
                    source = await self.repository.get(job.source_id)
                    if (job.action == "delete" or source is None or source.deleted_at
                            or source.revision != job.revision
                            or (source.valid_until and source.valid_until <= utcnow())):
                        await self.index.delete(job.source_id, job.revision)
                        result.deleted += 1
                    else:
                        vector = normalize_embedding(await self.embedder.embed(source.content))
                        await self.index.upsert(source, vector)
                        # A delete/correction may commit during the network await.
                        latest = await self.repository.get(source.id)
                        if latest is None or latest.deleted_at or latest.revision != job.revision:
                            await self.index.delete(job.source_id, job.revision)
                            result.deleted += 1
                        else:
                            result.indexed += 1
            except Exception:
                await self.repository.finish(job, succeeded=False)
                result.failed += 1
            else:
                await self.repository.finish(job, succeeded=True)
        return result

    async def delete(self, source_id: UUID, *, scope: str = "default") -> bool:
        return await self.repository.delete(source_id, scope=scope)

    async def expire(self, limit: int = 100) -> int:
        return await self.repository.expire(limit=max(1, min(limit, 1000)))
