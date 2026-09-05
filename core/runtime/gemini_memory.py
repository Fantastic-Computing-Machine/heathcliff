"""Native Gemini structured classifier and embedding adapter for personal memory."""

from __future__ import annotations

import json
from typing import Any

from core.runtime.contracts import MemoryCandidate, RuntimeEvent

_MEMORY_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string"},
                    "subject": {"type": "string"},
                    "content": {"type": "string"},
                    "confidence": {"type": "number"},
                    "source_kind": {"type": "string"},
                    "contains_secret": {"type": "boolean"},
                },
                "required": [
                    "kind",
                    "subject",
                    "content",
                    "confidence",
                    "source_kind",
                    "contains_secret",
                ],
            },
        }
    },
    "required": ["candidates"],
}


class GeminiMemoryClassifier:
    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model

    async def classify(self, source: RuntimeEvent) -> list[MemoryCandidate]:
        from google import genai

        client = genai.Client(api_key=self.api_key)
        prompt = (
            "Extract only durable personal memory records from this committed user "
            "event. Return non_memory when nothing should persist. Never infer facts, "
            "store credentials, tokens, or private secrets. Source kind must describe "
            "the evidence, not the assistant's guess.\n\n"
            f"EVENT: {source.model_dump_json()}"
        )
        response = await client.aio.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": _MEMORY_SCHEMA,
            },
        )
        data: dict[str, Any] = json.loads(response.text or '{"candidates": []}')
        return [
            MemoryCandidate(
                **candidate,
                source_event_id=source.id,
            )
            for candidate in data.get("candidates", [])
        ]


class GeminiEmbedder:
    def __init__(self, api_key: str, model: str = "gemini-embedding-001") -> None:
        self.api_key = api_key
        self.model = model

    async def embed(self, text: str) -> list[float]:
        from google import genai

        client = genai.Client(api_key=self.api_key)
        response = await client.aio.models.embed_content(
            model=self.model,
            contents=text,
        )
        embeddings = response.embeddings or []
        if not embeddings or not embeddings[0].values:
            raise ValueError("Gemini returned no embedding")
        return [float(value) for value in embeddings[0].values]
