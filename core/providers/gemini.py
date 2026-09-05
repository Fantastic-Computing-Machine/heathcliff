"""Native Gemini adapter preserving function-call and thought-signature metadata."""

from __future__ import annotations

import base64
from collections.abc import AsyncIterator
from typing import Any, cast

from core.providers.base import ModelProvider
from core.runtime.contracts import (
    ModelEvent,
    ModelRequest,
    PreparedModelCall,
    ProviderCapabilities,
)


class GeminiProvider(ModelProvider):
    """Small native Google GenAI adapter; no OpenAI compatibility translation."""

    def __init__(self, api_key: str, model: str, context_window: int = 32768) -> None:
        self.api_key = api_key
        self.capabilities = ProviderCapabilities(
            provider="google-genai",
            model=model,
            thought_signatures=True,
            multimodal=True,
            context_window=context_window,
        )

    async def prepare(self, request: ModelRequest) -> PreparedModelCall:
        if request.provider != self.capabilities:
            raise ValueError("Gemini request capabilities do not match this provider")
        return PreparedModelCall(
            request=request,
            effective_config={
                "provider": "google-genai",
                "model": self.capabilities.model,
                "native_tools": True,
            },
        )

    @staticmethod
    def _contents(request: ModelRequest) -> list[dict[str, Any]]:
        return [
            {"role": "user", "parts": [{"text": str(item.content)}]}
            for item in request.context
        ]

    @staticmethod
    def _function_schema(value: Any) -> Any:
        """Remove JSON Schema fields Gemini's function declaration API rejects."""
        if isinstance(value, list):
            return [GeminiProvider._function_schema(item) for item in value]
        if isinstance(value, dict):
            return {
                key: GeminiProvider._function_schema(item)
                for key, item in value.items()
                if key != "additionalProperties"
            }
        return value

    @staticmethod
    def _thought_signature(value: Any) -> str | None:
        """Keep Gemini's opaque continuation token JSON-safe for event replay."""
        if value is None:
            return None
        if isinstance(value, bytes):
            return base64.b64encode(value).decode("ascii")
        return str(value)

    async def stream(self, call: PreparedModelCall) -> AsyncIterator[ModelEvent]:
        """Yield normalized native Gemini events."""
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - dependency configuration
            raise RuntimeError(
                "google-genai is required for Gemini Runtime V2"
            ) from exc

        request = call.request
        client = genai.Client(api_key=self.api_key)
        config: dict[str, Any] = {"system_instruction": request.system_instruction}
        if request.tools:
            config["tools"] = [
                {
                    "function_declarations": [
                        {
                            "name": tool["name"],
                            "description": tool["description"],
                            "parameters": self._function_schema(tool["input_schema"]),
                        }
                        for tool in request.tools
                    ]
                }
            ]
        stream = await client.aio.models.generate_content_stream(
            model=self.capabilities.model,
            contents=cast(Any, self._contents(request)),
            config=cast(Any, config),
        )
        async for chunk in stream:
            for candidate in getattr(chunk, "candidates", []) or []:
                content = getattr(candidate, "content", None)
                for part in getattr(content, "parts", []) or []:
                    thought_signature = self._thought_signature(
                        getattr(part, "thought_signature", None)
                    )
                    if getattr(part, "text", None):
                        yield ModelEvent(
                            kind="text_delta",
                            data={
                                "text": part.text,
                                "thought_signature": thought_signature,
                            },
                        )
                    function_call = getattr(part, "function_call", None)
                    if function_call:
                        yield ModelEvent(
                            kind="tool_call",
                            data={
                                "name": function_call.name,
                                "arguments": dict(function_call.args or {}),
                                "thought_signature": thought_signature,
                            },
                        )
            usage = getattr(chunk, "usage_metadata", None)
            if usage:
                yield ModelEvent(kind="usage", data={"raw": str(usage)})
        yield ModelEvent(kind="completed")
