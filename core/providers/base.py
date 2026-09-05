"""Provider boundary; Runtime V2 never stores vendor SDK objects."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from core.runtime.contracts import (
    ModelEvent,
    ModelRequest,
    PreparedModelCall,
    ProviderCapabilities,
)


class ModelProvider(Protocol):
    capabilities: ProviderCapabilities

    async def prepare(self, request: ModelRequest) -> PreparedModelCall: ...

    def stream(self, call: PreparedModelCall) -> AsyncIterator[ModelEvent]: ...
