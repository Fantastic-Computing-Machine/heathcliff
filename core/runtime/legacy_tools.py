"""Conservative bridge from existing integrations to Runtime V2 contracts."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from typing import Any, cast

from core.approval_handler import requires_approval
from core.delegation.registry import build_default_registry
from core.runtime.contracts import (
    ApprovalPolicy,
    ParallelSafety,
    PrivacyClass,
    ResourceScope,
    ToolContract,
    ToolEffect,
)
from core.runtime.tools import ToolRegistry

_DESCRIPTIONS = {
    "info_agent_tool": "Retrieve researched information, news, weather, and web evidence.",
    "music_agent_tool": "Read or control the user's music provider with verified state.",
    "email_agent_tool": "Read, draft, or send the user's email.",
    "calendar_agent_tool": "Read or change the user's calendar.",
    "contacts_agent_tool": "Look up the user's contacts.",
    "comms_agent_tool": "Read or send messages through approved communication channels.",
    "recent_context": "Retrieve recent conversation context.",
}


def build_legacy_tool_bridge(tool_model: str | None = None) -> ToolRegistry:
    """Expose the current integrations as typed tools while direct tools migrate."""
    tools = ToolRegistry()
    for descriptor in build_default_registry(tool_model=tool_model).all_agents():
        effect = (
            ToolEffect.EXTERNAL_SIDE_EFFECT
            if descriptor.sensitive_actions
            else ToolEffect.READ
        )

        async def execute(
            arguments: dict[str, Any], invoke=descriptor.invoke_fn
        ) -> Any:
            request = str(arguments["request"])

            def call() -> Any:
                tool_invoke = getattr(invoke, "invoke", None)
                if callable(tool_invoke):
                    return tool_invoke({"request": request})
                if callable(invoke):
                    return cast(Callable[..., Any], invoke)(request=request)
                raise TypeError("Legacy integration has no callable entry point")

            result = await asyncio.to_thread(call)
            return await result if inspect.isawaitable(result) else result

        tools.register(
            ToolContract(
                name=descriptor.name,
                description=_DESCRIPTIONS.get(descriptor.name, descriptor.name),
                input_schema={
                    "type": "object",
                    "properties": {"request": {"type": "string"}},
                    "required": ["request"],
                    "additionalProperties": False,
                },
                effect=effect,
                approval_policy=(
                    ApprovalPolicy.ALWAYS
                    if requires_approval(descriptor.name)
                    else ApprovalPolicy.NEVER
                ),
                parallel_safety=(
                    ParallelSafety.EXCLUSIVE
                    if effect != ToolEffect.READ
                    else ParallelSafety.SAFE_READ
                ),
                resource_scope=ResourceScope(resource=f"agent:{descriptor.name}"),
                trace_privacy=(
                    PrivacyClass.SENSITIVE
                    if effect == ToolEffect.EXTERNAL_SIDE_EFFECT
                    else PrivacyClass.NORMAL
                ),
            ),
            execute,
        )
    return tools
