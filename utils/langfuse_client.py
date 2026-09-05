# ABOUTME: Langfuse instrumentation helpers for Heathcliff
# ABOUTME: Handles callback registration and manual trace/event logging

from __future__ import annotations

import os
from contextlib import ExitStack, contextmanager
from typing import Any, Dict, Generator, Iterable, Literal, Optional

from langfuse import Langfuse, propagate_attributes  # noqa: F401 (re-exported)
from langfuse.langchain import CallbackHandler
from opentelemetry import trace

from config import Config
from logger import logger

_langfuse_client: Optional[Langfuse] = None


def _is_enabled() -> bool:
    return bool(Config.LANGFUSE_PUBLIC_KEY and Config.LANGFUSE_SECRET_KEY)


def _build_client_kwargs() -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "public_key": Config.LANGFUSE_PUBLIC_KEY,
        "secret_key": Config.LANGFUSE_SECRET_KEY,
    }
    host = Config.LANGFUSE_BASE_URL or Config.LANGFUSE_HOST
    if host:
        kwargs["host"] = host
    if Config.LANGFUSE_RELEASE:
        kwargs["release"] = Config.LANGFUSE_RELEASE
    kwargs["environment"] = Config.ENVIRONMENT
    return kwargs


def get_langfuse_client() -> Optional[Langfuse]:
    """Return a cached Langfuse client, initializing on first call."""
    global _langfuse_client

    if not _is_enabled():
        return None

    if _langfuse_client is None:
        kwargs = _build_client_kwargs()
        try:
            _langfuse_client = Langfuse(**kwargs)
            logger.info(
                "Langfuse v4 client initialized (environment=%s)",
                Config.ENVIRONMENT,
            )
        except Exception as exc:
            logger.warning(f"Unable to initialize Langfuse client: {exc}")

    return _langfuse_client


def get_langfuse_callback_handler() -> Optional[CallbackHandler]:
    """Build a LangChain callback handler for one Heathcliff request.

    Session and user context should be injected per-request via
    ``propagate_attributes(session_id=..., user_id=...)`` rather than
    through handler constructor kwargs (not supported in Langfuse v4).

    Callback handlers retain LangChain run IDs while a request is active, so
    sharing one between concurrent requests disconnects child observations.
    """
    if not _is_enabled():
        return None

    if not get_langfuse_client():
        logger.warning(
            "Langfuse client unavailable; callback handler will not be registered"
        )
        return None

    try:
        handler = CallbackHandler(public_key=Config.LANGFUSE_PUBLIC_KEY)
        logger.debug("Langfuse callback handler initialized")
        return handler
    except Exception as exc:
        logger.warning(f"Unable to initialize Langfuse callback handler: {exc}")
        return None


def trace_tags(explicit_tags: Optional[Iterable[str]] = None) -> list[str]:
    """Return stable request tags, marking automated pytest runs distinctly."""
    tags = [tag.strip() for tag in explicit_tags or () if tag and tag.strip()]
    if os.getenv("PYTEST_CURRENT_TEST"):
        tags.extend(("test", "pytest"))
    return list(dict.fromkeys(tags))


def flush_langfuse(client: Optional[Langfuse] = None) -> None:
    """Synchronously export one completed request without affecting its result."""
    active_client = client or get_langfuse_client()
    if active_client is None:
        return
    try:
        active_client.flush()
        logger.debug("Langfuse trace flush completed")
    except Exception as exc:
        logger.warning("Unable to flush Langfuse traces: %s", exc)


@contextmanager
def trace_runtime_turn(
    *, thread_id: str, turn_id: str, user_input: str
) -> Generator[Any | None, None, None]:
    """Trace one native Runtime V2 turn independently of LangChain callbacks."""
    client = get_langfuse_client()
    if client is None:
        yield None
        return

    attributes = {
        "trace_name": f"{Config.TRACE_NAME}.v2",
        "session_id": thread_id,
        "user_id": Config.LANGFUSE_USER_ID,
        "environment": Config.ENVIRONMENT,
        "version": Config.LANGFUSE_VERSION,
        "metadata": {"runtime": "v2", "thread_id": thread_id, "turn_id": turn_id},
        "tags": trace_tags(["runtime-v2"]),
    }
    try:
        with ExitStack() as stack:
            try:
                stack.enter_context(propagate_attributes(**attributes))
                observation = stack.enter_context(
                    client.start_as_current_observation(
                        name="heathcliff.runtime.v2",
                        as_type="agent",
                        input={"user_input": user_input, "turn_id": turn_id},
                    )
                )
            except Exception as exc:
                logger.warning("Unable to start Runtime V2 trace: %s", exc)
                yield None
                return
            yield observation
    finally:
        flush_langfuse(client)


@contextmanager
def trace_observation(
    name: str,
    *,
    input: Any = None,
    as_type: Literal["agent", "chain", "generation", "span", "tool"] = "span",
) -> Generator[Any | None, None, None]:
    """Nest a coordinator step under the active Langfuse request, if any."""
    if not trace.get_current_span().get_span_context().is_valid:
        yield None
        return

    client = get_langfuse_client()
    if client is None:
        yield None
        return

    try:
        observation_context = client.start_as_current_observation(
            name=name,
            as_type=as_type,
            input=input,
        )
    except Exception as exc:
        logger.warning("Unable to start Langfuse observation %s: %s", name, exc)
        yield None
        return

    with observation_context as observation:
        yield observation


def is_langfuse_callback_handler(callback: Any) -> bool:
    """Return whether a callback is Langfuse's LangChain adapter."""
    return isinstance(callback, CallbackHandler)


if __name__ == "__main__":  # pragma: no cover - manual diagnostic helper
    client = get_langfuse_client()
    if not client:
        print(
            "Langfuse client unavailable. Check LANGFUSE_PUBLIC_KEY / SECRET_KEY / HOST env vars."
        )
    else:
        try:
            print("Langfuse auth check:", client.auth_check())
        except Exception as exc:
            print("Langfuse auth check failed:", exc)
