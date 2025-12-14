# ABOUTME: Langfuse instrumentation helpers for Heathcliff
# ABOUTME: Handles callback registration and manual trace/event logging

from __future__ import annotations

from typing import Any, Dict, Optional

from config import get_config
from logger import logger

try:  # Optional dependency
    from langfuse import Langfuse
except Exception as exc:  # pragma: no cover - langfuse not installed
    Langfuse = None  # type: ignore
    _LANGFUSE_IMPORT_ERROR = exc
else:  # pragma: no cover - debug helper
    _LANGFUSE_IMPORT_ERROR = None

CallbackHandler = None  # type: ignore
_CALLBACK_IMPORT_ERROR: Optional[Exception] = None

if CallbackHandler is None:  # pragma: no cover - import detection
    try:
        from langfuse.langchain import CallbackHandler  # type: ignore
    except Exception as exc_langchain:
        try:
            from langfuse.callback import CallbackHandler  # type: ignore[assignment]
        except Exception as exc_legacy:
            _CALLBACK_IMPORT_ERROR = exc_legacy
            CallbackHandler = None  # type: ignore
        else:
            _CALLBACK_IMPORT_ERROR = None
    else:
        _CALLBACK_IMPORT_ERROR = None

_langfuse_client: Optional["Langfuse"] = None
_langfuse_handler: Optional["CallbackHandler"] = None


def _is_enabled(config) -> bool:
    if not config.get("observability.langfuse.enabled", True):
        return False
    return bool(config.langfuse_public_key and config.langfuse_secret_key)


def _resolve_base_url(config) -> Optional[str]:
    return (
        config.get("observability.langfuse.base_url")
        or config.langfuse_base_url
        or config.langfuse_host
    )


def _build_kwargs(config) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "public_key": config.langfuse_public_key,
        "secret_key": config.langfuse_secret_key,
    }

    base_url = _resolve_base_url(config)
    if base_url:
        kwargs["host"] = base_url

    if config.langfuse_release:
        kwargs["release"] = config.langfuse_release

    return kwargs


def get_langfuse_client() -> Optional["Langfuse"]:
    """Return cached Langfuse client instance if configuration exists."""

    global _langfuse_client
    config = get_config()

    if Langfuse is None:
        if _LANGFUSE_IMPORT_ERROR:
            logger.debug(
                "Langfuse SDK unavailable: %s. Install `langfuse` to enable traces.",
                _LANGFUSE_IMPORT_ERROR,
            )
        return None

    if not _is_enabled(config):
        return None

    if _langfuse_client is None:
        kwargs = _build_kwargs(config)
        try:
            _langfuse_client = Langfuse(**kwargs)
            logger.info("Langfuse client initialized")
        except TypeError:
            # Older SDKs don't support 'release' kwarg
            kwargs.pop("release", None)
            _langfuse_client = Langfuse(**kwargs)
            logger.info("Langfuse client initialized without release metadata")
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.warning(f"Unable to initialize Langfuse client: {exc}")
            _langfuse_client = None

    return _langfuse_client


def get_langfuse_callback_handler() -> Optional["CallbackHandler"]:
    """
    Create or return Langfuse callback handler for LangChain-compatible components.
    """

    global _langfuse_handler
    config = get_config()

    if CallbackHandler is None:
        if _CALLBACK_IMPORT_ERROR:
            logger.warning(
                "Langfuse callback handler not available: %s", _CALLBACK_IMPORT_ERROR
            )
        return None

    if not _is_enabled(config):
        return None

    # Ensure the client is initialized before instantiating handler
    client = get_langfuse_client()
    if not client:
        logger.warning(
            "Langfuse client unavailable; callback handler will not be registered"
        )
        return None

    if _langfuse_handler is None:
        try:
            _langfuse_handler = CallbackHandler()
            logger.info("Langfuse callback handler registered")
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.warning(f"Unable to initialize Langfuse callback handler: {exc}")
            _langfuse_handler = None

    return _langfuse_handler


def log_langfuse_interaction(
    session_id: str,
    user_input: str,
    response: str,
    status: str = "success",
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Record a high-level agent trace for observability dashboards."""

    client = get_langfuse_client()
    if not client or not hasattr(client, "trace"):
        return

    config = get_config()
    user_id = config.get("observability.langfuse.user_id", "adiagarwal")
    metadata: Dict[str, Any] = {
        "session_id": session_id,
        "status": status,
    }
    environment = config.get("observability.langfuse.environment")
    if environment:
        metadata["environment"] = environment
    if extra_metadata:
        metadata.update(extra_metadata)

    payload = {
        "name": config.get("observability.langfuse.trace_name", "heathcliff.agent"),
        "input": {"user_input": user_input},
        "output": {"assistant_response": response},
        "user_id": user_id,
        "metadata": metadata,
    }

    try:
        client.trace(**payload)
    except TypeError:
        # Older SDKs might not support metadata dictionaries
        payload.pop("metadata", None)
        try:
            client.trace(**payload)
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.debug(f"Langfuse trace logging skipped: {exc}")
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.debug(f"Langfuse trace logging skipped: {exc}")


def log_langfuse_tool_event(
    session_id: str,
    tool_name: str,
    tool_args: Dict[str, Any],
    result: Any,
    status: str = "success",
) -> None:
    """Log tool execution results to Langfuse for debugging."""

    client = get_langfuse_client()
    if not client or not hasattr(client, "event"):
        return

    event_payload = {
        "name": f"tool::{tool_name}",
        "input": {"args": tool_args},
        "output": {"result": result},
        "metadata": {
            "session_id": session_id,
            "status": status,
            "environment": get_config().get("observability.langfuse.environment"),
            "user_id": get_config().get("observability.langfuse.user_id", "adiagarwal"),
        },
        "level": "DEFAULT",
    }

    try:
        client.event(**event_payload)
    except TypeError:
        event_payload.pop("metadata", None)
        try:
            client.event(**event_payload)
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.debug(f"Langfuse tool event logging skipped: {exc}")
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.debug(f"Langfuse tool event logging skipped: {exc}")


if __name__ == "__main__":  # pragma: no cover - manual diagnostic helper
    client = get_langfuse_client()
    if not client:
        print(
            "Langfuse client unavailable. Check that LANGFUSE_PUBLIC_KEY / SECRET_KEY / BASE_URL are set."
        )
    else:
        try:
            print("Langfuse auth check:", client.auth_check())
        except Exception as exc:
            print("Langfuse auth check failed:", exc)
