import functools
import time
from typing import Any, Callable, ParamSpec, TypeVar

from logger import logger

P = ParamSpec("P")
R = TypeVar("R")


def retry(
    max_retries: int = 3,
    error_message: str = "Operation failed",
    exponential_backoff: bool = True,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Retry decorator with optional exponential backoff.

    Args:
        max_retries: Number of retry attempts after the initial call.
        error_message: Message to log when a retryable error occurs.
        exponential_backoff: Whether to use exponential backoff between retries.

    Returns:
        A decorator that wraps the target callable with retry logic.
    """
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            attempt = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception:
                    if attempt >= max_retries:
                        logger.error(
                            "%s. Reached maximum retries (%d).",
                            error_message,
                            max_retries,
                            exc_info=True,
                        )
                        raise

                    attempt += 1
                    logger.warning(
                        "%s. Retrying %s (attempt %d/%d)...",
                        error_message,
                        func.__qualname__,
                        attempt,
                        max_retries,
                        exc_info=True,
                    )

                    if exponential_backoff:
                        sleep_seconds = 2 ** (attempt - 1)
                        time.sleep(sleep_seconds)

        return wrapper

    return decorator
