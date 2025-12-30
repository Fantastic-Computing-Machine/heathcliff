# ABOUTME: Middleware configuration for agent execution control
# ABOUTME: Includes moderation, tool selection, rate limiting, and summarization

from typing import Any, List, Optional

from logger import logger


def create_middleware_stack(
    llm: Optional[Any] = None,
) -> List[Any]:
    """
    Create middleware stack for agent execution.

    Middleware order matters - they execute in the order they're added:
    1. Tool selection (filter tools before model call)
    2. Model call limits (prevent excessive LLM calls)
    3. Tool call limits (prevent excessive tool calls)
    4. Tool retry (handle transient failures)
    5. Summarization (manage conversation history)

    Args:
        config: Configuration object with middleware settings
        llm: Optional LLM instance for tool selection and summarization

    Returns:
        List of middleware instances
    """
    logger.info("Middleware disabled (class-based config has no middleware section).")
    return []
