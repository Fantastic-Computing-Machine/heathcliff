# ABOUTME: Delegation framework for agent-to-agent coordination
# ABOUTME: Exports delegation contracts.

from core.delegation.contracts import (
    ErrorType,
    TaskResult,
    TaskSpec,
    TaskStatus,
)

__all__ = [
    "TaskSpec",
    "TaskResult",
    "TaskStatus",
    "ErrorType",
]
