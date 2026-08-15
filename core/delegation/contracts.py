# ABOUTME: Typed delegation contracts for agent-to-agent coordination
# ABOUTME: TaskSpec and TaskResult with strict Pydantic validation

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """Status of a delegated task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    APPROVAL_REJECTED = "approval_rejected"
    DEPENDENCY_FAILED = "dependency_failed"


class ErrorType(str, Enum):
    """Canonical error taxonomy for delegation failures."""

    VALIDATION_ERROR = "validation_error"
    EXECUTION_ERROR = "execution_error"
    TIMEOUT = "timeout"
    APPROVAL_REJECTED = "approval_rejected"
    DEPENDENCY_FAILED = "dependency_failed"


class TaskSpec(BaseModel):
    """Specification for a delegated task."""

    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    goal: str = Field(..., min_length=1, description="What the task should accomplish")
    target_agent: str = Field(
        ..., min_length=1, description="Name of the agent to handle the task"
    )
    inputs: Dict[str, Any] = Field(
        default_factory=dict, description="Input data for the task"
    )
    constraints: Dict[str, Any] = Field(
        default_factory=dict, description="Constraints on execution"
    )
    depends_on: List[str] = Field(
        default_factory=list, description="Task IDs this task depends on"
    )
    session_id: str = Field(default="", description="Session context identifier")
    parent_task_id: Optional[str] = Field(
        default=None, description="ID of the parent task if nested"
    )

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TaskSpec:
        return cls.model_validate(data)


class TaskResult(BaseModel):
    """Result of a delegated task execution."""

    task_id: str = Field(..., description="ID of the completed task")
    status: TaskStatus = Field(default=TaskStatus.PENDING)
    output: str = Field(default="", description="Text output from the task")
    artifacts: Dict[str, Any] = Field(
        default_factory=dict, description="Structured artifacts produced"
    )
    errors: List[str] = Field(default_factory=list, description="Error messages if any")
    error_type: Optional[ErrorType] = Field(
        default=None, description="Canonical error classification"
    )
    cost: float = Field(default=0.0, ge=0, description="Estimated cost in USD")
    latency_ms: int = Field(default=0, ge=0, description="Execution time in ms")
    producer_agent: str = Field(
        default="", description="Agent that produced this result"
    )

    @property
    def is_success(self) -> bool:
        return self.status == TaskStatus.COMPLETED

    @property
    def is_terminal(self) -> bool:
        return self.status in {
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.TIMEOUT,
            TaskStatus.APPROVAL_REJECTED,
            TaskStatus.DEPENDENCY_FAILED,
        }

    def to_dict(self) -> Dict[str, Any]:
        data = self.model_dump()
        data["status"] = self.status.value
        if self.error_type:
            data["error_type"] = self.error_type.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TaskResult:
        return cls.model_validate(data)

    @classmethod
    def failure(
        cls,
        task_id: str,
        error: str,
        error_type: ErrorType = ErrorType.EXECUTION_ERROR,
        producer_agent: str = "",
        latency_ms: int = 0,
    ) -> TaskResult:
        """Create a failed TaskResult."""
        return cls(
            task_id=task_id,
            status=TaskStatus.FAILED,
            errors=[error],
            error_type=error_type,
            producer_agent=producer_agent,
            latency_ms=latency_ms,
        )
