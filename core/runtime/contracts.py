"""Provider-neutral Runtime V2 contracts."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Literal, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class TurnStatus(str, Enum):
    ADMITTED = "admitted"
    RUNNING = "running"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class EventKind(str, Enum):
    INPUT_ADMITTED = "input.admitted"
    TURN_STARTED = "turn.started"
    MODEL_STARTED = "model.started"
    MODEL_COMPLETED = "model.completed"
    TOOL_PROPOSED = "tool.proposed"
    TOOL_COMPLETED = "tool.completed"
    TOOL_OUTCOME_UNKNOWN = "tool.outcome_unknown"
    APPROVAL_REQUIRED = "approval.required"
    APPROVAL_DECIDED = "approval.decided"
    CONTEXT_COMPACTED = "context.compacted"
    TURN_COMPLETED = "turn.completed"
    TURN_CANCELLED = "turn.cancelled"
    TURN_FAILED = "turn.failed"


class ToolEffect(str, Enum):
    READ = "read"
    REVERSIBLE_WRITE = "reversible_write"
    EXTERNAL_SIDE_EFFECT = "external_side_effect"


class ParallelSafety(str, Enum):
    SAFE_READ = "safe_read"
    EXCLUSIVE = "exclusive"


class RetrySafety(str, Enum):
    SAFE = "safe"
    NEVER_AFTER_UNCERTAIN_OUTCOME = "never_after_uncertain_outcome"


class ToolOutcome(str, Enum):
    NOT_STARTED = "not_started"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    OUTCOME_UNKNOWN = "outcome_unknown"


class ApprovalPolicy(str, Enum):
    NEVER = "never"
    ALWAYS = "always"
    EXTERNAL_SIDE_EFFECTS = "external_side_effects"


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


class PrivacyClass(str, Enum):
    NORMAL = "normal"
    SENSITIVE = "sensitive"
    SECRET = "secret"


class ResourceScope(BaseModel):
    account: str = "default"
    resource: str
    privacy: PrivacyClass = PrivacyClass.NORMAL


class Thread(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=utcnow)
    next_event_seq: int = 1


class Turn(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    thread_id: UUID
    input_id: UUID
    status: TurnStatus = TurnStatus.ADMITTED
    created_at: datetime = Field(default_factory=utcnow)
    completed_at: Optional[datetime] = None
    was_created: bool = False


class RuntimeItem(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    thread_id: UUID
    turn_id: UUID
    kind: str
    content: Any
    provider_state: dict[str, Any] = Field(default_factory=dict)


class RuntimeEvent(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    thread_id: UUID
    turn_id: Optional[UUID] = None
    sequence: int = 0
    kind: EventKind
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)


class PendingInput(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    thread_id: UUID
    content: str = Field(min_length=1, max_length=10000)
    idempotency_key: str = Field(min_length=1, max_length=200)
    admitted_at: datetime = Field(default_factory=utcnow)
    was_admitted: bool = False


class ProviderCapabilities(BaseModel):
    provider: str
    model: str
    native_tools: bool = True
    structured_output: bool = True
    thought_signatures: bool = False
    multimodal: bool = False
    context_window: int = Field(default=32768, gt=0)


class ModelRequest(BaseModel):
    thread_id: UUID
    turn_id: UUID
    system_instruction: str
    context: list[RuntimeItem] = Field(default_factory=list)
    tools: list[dict[str, Any]] = Field(default_factory=list)
    provider: ProviderCapabilities


class PreparedModelCall(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    request: ModelRequest
    effective_config: dict[str, Any] = Field(default_factory=dict)


class ModelEvent(BaseModel):
    kind: Literal["text_delta", "tool_call", "usage", "completed", "safety"]
    data: dict[str, Any] = Field(default_factory=dict)


class ToolContract(BaseModel):
    name: str = Field(pattern=r"^[a-zA-Z0-9_:.]+$")
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any] = Field(default_factory=dict)
    effect: ToolEffect = ToolEffect.READ
    resource_scope: ResourceScope
    approval_policy: ApprovalPolicy = ApprovalPolicy.EXTERNAL_SIDE_EFFECTS
    parallel_safety: ParallelSafety = ParallelSafety.SAFE_READ
    timeout_seconds: int = Field(default=60, ge=1, le=600)
    retry_safety: RetrySafety = RetrySafety.SAFE
    verification_tool: Optional[str] = None
    trace_privacy: PrivacyClass = PrivacyClass.NORMAL
    schema_revision: str = "v1"


class ToolCall(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str = Field(default_factory=lambda: str(uuid4()))


class ToolResult(BaseModel):
    call_id: UUID
    outcome: ToolOutcome
    output: Any = None
    error: Optional[str] = None
    verification: Any = None


class ApprovalRequest(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    thread_id: UUID
    turn_id: UUID
    tool_call: ToolCall
    resource_scope: ResourceScope
    expires_at: datetime = Field(
        default_factory=lambda: utcnow() + timedelta(minutes=10)
    )
    status: ApprovalStatus = ApprovalStatus.PENDING


class ApprovalDecision(BaseModel):
    approval_id: UUID
    approved: bool
    decided_at: datetime = Field(default_factory=utcnow)


class ArtifactRef(BaseModel):
    uri: str
    content_hash: str
    content_type: str
    size_bytes: int = Field(ge=0)


class ContextCheckpoint(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    thread_id: UUID
    through_sequence: int
    active_goals: list[str] = Field(default_factory=list)
    commitments: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    corrections: list[str] = Field(default_factory=list)
    entities: dict[str, Any] = Field(default_factory=dict)
    verified_state: dict[str, Any] = Field(default_factory=dict)
    unknown_outcomes: list[str] = Field(default_factory=list)
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    provider_state: dict[str, Any] = Field(default_factory=dict)


class MemoryCandidate(BaseModel):
    kind: Literal[
        "preference",
        "stable_fact",
        "relationship",
        "episodic_event",
        "commitment",
        "correction",
        "non_memory",
    ]
    subject: str
    content: str
    confidence: float = Field(ge=0, le=1)
    source_event_id: UUID
    source_kind: Literal["user", "verified_provider", "assistant_inference"]
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    contains_secret: bool = False


class PersonalMemory(MemoryCandidate):
    id: UUID = Field(default_factory=uuid4)
    supersedes_id: Optional[UUID] = None
    embedding: Optional[list[float]] = None
    created_at: datetime = Field(default_factory=utcnow)


class MemoryJob(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    source_event_id: UUID
    attempts: int = 0


class RuntimeLease(BaseModel):
    name: str = "heathcliff-runtime"
    holder: str
    expires_at: datetime
