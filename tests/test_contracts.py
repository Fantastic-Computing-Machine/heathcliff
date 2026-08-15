# ABOUTME: Unit tests for delegation task contracts.
# ABOUTME: Validates Pydantic models, serialization, error taxonomy, and factory methods

import pytest

from core.delegation.contracts import (
    ErrorType,
    TaskResult,
    TaskSpec,
    TaskStatus,
)


class TestTaskSpec:
    def test_required_fields(self):
        spec = TaskSpec(goal="Search weather", target_agent="info_agent_tool")
        assert spec.goal == "Search weather"
        assert spec.target_agent == "info_agent_tool"
        assert spec.task_id  # auto-generated UUID

    def test_optional_fields_default(self):
        spec = TaskSpec(goal="test", target_agent="agent")
        assert spec.inputs == {}
        assert spec.constraints == {}
        assert spec.depends_on == []
        assert spec.session_id == ""
        assert spec.parent_task_id is None

    def test_serialization_roundtrip(self):
        spec = TaskSpec(
            goal="find weather",
            target_agent="info_agent_tool",
            inputs={"city": "NYC"},
            depends_on=["task-1"],
            session_id="sess-1",
        )
        data = spec.to_dict()
        restored = TaskSpec.from_dict(data)
        assert restored.goal == spec.goal
        assert restored.target_agent == spec.target_agent
        assert restored.inputs == spec.inputs
        assert restored.depends_on == spec.depends_on

    def test_empty_goal_raises(self):
        with pytest.raises(Exception):
            TaskSpec(goal="", target_agent="agent")

    def test_empty_target_agent_raises(self):
        with pytest.raises(Exception):
            TaskSpec(goal="test", target_agent="")


class TestTaskResult:
    def test_default_status(self):
        result = TaskResult(task_id="t1")
        assert result.status == TaskStatus.PENDING
        assert not result.is_success
        assert not result.is_terminal

    def test_success_result(self):
        result = TaskResult(
            task_id="t1",
            status=TaskStatus.COMPLETED,
            output="Weather is sunny",
            producer_agent="info_agent_tool",
        )
        assert result.is_success
        assert result.is_terminal
        assert result.output == "Weather is sunny"

    def test_failure_factory(self):
        result = TaskResult.failure(
            task_id="t1",
            error="API timeout",
            error_type=ErrorType.TIMEOUT,
            producer_agent="info",
        )
        assert result.status == TaskStatus.FAILED
        assert result.is_terminal
        assert not result.is_success
        assert "API timeout" in result.errors
        assert result.error_type == ErrorType.TIMEOUT

    def test_serialization_roundtrip(self):
        result = TaskResult(
            task_id="t1",
            status=TaskStatus.COMPLETED,
            output="hello",
            cost=0.01,
            latency_ms=150,
            producer_agent="agent",
        )
        data = result.to_dict()
        restored = TaskResult.from_dict(data)
        assert restored.status == TaskStatus.COMPLETED
        assert restored.output == "hello"

    def test_terminal_statuses(self):
        for status in [
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.TIMEOUT,
            TaskStatus.APPROVAL_REJECTED,
            TaskStatus.DEPENDENCY_FAILED,
        ]:
            r = TaskResult(task_id="t", status=status)
            assert r.is_terminal

    def test_non_terminal_statuses(self):
        for status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
            r = TaskResult(task_id="t", status=status)
            assert not r.is_terminal


class TestErrorType:
    def test_all_variants(self):
        assert len(ErrorType) == 5
        assert ErrorType.VALIDATION_ERROR.value == "validation_error"
        assert ErrorType.EXECUTION_ERROR.value == "execution_error"
        assert ErrorType.TIMEOUT.value == "timeout"
        assert ErrorType.APPROVAL_REJECTED.value == "approval_rejected"
        assert ErrorType.DEPENDENCY_FAILED.value == "dependency_failed"
