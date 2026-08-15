# ABOUTME: Coordinator graph stability tests for planning, execution, timeouts, approvals, and streaming

import json
import os
import sys
import time
from unittest.mock import Mock

import pytest
from langchain_community.callbacks.human import HumanRejectedException

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from core.delegation.contracts import ErrorType, TaskResult, TaskSpec, TaskStatus
from core.delegation.registry import AgentDescriptor, CapabilityRegistry


def _make_llm(plan, aggregate_response="Aggregated response"):
    llm = Mock()

    def _invoke(messages):
        content = "\n".join(str(getattr(m, "content", "")) for m in messages)
        result = Mock()
        if "Subtask results:" in content:
            result.content = aggregate_response
        else:
            result.content = json.dumps(plan)
        return result

    llm.invoke = _invoke
    return llm


@pytest.fixture
def registry():
    reg = CapabilityRegistry()

    def _static(response):
        def fn(request="", **kwargs):
            return response

        return fn

    reg.register(
        AgentDescriptor(
            name="info_agent_tool",
            capabilities=["search", "weather"],
            invoke_fn=_static("Weather is sunny"),
        )
    )
    reg.register(
        AgentDescriptor(
            name="calendar_agent_tool",
            capabilities=["calendar", "event"],
            invoke_fn=_static("Meeting at 3 PM"),
        )
    )
    reg.register(
        AgentDescriptor(
            name="email_agent_tool",
            capabilities=["email", "send"],
            invoke_fn=_static("Draft prepared"),
        )
    )
    return reg


def _base_state():
    return {
        "messages": [],
        "task_specs": [],
        "task_results": [],
        "final_response": "",
        "error": None,
        "callbacks": [],
        "coordinator_started_at": time.monotonic(),
    }


class TestCoordinatorStability:
    def test_dependency_chain_invoke_no_crash(self, registry):
        from core.coordinator_graph import build_coordinator_graph, invoke_coordinator

        plan = [
            {
                "goal": "Get weather",
                "target_agent": "info_agent_tool",
                "depends_on": [],
                "parallelizable": False,
            },
            {
                "goal": "Use weather for scheduling",
                "target_agent": "calendar_agent_tool",
                "depends_on": [0],
                "parallelizable": False,
            },
            {
                "goal": "Draft confirmation",
                "target_agent": "email_agent_tool",
                "depends_on": [1],
                "parallelizable": False,
            },
        ]
        llm = _make_llm(plan, aggregate_response="All tasks completed.")
        graph = build_coordinator_graph(registry, llm)

        result = invoke_coordinator(graph, "Plan my afternoon", "sess-chain")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_failed_specialist_output_blocks_dependent_action(self, registry):
        from core.coordinator_graph import build_coordinator_graph

        email_called = False

        def failed_research(request="", **kwargs):
            return "Research failed: Gemini quota exhausted"

        def downstream(request="", **kwargs):
            nonlocal email_called
            email_called = True
            return "Downstream action completed"

        registry.register(
            AgentDescriptor(
                name="info_agent_tool",
                capabilities=["research"],
                invoke_fn=failed_research,
            )
        )
        registry.register(
            AgentDescriptor(
                name="downstream_agent",
                capabilities=["downstream"],
                invoke_fn=downstream,
            )
        )
        plan = [
            {
                "goal": "Research Korea",
                "target_agent": "info_agent_tool",
                "depends_on": [],
                "parallelizable": False,
            },
            {
                "goal": "Use the research context",
                "target_agent": "downstream_agent",
                "depends_on": [0],
                "parallelizable": False,
            },
        ]
        graph = build_coordinator_graph(registry, _make_llm(plan))
        state = _base_state()
        state.update({"user_input": "research then email", "session_id": "sess-failed"})

        result = graph.invoke(state)
        task_results = [TaskResult.from_dict(d) for d in result["task_results"]]

        assert task_results[0].status == TaskStatus.FAILED
        assert task_results[1].status == TaskStatus.DEPENDENCY_FAILED
        assert not email_called

    def test_invalid_dependency_refs_and_cycle_mark_dependency_failed(self, registry):
        from core.coordinator_graph import build_coordinator_graph

        plan = [
            {
                "goal": "invalid out-of-range dep",
                "target_agent": "info_agent_tool",
                "depends_on": [99],
                "parallelizable": False,
            },
            {
                "goal": "self dependency",
                "target_agent": "info_agent_tool",
                "depends_on": [1],
                "parallelizable": False,
            },
            {
                "goal": "forward dep, cycle part 1",
                "target_agent": "info_agent_tool",
                "depends_on": [3],
                "parallelizable": False,
            },
            {
                "goal": "cycle part 2",
                "target_agent": "info_agent_tool",
                "depends_on": [2],
                "parallelizable": False,
            },
            {
                "goal": "independent valid",
                "target_agent": "info_agent_tool",
                "depends_on": [],
                "parallelizable": False,
            },
        ]
        llm = _make_llm(plan)
        graph = build_coordinator_graph(registry, llm)
        state = _base_state()
        state.update({"user_input": "run plan", "session_id": "sess-invalid"})

        result = graph.invoke(state)
        task_results = [TaskResult.from_dict(d) for d in result["task_results"]]

        assert task_results[0].status == TaskStatus.DEPENDENCY_FAILED
        assert task_results[1].status == TaskStatus.DEPENDENCY_FAILED
        assert task_results[2].status == TaskStatus.DEPENDENCY_FAILED
        assert task_results[3].status == TaskStatus.DEPENDENCY_FAILED
        assert task_results[4].status == TaskStatus.COMPLETED

    def test_unknown_target_agent_maps_to_validation_failure(self, registry):
        from core.coordinator_graph import build_coordinator_graph

        plan = [
            {
                "goal": "unknown route",
                "target_agent": "not_registered_agent",
                "depends_on": [],
                "parallelizable": False,
            },
            {
                "goal": "known route",
                "target_agent": "info_agent_tool",
                "depends_on": [],
                "parallelizable": False,
            },
        ]
        llm = _make_llm(plan)
        graph = build_coordinator_graph(registry, llm)
        state = _base_state()
        state.update({"user_input": "route test", "session_id": "sess-unknown"})

        result = graph.invoke(state)
        task_results = [TaskResult.from_dict(d) for d in result["task_results"]]
        assert task_results[0].status == TaskStatus.FAILED
        assert task_results[0].error_type == ErrorType.VALIDATION_ERROR
        assert task_results[1].status == TaskStatus.COMPLETED

    def test_per_task_timeout_marks_timeout_and_continues(self, registry, monkeypatch):
        from core.coordinator_graph import build_coordinator_graph

        def slow(request="", **kwargs):
            time.sleep(0.08)
            return "slow done"

        registry.register(
            AgentDescriptor(
                name="slow_agent",
                capabilities=["slow"],
                invoke_fn=slow,
            )
        )

        monkeypatch.setattr(Config, "PER_TASK_TIMEOUT_MS", 20)
        monkeypatch.setattr(Config, "MAX_TOTAL_RUNTIME_MS", 1000)

        plan = [
            {
                "goal": "run slow",
                "target_agent": "slow_agent",
                "depends_on": [],
                "parallelizable": False,
            },
            {
                "goal": "run fast",
                "target_agent": "info_agent_tool",
                "depends_on": [],
                "parallelizable": False,
            },
        ]
        llm = _make_llm(plan)
        graph = build_coordinator_graph(registry, llm)
        state = _base_state()
        state.update({"user_input": "timeouts", "session_id": "sess-timeout"})

        result = graph.invoke(state)
        task_results = [TaskResult.from_dict(d) for d in result["task_results"]]

        assert task_results[0].status == TaskStatus.TIMEOUT
        assert task_results[1].status == TaskStatus.COMPLETED

    def test_sensitive_task_does_not_outlive_its_timeout(self, registry):
        from core.coordinator_graph import _execute_single_task

        def slow_send(request="", **kwargs):
            time.sleep(0.03)
            return "Email sent"

        registry.register(
            AgentDescriptor(
                name="email_agent_tool", capabilities=["email"], invoke_fn=slow_send
            )
        )
        result = _execute_single_task(
            TaskSpec(goal="Send an email", target_agent="email_agent_tool"),
            registry,
            callbacks=(),
            timeout_ms=1,
        )

        assert result.status == TaskStatus.COMPLETED

    def test_total_runtime_cutoff_stops_scheduling_new_tasks(
        self, registry, monkeypatch
    ):
        from core.coordinator_graph import build_coordinator_graph

        def medium(request="", **kwargs):
            time.sleep(0.05)
            return "done"

        registry.register(
            AgentDescriptor(
                name="medium_agent", capabilities=["medium"], invoke_fn=medium
            )
        )
        monkeypatch.setattr(Config, "PER_TASK_TIMEOUT_MS", 100)
        monkeypatch.setattr(Config, "MAX_TOTAL_RUNTIME_MS", 10)

        plan = [
            {
                "goal": "first medium",
                "target_agent": "medium_agent",
                "depends_on": [],
                "parallelizable": False,
            },
            {
                "goal": "second fast",
                "target_agent": "info_agent_tool",
                "depends_on": [],
                "parallelizable": False,
            },
        ]
        llm = _make_llm(plan)
        graph = build_coordinator_graph(registry, llm)
        state = _base_state()
        state.update({"user_input": "runtime cutoff", "session_id": "sess-cutoff"})

        result = graph.invoke(state)
        assert len(result["task_results"]) == 1
        first = TaskResult.from_dict(result["task_results"][0])
        assert first.status in {TaskStatus.TIMEOUT, TaskStatus.COMPLETED}

    def test_callback_approval_rejection_maps_to_approval_rejected(self, registry):
        from core.coordinator_graph import CoordinatorContext, build_coordinator_graph

        class RejectingCallback:
            def on_tool_start(self, *args, **kwargs):
                raise HumanRejectedException("Denied by user")

        plan = [
            {
                "goal": "approval protected call",
                "target_agent": "info_agent_tool",
                "depends_on": [],
                "parallelizable": False,
            }
        ]
        llm = _make_llm(plan)
        graph = build_coordinator_graph(registry, llm)
        session_id = "sess-approval"
        state = _base_state()
        state.update({"user_input": "approval test", "session_id": session_id})

        result = graph.invoke(
            state,
            {"configurable": {"thread_id": session_id}},
            context=CoordinatorContext(callbacks=[RejectingCallback()]),
        )
        task_result = TaskResult.from_dict(result["task_results"][0])
        assert task_result.status == TaskStatus.APPROVAL_REJECTED
        assert task_result.error_type == ErrorType.APPROVAL_REJECTED

    def test_planner_validation_repair_path(self, registry):
        from core.coordinator_graph import build_coordinator_graph, invoke_coordinator

        llm = Mock()
        calls = {"count": 0}

        def _invoke(messages):
            calls["count"] += 1
            result = Mock()
            if calls["count"] == 1:
                result.content = json.dumps(
                    [
                        {
                            "goal": "bad object",
                            "target_agent": "info_agent_tool",
                            "depends_on": [],
                            "parallelizable": False,
                            "unexpected_field": "boom",
                        }
                    ]
                )
                return result
            if calls["count"] == 2:
                result.content = json.dumps(
                    [
                        {
                            "goal": "repaired plan",
                            "target_agent": "info_agent_tool",
                            "depends_on": [],
                            "parallelizable": False,
                        }
                    ]
                )
                return result
            result.content = "Repaired execution complete."
            return result

        llm.invoke = _invoke
        graph = build_coordinator_graph(registry, llm)
        response = invoke_coordinator(graph, "repair test", "sess-repair")

        assert isinstance(response, str)
        assert calls["count"] >= 2

    def test_stream_complete_uses_deduped_first_seen_agents(self, registry):
        from core.coordinator_graph import build_coordinator_graph, stream_coordinator

        registry.register(
            AgentDescriptor(
                name="secondary_agent",
                capabilities=["secondary"],
                invoke_fn=lambda request="", **_: "Secondary completed",
            )
        )

        plan = [
            {
                "goal": "first info",
                "target_agent": "info_agent_tool",
                "depends_on": [],
                "parallelizable": False,
            },
            {
                "goal": "second info",
                "target_agent": "info_agent_tool",
                "depends_on": [],
                "parallelizable": False,
            },
            {
                "goal": "secondary task",
                "target_agent": "secondary_agent",
                "depends_on": [],
                "parallelizable": False,
            },
        ]
        llm = _make_llm(plan)
        graph = build_coordinator_graph(registry, llm)

        events = list(stream_coordinator(graph, "stream test", "sess-stream"))
        complete = [e for e in events if e.get("type") == "complete"][-1]

        assert complete["data"]["agents_used"] == [
            "info_agent_tool",
            "secondary_agent",
        ]
        assert complete["data"]["agent_count"] == 2
