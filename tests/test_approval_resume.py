# ABOUTME: Regression tests for durable coordinator approval pause and resume
# ABOUTME: Covers exact-once execution, rejection, agent resume, and UI wiring

import ast
import json
from pathlib import Path
from unittest.mock import Mock, patch

from core.coordinator_graph import (
    build_coordinator_graph,
    resume_coordinator,
    stream_coordinator,
)
from core.delegation.contracts import ErrorType, TaskResult, TaskStatus
from core.delegation.registry import AgentDescriptor, CapabilityRegistry


def _make_llm(plan):
    llm = Mock()

    def invoke(messages):
        content = "\n".join(
            str(getattr(message, "content", "")) for message in messages
        )
        result = Mock()
        result.content = (
            "Research complete\n\n---\n\nEmail sent"
            if "Subtask results:" in content
            else json.dumps(plan)
        )
        return result

    llm.invoke = invoke
    return llm


def _register(registry, name, calls, response):
    def invoke(request="", **kwargs):
        calls.append(request)
        return response

    registry.register(AgentDescriptor(name=name, capabilities=[name], invoke_fn=invoke))


def test_sensitive_action_pauses_then_approval_executes_all_tasks_once():
    calls = {"info": [], "email": []}
    registry = CapabilityRegistry()
    _register(registry, "info_agent_tool", calls["info"], "Research complete")
    _register(registry, "email_agent_tool", calls["email"], "Email sent")
    plan = [
        {
            "goal": "research the topic",
            "target_agent": "info_agent_tool",
            "depends_on": [],
            "parallelizable": False,
        },
        {
            "goal": "send email with the result",
            "target_agent": "email_agent_tool",
            "depends_on": [0],
            "parallelizable": False,
        },
    ]
    graph = build_coordinator_graph(registry, _make_llm(plan))
    session_id = "approval-exactly-once"

    events = list(stream_coordinator(graph, "research and email", session_id))

    approval = next(event for event in events if event["type"] == "approval_required")
    assert approval["data"]["session_id"] == session_id
    assert approval["data"]["actions"][0]["tool_name"] == "email_agent_tool"
    assert not any(event["type"] == "complete" for event in events)
    assert calls == {"info": [], "email": []}

    response = resume_coordinator(graph, session_id, approved=True)

    assert response == "Research complete\n\n---\n\nEmail sent"
    assert len(calls["info"]) == 1
    assert len(calls["email"]) == 1


def test_rejection_resumes_without_sensitive_execution_and_maps_status():
    calls = []
    registry = CapabilityRegistry()
    _register(registry, "email_agent_tool", calls, "Email sent")
    plan = [
        {
            "goal": "send email to Alex",
            "target_agent": "email_agent_tool",
            "depends_on": [],
            "parallelizable": False,
        }
    ]
    graph = build_coordinator_graph(registry, _make_llm(plan))
    session_id = "approval-rejected"

    events = list(stream_coordinator(graph, "send email", session_id))
    assert any(event["type"] == "approval_required" for event in events)

    response = resume_coordinator(graph, session_id, approved=False)
    state = graph.get_state({"configurable": {"thread_id": session_id}}).values
    result = TaskResult.from_dict(state["task_results"][0])

    assert calls == []
    assert result.status == TaskStatus.APPROVAL_REJECTED
    assert result.error_type == ErrorType.APPROVAL_REJECTED
    assert "Approval rejected by user" in response


def test_agent_resume_uses_saved_coordinator_thread_without_new_prompt():
    from core.agent_core import HeathcliffAgent

    memory = Mock()
    memory.save_turn = Mock()
    HeathcliffAgent.reset()
    with (
        patch("core.agent_core.init_chat_model"),
        patch("core.agent_core.build_coordinator_graph", return_value=Mock()),
        patch("core.agent_core.build_default_registry"),
    ):
        agent = HeathcliffAgent(memory_manager=memory)

    with (
        patch("core.agent_core.invoke_coordinator") as initial_invoke,
        patch(
            "core.agent_core.resume_coordinator", return_value="Email sent"
        ) as resume,
        patch.object(agent, "_build_callbacks", return_value=None),
    ):
        response = agent.resume_approval(
            conversation_id="saved-thread",
            user_input="send the email",
            approved=True,
        )

    assert response == "Email sent"
    initial_invoke.assert_not_called()
    resume.assert_called_once_with(
        compiled_graph=agent.coordinator,
        session_id="saved-thread",
        approved=True,
        modified_input=None,
        callbacks=None,
    )
    memory.save_turn.assert_called_once_with(
        "send the email", "Email sent", "saved-thread"
    )
    HeathcliffAgent.reset()


def test_agent_stream_stops_at_approval_without_saving_a_completed_turn():
    from core.agent_core import HeathcliffAgent

    memory = Mock()
    memory.build_langchain_history.return_value = []
    memory.recall.return_value = {}
    memory.save_turn = Mock()
    HeathcliffAgent.reset()
    with (
        patch("core.agent_core.init_chat_model"),
        patch("core.agent_core.build_coordinator_graph", return_value=Mock()),
        patch("core.agent_core.build_default_registry"),
    ):
        agent = HeathcliffAgent(memory_manager=memory)

    approval_event = {
        "type": "approval_required",
        "message": "Approval required",
        "data": {"session_id": "saved-thread", "actions": []},
    }
    with (
        patch(
            "core.agent_core.stream_coordinator", return_value=iter([approval_event])
        ),
        patch.object(agent, "_build_callbacks", return_value=None),
    ):
        events = list(agent.stream_invoke("send email", "saved-thread"))

    assert events == [approval_event]
    memory.save_turn.assert_not_called()
    HeathcliffAgent.reset()


def test_streamlit_approve_and_reject_buttons_call_resume_api():
    source = Path("ui/views/command_center.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    approval_values = {
        keyword.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_resume_approval"
        for keyword in node.keywords
        if keyword.arg == "approved" and isinstance(keyword.value, ast.Constant)
    }

    assert approval_values == {True, False}
    assert "additional_callbacks=[approval_handler]" not in source
