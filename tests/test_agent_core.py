# ABOUTME: Unit tests for the coordinator-backed HeathcliffAgent.

from unittest.mock import Mock, patch

import pytest


@pytest.fixture(autouse=True)
def reset_singleton():
    from core.agent_core import HeathcliffAgent

    HeathcliffAgent.reset()
    yield
    HeathcliffAgent.reset()


@pytest.fixture
def memory():
    manager = Mock()
    manager.recall.return_value = {
        "documents": [["User likes dark mode"]],
        "metadatas": [[{}]],
        "ids": [["memory-1"]],
        "distances": [[0.1]],
    }
    manager.build_langchain_history.return_value = []
    manager.save_turn.return_value = ("user", "assistant")
    return manager


@pytest.fixture
def agent(memory):
    from core.agent_core import HeathcliffAgent

    with (
        patch("core.agent_core.init_chat_model"),
        patch("core.agent_core.build_default_registry"),
        patch("core.agent_core.build_coordinator_graph", return_value=Mock()),
    ):
        return HeathcliffAgent(memory_manager=memory)


def test_singleton_reuses_instance(memory):
    from core.agent_core import HeathcliffAgent

    with (
        patch("core.agent_core.init_chat_model"),
        patch("core.agent_core.build_default_registry"),
        patch("core.agent_core.build_coordinator_graph"),
    ):
        first = HeathcliffAgent(memory_manager=memory)
        assert HeathcliffAgent() is first
        assert HeathcliffAgent.instance() is first


def test_instance_requires_initialisation():
    from core.agent_core import HeathcliffAgent

    with pytest.raises(RuntimeError, match="not been initialised"):
        HeathcliffAgent.instance()


def test_initialisation_builds_model_and_coordinator(memory):
    from config import Config
    from core.agent_core import HeathcliffAgent

    with (
        patch("core.agent_core.init_chat_model") as model,
        patch("core.agent_core.build_default_registry"),
        patch("core.agent_core.build_coordinator_graph") as graph,
    ):
        agent = HeathcliffAgent(memory_manager=memory)

    assert agent.coordinator is graph.return_value
    assert model.call_args.kwargs["model"] == Config.SUPERVISOR_MODEL


@pytest.mark.parametrize("user_input", ["", "   ", "X" * 10_001])
def test_invoke_rejects_invalid_input(agent, user_input):
    with pytest.raises(ValueError):
        agent.invoke(user_input)


def test_invoke_adds_context_and_saves_turn(agent, memory):
    captured = {}

    def invoke(**kwargs):
        captured.update(kwargs)
        return "Certainly."

    with patch("core.agent_core.invoke_coordinator", side_effect=invoke):
        response = agent.invoke("What do I like?", conversation_id="conversation-1")

    assert response == "Certainly."
    assert captured["session_id"] == "conversation-1"
    prompt = captured["messages"][-1].content
    assert "- User likes dark mode" in prompt
    assert "<USER_QUERY>" in prompt
    memory.save_turn.assert_called_once_with(
        "What do I like?", "Certainly.", "conversation-1"
    )


def test_invoke_returns_error_when_coordinator_fails(agent):
    with patch("core.agent_core.invoke_coordinator", side_effect=RuntimeError("down")):
        assert "encountered an error" in agent.invoke("Hello")


def test_stream_does_not_save_until_approval_resumes(agent, memory):
    approval = {"type": "approval_required", "data": {"session_id": "thread"}}
    with patch("core.agent_core.stream_coordinator", return_value=iter([approval])):
        events = list(agent.stream_invoke("send an email", "thread"))

    assert events == [approval]

    memory.save_turn.assert_not_called()


def test_resume_approval_uses_existing_thread(agent, memory):
    with patch(
        "core.agent_core.resume_coordinator", return_value="Email sent"
    ) as resume:
        assert (
            agent.resume_approval(
                conversation_id="thread",
                user_input="send it",
                approved=True,
                execution_events=[],
            )
            == "Email sent"
        )

    assert resume.call_args.kwargs["session_id"] == "thread"
    memory.save_turn.assert_called_once_with(
        "send it",
        "Email sent",
        "thread",
        execution_events=[
            {
                "type": "approval_resolved",
                "message": "Action approved",
                "data": {"modified_input": ""},
            }
        ],
    )


def test_stream_saves_intermediate_events_with_completed_turn(agent, memory):
    events = iter(
        [
            {"type": "plan", "message": "One task", "data": {"count": 1}},
            {"type": "response", "data": "Certainly."},
            {"type": "complete", "message": "Done", "data": {}},
        ]
    )
    with patch("core.agent_core.stream_coordinator", return_value=events):
        list(agent.stream_invoke("help", "thread"))

    saved_events = memory.save_turn.call_args.kwargs["execution_events"]
    assert [event["type"] for event in saved_events] == [
        "run_started",
        "plan",
        "complete",
    ]
