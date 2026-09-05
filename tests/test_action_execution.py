# ABOUTME: Regression tests for multi-turn action chains and external side effects
# ABOUTME: Covers planner history, dependency context, and awaited Telegram sends

import json
from unittest.mock import AsyncMock, Mock

from langchain_core.messages import AIMessage, HumanMessage

from config import Config
from core.delegation.registry import AgentDescriptor, CapabilityRegistry


def _make_llm(plan):
    llm = Mock()

    def _invoke(messages):
        text = "\n".join(str(getattr(message, "content", "")) for message in messages)
        result = Mock()
        result.content = (
            "Action chain completed."
            if "Subtask results:" in text
            else json.dumps(plan)
        )
        return result

    llm.invoke = _invoke
    return llm


def test_planner_receives_history_for_clarification_follow_up():
    from core.coordinator_graph import build_coordinator_graph, invoke_coordinator

    registry = CapabilityRegistry()
    registry.register(
        AgentDescriptor(
            name="email_agent_tool",
            capabilities=["email"],
            invoke_fn=lambda request="", **_: "Email sent.",
        )
    )
    captured_messages = []
    llm = Mock()

    def _invoke(messages):
        captured_messages.append(messages)
        result = Mock()
        result.content = json.dumps(
            [
                {
                    "goal": "Send the saved research to philip@example.com",
                    "target_agent": "email_agent_tool",
                    "depends_on": [],
                    "parallelizable": False,
                }
            ]
        )
        return result

    llm.invoke = _invoke
    history = [
        HumanMessage(content="Research sea levels and email Philip a summary."),
        AIMessage(content="I couldn't find Philip. What is his email address?"),
        HumanMessage(content="philip@example.com"),
    ]

    invoke_coordinator(
        build_coordinator_graph(registry, llm),
        "philip@example.com",
        "follow-up",
        messages=history,
    )

    planner_text = "\n".join(
        str(getattr(message, "content", "")) for message in captured_messages[0]
    )
    assert "Research sea levels and email Philip" in planner_text
    assert "What is his email address?" in planner_text
    assert "philip@example.com" in planner_text


def test_dependency_outputs_reach_research_contact_email_task():
    from core.coordinator_graph import (
        build_coordinator_graph,
        invoke_coordinator,
    )

    registry = CapabilityRegistry()
    email_requests = []
    registry.register(
        AgentDescriptor(
            name="info_agent_tool",
            capabilities=["research"],
            invoke_fn=lambda request="", **_: "Sea levels are rising 3.7mm/year.",
        )
    )
    registry.register(
        AgentDescriptor(
            name="contacts_agent_tool",
            capabilities=["contact"],
            invoke_fn=lambda request="", **_: "Philip: philip@example.com",
        )
    )

    def send_email(request="", **_):
        email_requests.append(request)
        return "Email sent to philip@example.com."

    registry.register(
        AgentDescriptor(
            name="email_agent_tool",
            capabilities=["email"],
            invoke_fn=send_email,
        )
    )
    plan = [
        {
            "goal": "Research current sea-level rise",
            "target_agent": "info_agent_tool",
            "depends_on": [],
            "parallelizable": True,
        },
        {
            "goal": "Find Philip's email address",
            "target_agent": "contacts_agent_tool",
            "depends_on": [],
            "parallelizable": True,
        },
        {
            "goal": "Email Philip a concise research summary",
            "target_agent": "email_agent_tool",
            "depends_on": [0, 1],
            "parallelizable": False,
        },
    ]

    graph = build_coordinator_graph(registry, _make_llm(plan))
    response = invoke_coordinator(
        graph,
        "Research sea levels and email Philip a summary",
        "research-email",
    )
    assert response == "Action chain completed."

    assert len(email_requests) == 1
    assert "3.7mm/year" in email_requests[0]
    assert "philip@example.com" in email_requests[0]


def test_telegram_send_waits_for_api(monkeypatch):
    import core.subagents.comms.tools as comms_tools

    bot = Mock()
    bot.send_message = AsyncMock()
    monkeypatch.setattr(comms_tools, "_get_telegram_bot", lambda: bot)
    monkeypatch.setattr(Config, "TELEGRAM_CHAT_ID", "123")

    result = comms_tools.send_to_telegram.invoke({"message": "Build passed"})

    call = bot.send_message.await_args
    assert call is not None
    sent_message = call.kwargs["text"]
    assert sent_message.startswith("Build passed\n\nHeathcliff o.b.o ")
    assert (
        "This is sent by Heathcliff an Autonomous Intelligence system." in sent_message
    )
    assert "successfully" in result.lower()
