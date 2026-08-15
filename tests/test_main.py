# ABOUTME: Tests for Heathcliff's command-line mode selection
# ABOUTME: Keeps text mode independent of optional audio hardware

import sys
from unittest.mock import Mock

import main


def test_main_defaults_to_text_mode(monkeypatch):
    calls = []

    class FakeAssistant:
        def __init__(self, enable_audio):
            calls.append(("init", enable_audio))

        def run_text_mode(self):
            calls.append(("text",))

        def run_voice_mode(self):
            calls.append(("voice",))

    monkeypatch.setattr(main, "HeathcliffAssistant", FakeAssistant)
    monkeypatch.setattr(sys, "argv", ["main.py"])

    main.main()

    assert calls == [("init", False), ("text",)]


def test_text_mode_resumes_pending_approval_on_sure():
    assistant = object.__new__(main.HeathcliffAssistant)
    assistant.conversation_id = "thread"
    assistant.pending_approval = None
    assistant.agent = Mock()
    assistant.agent.stream_invoke.return_value = iter(
        [
            {
                "type": "approval_required",
                "data": {
                    "session_id": "thread",
                    "tool_name": "calendar_agent_tool",
                },
            }
        ]
    )
    assistant.agent.resume_approval.return_value = "Appointment created."

    prompt = assistant._process_text_input("Create a barber appointment")
    response = assistant._process_text_input("sure")

    assert "Type 'approve'" in prompt
    assert response == "Appointment created."
    assistant.agent.resume_approval.assert_called_once_with(
        conversation_id="thread",
        user_input="Create a barber appointment",
        approved=True,
    )
