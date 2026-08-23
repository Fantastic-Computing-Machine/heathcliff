import base64
from email import message_from_bytes
from email.message import Message
from typing import Any, cast
from unittest.mock import Mock

from config import Config
from core.subagents.email.tools import (
    SafeGmailSearch,
    SignedGmailCreateDraft,
    SignedGmailSendMessage,
)
from utils.outbound_signature import append_outbound_signature, format_outbound_email


def test_plain_signature_uses_master_name_once(monkeypatch):
    monkeypatch.setattr(Config, "MASTER_INFO", {"full_name": "Ada Lovelace"})

    signed = append_outbound_signature("Hello")

    assert signed == (
        "Hello\n\nHeathcliff o.b.o Ada Lovelace\n"
        "This is sent by Heathcliff an Autonomous Intelligence system. It may make mistakes."
    )
    assert append_outbound_signature(signed) == signed


def test_gmail_drafts_and_sends_include_signature(monkeypatch):
    monkeypatch.setattr(Config, "MASTER_INFO", {"name": "Ada Lovelace"})
    api_resource = Mock()

    draft = SignedGmailCreateDraft.model_construct(
        api_resource=api_resource
    )._prepare_draft_message("Hello", ["to@example.com"], "Subject")
    draft_raw = draft["message"]["raw"]
    draft_message = message_from_bytes(base64.urlsafe_b64decode(draft_raw))
    draft_html = str(cast(Any, draft_message).get_payload()[-1].get_payload())
    assert "Heathcliff o.b.o Ada Lovelace" in draft_html

    sent = SignedGmailSendMessage.model_construct(
        api_resource=api_resource
    )._prepare_message("Hello", "to@example.com", "Subject")
    sent_message = message_from_bytes(base64.urlsafe_b64decode(sent["raw"]))
    payload = sent_message.get_payload()
    assert isinstance(payload, list)
    assert isinstance(payload[0], Message)
    html_body = payload[0].get_payload()
    assert isinstance(html_body, str)
    assert "Heathcliff o.b.o Ada Lovelace" in html_body
    assert "This is sent by Heathcliff an Autonomous Intelligence system." in html_body


def test_email_html_renders_headings_bullets_and_safe_text(monkeypatch):
    monkeypatch.setattr(Config, "MASTER_INFO", {"name": "Ada Lovelace"})

    rendered = format_outbound_email(
        "## Korea trip\n\n- **Flights**: Seoul\n- Hotels\n\nVisit https://example.com/?a=1&b=2"
    )

    assert 'class="heathcliff-card"' in rendered
    assert "background:#121d31" in rendered
    assert "<h3>Korea trip</h3>" in rendered
    assert "<ul>" in rendered
    assert "<li><strong>Flights</strong>: Seoul</li>" in rendered
    assert '<a href="https://example.com/?a=1&amp;b=2">' in rendered
    assert "Heathcliff o.b.o Ada Lovelace" in rendered
    assert "This is sent by Heathcliff an Autonomous Intelligence system." in rendered


def test_gmail_search_accepts_full_payload_without_raw():
    encoded = base64.urlsafe_b64encode(b"Hello from Gmail").decode()
    response = {
        "threadId": "thread-1",
        "snippet": "Hello from Gmail",
        "payload": {
            "mimeType": "text/plain",
            "headers": [
                {"name": "Subject", "value": "Greetings"},
                {"name": "From", "value": "sender@example.com"},
            ],
            "body": {"data": encoded},
        },
    }
    api_resource = Mock()
    api_resource.users().messages().get().execute.return_value = response

    results = SafeGmailSearch.model_construct(
        api_resource=api_resource
    )._parse_messages([{"id": "message-1"}])

    assert results[0]["subject"] == "Greetings"
    assert results[0]["body"] == "Hello from Gmail"
    assert api_resource.users().messages().get.call_args.kwargs["format"] == "full"


def test_gmail_search_reads_a_full_html_body_instead_of_a_snippet():
    html = "<html><style>.ignored { color: red; }</style><body><p>Hello Ram,</p><ul><li>First full point</li><li>Second full point</li></ul></body></html>"
    encoded = base64.urlsafe_b64encode(html.encode()).decode()
    response = {
        "threadId": "thread-1",
        "snippet": "Hello Ram, Here is the first point...",
        "payload": {
            "mimeType": "text/html",
            "headers": [],
            "body": {"data": encoded},
        },
    }
    api_resource = Mock()
    api_resource.users().messages().get().execute.return_value = response

    results = SafeGmailSearch.model_construct(
        api_resource=api_resource
    )._parse_messages([{"id": "message-1"}])

    assert results[0]["body"] == "Hello Ram,\n- First full point\n- Second full point"
    assert results[0]["body"] != response["snippet"]
