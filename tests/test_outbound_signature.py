import base64
from email import message_from_bytes
from email.message import Message
from unittest.mock import Mock

from config import Config
from core.subagents.email.tools import (
    SafeGmailSearch,
    SignedGmailCreateDraft,
    SignedGmailSendMessage,
)
from utils.outbound_signature import append_outbound_signature


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
    assert "Heathcliff o.b.o Ada Lovelace" in draft_message.get_payload()

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
