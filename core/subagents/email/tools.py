# ABOUTME: Gmail integration via LangChain Gmail toolkit
# ABOUTME: Provides search, read, draft tools using Google OAuth credentials

from base64 import urlsafe_b64decode, urlsafe_b64encode
from email.message import EmailMessage
from typing import Any, Dict, List, cast

from googleapiclient.discovery import build
from langchain_community.agent_toolkits import GmailToolkit
from langchain_community.tools.gmail.create_draft import GmailCreateDraft
from langchain_community.tools.gmail.search import GmailSearch, clean_email_body
from langchain_community.tools.gmail.send_message import GmailSendMessage

from utils.google_auth import GMAIL_SCOPES, get_google_credentials
from utils.outbound_signature import append_outbound_signature, format_outbound_email


def _get_gmail_service():
    """Get authenticated Gmail API service."""
    creds = get_google_credentials(GMAIL_SCOPES)
    return build("gmail", "v1", credentials=creds)


_gmail_api_resource = None


class SignedGmailCreateDraft(GmailCreateDraft):
    """Create drafts with Heathcliff's mandatory sender disclosure."""

    def _prepare_draft_message(self, message, to, subject, cc=None, bcc=None):
        draft_message = EmailMessage()
        draft_message.set_content(append_outbound_signature(message))
        draft_message.add_alternative(format_outbound_email(message), subtype="html")
        draft_message["To"] = ", ".join(to)
        draft_message["Subject"] = subject
        if cc is not None:
            draft_message["Cc"] = ", ".join(cc)
        if bcc is not None:
            draft_message["Bcc"] = ", ".join(bcc)
        return {
            "message": {"raw": urlsafe_b64encode(draft_message.as_bytes()).decode()}
        }


class SignedGmailSendMessage(GmailSendMessage):
    """Send mail with Heathcliff's mandatory sender disclosure."""

    def _prepare_message(self, message, to, subject, cc=None, bcc=None):
        return super()._prepare_message(
            format_outbound_email(message), to, subject, cc, bcc
        )


class SafeGmailSearch(GmailSearch):
    """Read Gmail's documented full payload instead of requiring ``raw``."""

    @staticmethod
    def _plain_text(payload: Dict[str, Any]) -> str:
        body = payload.get("body", {})
        encoded = body.get("data", "")
        if payload.get("mimeType", "").lower() == "text/plain" and encoded:
            padding = "=" * (-len(encoded) % 4)
            return urlsafe_b64decode(encoded + padding).decode("utf-8", "replace")
        for part in payload.get("parts", []):
            text = SafeGmailSearch._plain_text(part)
            if text:
                return text
        return ""

    def _parse_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        for message in messages:
            api_resource = cast(Any, self.api_resource)
            message_data = (
                api_resource.users()
                .messages()
                .get(userId="me", format="full", id=message["id"])
                .execute()
            )
            headers = {
                header["name"].lower(): header.get("value", "")
                for header in message_data.get("payload", {}).get("headers", [])
            }
            results.append(
                {
                    "id": message["id"],
                    "threadId": message_data.get("threadId", ""),
                    "snippet": message_data.get("snippet", ""),
                    "body": clean_email_body(
                        self._plain_text(message_data.get("payload", {}))
                    ),
                    "subject": headers.get("subject", ""),
                    "sender": headers.get("from", ""),
                    "from": headers.get("from", ""),
                    "date": headers.get("date", ""),
                    "to": headers.get("to", ""),
                    "cc": headers.get("cc", ""),
                }
            )
        return results


def _get_api_resource():
    global _gmail_api_resource
    if not _gmail_api_resource:
        _gmail_api_resource = _get_gmail_service()
    return _gmail_api_resource


def get_gmail_toolkit_tools() -> List[Any]:
    """Get all Gmail tools from the LangChain toolkit.

    Returns:
        List of LangChain tools (search, read, get thread, create draft)
    """
    toolkit = GmailToolkit(api_resource=_get_api_resource())
    return [
        (
            SafeGmailSearch(api_resource=tool.api_resource)
            if isinstance(tool, GmailSearch)
            else SignedGmailCreateDraft(api_resource=tool.api_resource)
            if isinstance(tool, GmailCreateDraft)
            else SignedGmailSendMessage(api_resource=tool.api_resource)
            if isinstance(tool, GmailSendMessage)
            else tool
        )
        for tool in toolkit.get_tools()
    ]
