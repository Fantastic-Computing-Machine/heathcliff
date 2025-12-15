# ABOUTME: Google OAuth2 authentication utilities for Gmail, Calendar, and Drive APIs
# ABOUTME: Handles credential loading, token refresh, and scope management

import os
import pickle
from threading import Lock
from typing import Dict, List, Optional, Tuple

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

_cache_lock = Lock()
_credential_cache: Dict[Tuple, Credentials] = {}


def _cache_key(scopes: List[str], token_file: str, credentials_file: Optional[str]) -> Tuple:
    """Generate a hashable cache key for credential retrieval."""

    normalized_scopes = tuple(sorted(set(scopes)))
    resolved_credentials_file = credentials_file or os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
    return (normalized_scopes, token_file, resolved_credentials_file)


def _load_from_disk(token_file: str) -> Optional[Credentials]:
    """Load credentials from disk if the token file exists."""

    if os.path.exists(token_file):
        with open(token_file, "rb") as token:
            return pickle.load(token)
    return None


def get_google_credentials(
    scopes: List[str],
    token_file: str = "token.pickle",
    credentials_file: Optional[str] = None,
) -> Credentials:
    """
    Get or create Google OAuth2 credentials with specified scopes.

    Args:
        scopes: List of OAuth2 scopes required
        token_file: Path to store/load the token pickle file
        credentials_file: Path to credentials.json (if None, uses GOOGLE_APPLICATION_CREDENTIALS env var)

    Returns:
        Google OAuth2 Credentials object

    Raises:
        FileNotFoundError: If credentials file not found
    """
    key = _cache_key(scopes, token_file, credentials_file)

    with _cache_lock:
        creds = _credential_cache.get(key)

        if not creds:
            creds = _load_from_disk(token_file)

        if creds and not creds.valid and creds.expired and creds.refresh_token:
            creds.refresh(Request())

        if not creds or not creds.valid:
            resolved_credentials_file = credentials_file or os.getenv(
                "GOOGLE_APPLICATION_CREDENTIALS"
            )

            if not resolved_credentials_file or not os.path.exists(resolved_credentials_file):
                raise FileNotFoundError(
                    "Google credentials file not found. "
                    "Set GOOGLE_APPLICATION_CREDENTIALS env var or pass credentials_file parameter."
                )

            flow = InstalledAppFlow.from_client_secrets_file(resolved_credentials_file, scopes)
            creds = flow.run_local_server(port=0)

            with open(token_file, "wb") as token:
                pickle.dump(creds, token)

        _credential_cache[key] = creds

        return creds


# Common Google API scopes
GMAIL_SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.compose",
]

CALENDAR_SCOPES = [
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/calendar.events",
]

DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

PEOPLE_SCOPES = ["https://www.googleapis.com/auth/contacts.readonly"]

ALL_GOOGLE_SCOPES = GMAIL_SCOPES + CALENDAR_SCOPES + DRIVE_SCOPES + PEOPLE_SCOPES
