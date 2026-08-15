# ABOUTME: Google OAuth2 authentication utilities for Gmail, Calendar, and Drive APIs
# ABOUTME: Handles credential loading, token refresh, and scope management
import sys

sys.path.append(".")

import os
from threading import Lock
from typing import Dict, List, Optional, Tuple

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

_cache_lock = Lock()
_credential_cache: Dict[Tuple, Credentials] = {}

GOOGLE_CREDENTIALS_FILE = "keys/credentials.json"
GOOGLE_TOKEN_FILE = "keys/token.json"


def _cache_key(
    scopes: List[str], token_file: str, credentials_file: Optional[str]
) -> Tuple:
    """Generate a hashable cache key for credential retrieval."""

    normalized_scopes = tuple(sorted(set(scopes)))
    resolved_credentials_file = credentials_file or os.getenv(
        "GOOGLE_APPLICATION_CREDENTIALS", ""
    )
    return (normalized_scopes, token_file, resolved_credentials_file)


def _load_from_disk(
    token_file: str, scopes: Optional[List[str]] = None
) -> Optional[Credentials]:
    """Load credentials from disk if the token file exists."""

    if os.path.exists(token_file):
        try:
            return Credentials.from_authorized_user_file(token_file, scopes)
        except Exception:
            pass
    return None


def get_google_credentials(
    scopes: List[str],
    token_file: str = GOOGLE_TOKEN_FILE,
    credentials_file: Optional[str] = GOOGLE_CREDENTIALS_FILE,
) -> Credentials:
    """
    Get or create Google OAuth2 credentials with specified scopes.

    Args:
        scopes: List of OAuth2 scopes required
        token_file: Path to store/load the token pickle file
        credentials_file: Path to credentials.json (if None, uses GOOGLE_APPLICATION_CREDENTIALS env var or keys/credentials.json)

    Returns:
        Google OAuth2 Credentials object

    Raises:
        FileNotFoundError: If credentials file not found
    """
    key = _cache_key(scopes, token_file, credentials_file)

    with _cache_lock:
        creds = _credential_cache.get(key)

        if not creds:
            creds = _load_from_disk(token_file, scopes)

        if creds and not creds.valid and creds.expired and creds.refresh_token:
            creds.refresh(Request())

        if not creds or not creds.valid:
            resolved_credentials_file = credentials_file or os.getenv(
                "GOOGLE_APPLICATION_CREDENTIALS"
            )

            if not resolved_credentials_file and os.path.exists(
                GOOGLE_CREDENTIALS_FILE
            ):
                resolved_credentials_file = GOOGLE_CREDENTIALS_FILE

            if not resolved_credentials_file or not os.path.exists(
                resolved_credentials_file
            ):
                raise FileNotFoundError(
                    "Google credentials file not found. "
                    "Set GOOGLE_APPLICATION_CREDENTIALS env var or pass credentials_file parameter."
                )

            flow = InstalledAppFlow.from_client_secrets_file(
                resolved_credentials_file, scopes
            )
            creds = flow.run_local_server(port=0, open_browser=False)

            token_dir = os.path.dirname(token_file)
            if token_dir:
                os.makedirs(token_dir, exist_ok=True)

            with open(token_file, "w") as token:
                token.write(creds.to_json())

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

KEEP_SCOPES = ["https://www.googleapis.com/auth/keep.readonly"]

DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

PEOPLE_SCOPES = ["https://www.googleapis.com/auth/contacts.readonly"]

ALL_GOOGLE_SCOPES = GMAIL_SCOPES + CALENDAR_SCOPES + DRIVE_SCOPES + PEOPLE_SCOPES


# if __name__ == "__main__":
#     creds = get_google_credentials(ALL_GOOGLE_SCOPES)
#     print(creds)
