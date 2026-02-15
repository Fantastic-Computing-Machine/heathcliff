# ABOUTME: Google OAuth2 authentication utilities for Gmail, Calendar, and Drive APIs
# ABOUTME: Handles credential loading, token refresh, and scope management
import json
import os
import warnings
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

# Suppress DeprecationWarnings from httplib2 (used by google-auth) which uses deprecated pyparsing methods
warnings.filterwarnings("ignore", category=DeprecationWarning, module="httplib2")

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from config import Config

_cache_lock = Lock()
_credential_cache: Dict[Tuple, Any] = {}

TOKEN_FILE_DEFAULT = Config.GOOGLE_TOKEN_FILE_PATH


def _cache_key(
    scopes: List[str], token_file: str, credentials_file: Optional[str]
) -> Tuple:
    """Generate a hashable cache key for credential retrieval."""

    normalized_scopes = tuple(sorted(set(scopes)))
    resolved_credentials_file = credentials_file or os.getenv(
        "GOOGLE_APPLICATION_CREDENTIALS", ""
    )
    return (normalized_scopes, token_file, resolved_credentials_file)


def _load_from_disk(token_file: str) -> Optional[Any]:
    """Load credentials from disk if the token file exists."""

    if os.path.exists(token_file):
        with open(token_file, "r") as token:
            data = json.load(token)
            return Credentials(
                token=data.get("token"),
                refresh_token=data.get("refresh_token"),
                token_uri=data.get("token_uri"),
                client_id=data.get("client_id"),
                client_secret=data.get("client_secret"),
                scopes=data.get("scopes"),
            )
    return None


def _save_to_disk(creds: Any, token_file: str) -> None:
    """Save credentials to disk as JSON."""

    data = {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": creds.scopes,
    }
    with open(token_file, "w") as token:
        json.dump(data, token, indent=2)


def get_google_credentials(
    scopes: List[str],
    credentials_file: Optional[str] = None,
) -> Any:
    """
    Get or create Google OAuth2 credentials with specified scopes.

    Args:
        scopes: List of OAuth2 scopes required
        credentials_file: Path to credentials.json (if None, uses GOOGLE_APPLICATION_CREDENTIALS env var)

    Returns:
        Google OAuth2 Credentials object

    Raises:
        FileNotFoundError: If credentials file not found
    """
    key = _cache_key(scopes, TOKEN_FILE_DEFAULT, credentials_file)

    with _cache_lock:
        creds = _credential_cache.get(key)

        if not creds:
            creds = _load_from_disk(TOKEN_FILE_DEFAULT)

        if creds and not creds.valid and creds.expired and creds.refresh_token:
            creds.refresh(Request())

        if not creds or not creds.valid:
            resolved_credentials_file = credentials_file or os.getenv(
                "GOOGLE_APPLICATION_CREDENTIALS"
            )

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

            # Use run_local_server with open_browser=False for headless/CLI environments.
            # This prints the auth URL to console for manual browser access.
            creds = flow.run_local_server(port=0, open_browser=False)

            _save_to_disk(creds, TOKEN_FILE_DEFAULT)

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

if __name__ == "__main__":
    get_google_credentials(ALL_GOOGLE_SCOPES)
