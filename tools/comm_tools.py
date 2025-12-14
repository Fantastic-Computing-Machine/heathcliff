# ABOUTME: Communication tools for Telegram messaging and Google Drive file access
# ABOUTME: Integrates Telegram Bot API and Google Drive API

import io
from typing import TYPE_CHECKING, Any, List

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from langchain.tools import tool

from config import get_config
from logger import logger
from utils.google_auth import DRIVE_SCOPES, get_google_credentials

if TYPE_CHECKING:  # pragma: no cover - typing helper
    from telegram import Bot as TelegramBot

_telegram_bot = None


def _import_telegram_bot():
    """Import python-telegram-bot's Bot class lazily to avoid hard dependency at import time."""
    try:
        from telegram import Bot as TelegramBot
    except ImportError as exc:  # pragma: no cover - dependency missing
        raise ImportError(
            "python-telegram-bot is required for Telegram features. "
            "Install it via `pip install python-telegram-bot`."
        ) from exc

    return TelegramBot


def _get_telegram_bot() -> "TelegramBot":
    """Get Telegram Bot instance (singleton)."""
    global _telegram_bot

    if _telegram_bot is None:
        config = get_config()
        token = config.telegram_token
        if not token:
            raise ValueError("Telegram bot token not configured")
        TelegramBot = _import_telegram_bot()
        _telegram_bot = TelegramBot(token=token)

    return _telegram_bot


def _get_drive_service():
    """Get authenticated Google Drive API service."""
    creds = get_google_credentials(DRIVE_SCOPES, token_file="drive_token.pickle")
    return build("drive", "v3", credentials=creds)


@tool
def send_to_telegram(message: str) -> str:
    """
    Send a message to Telegram. Use this to notify the user via Telegram.

    Args:
        message: Message text to send

    Returns:
        Confirmation message
    """
    try:
        logger.debug(f"Sending Telegram message: {message[:50]}...")
        config = get_config()
        bot = _get_telegram_bot()
        chat_id = config.telegram_chat_id

        if not chat_id:
            logger.warning("Telegram chat ID not configured")
            return "Telegram chat ID not configured"

        bot.send_message(chat_id=chat_id, text=message)
        logger.info("Telegram message sent successfully")
        return f"Message sent to Telegram successfully"

    except Exception as e:
        logger.error(f"Error sending Telegram message: {e}", exc_info=True)
        return f"Error sending Telegram message: {str(e)}"


@tool
def read_gdrive_file(file_id: str) -> str:
    """
    Read a text file from Google Drive. Use this to access user's Drive files.

    Args:
        file_id: Google Drive file ID

    Returns:
        File content as text
    """
    try:
        logger.debug(f"Reading Google Drive file: {file_id}")
        service = _get_drive_service()

        # Get file metadata
        file_metadata = service.files().get(fileId=file_id).execute()
        file_name = file_metadata.get("name", "Unknown")
        mime_type = file_metadata.get("mimeType", "")
        logger.debug(f"File metadata: name={file_name}, mime_type={mime_type}")

        # Check if it's a text file
        if not any(t in mime_type for t in ["text", "plain", "document"]):
            logger.warning(f"File '{file_name}' is not a text file (type: {mime_type})")
            return f"File '{file_name}' is not a text file (type: {mime_type})"

        # Download file content
        request = service.files().get_media(fileId=file_id)
        file_buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(file_buffer, request)

        done = False
        while not done:
            status, done = downloader.next_chunk()
            if status:
                logger.debug(f"Download progress: {int(status.progress() * 100)}%")

        # Read content
        file_buffer.seek(0)
        content = file_buffer.read().decode("utf-8")
        logger.info(f"Successfully read Google Drive file: {file_name} ({len(content)} bytes)")

        return f"File: {file_name}\n\n{content}"

    except Exception as e:
        logger.error(f"Error reading Google Drive file {file_id}: {e}", exc_info=True)
        return f"Error reading Google Drive file: {str(e)}"


def get_comm_tools() -> List[Any]:
    """
    Get all communication tools as a list for agent registration.

    Returns:
        List of LangChain tools
    """
    return [send_to_telegram, read_gdrive_file]
