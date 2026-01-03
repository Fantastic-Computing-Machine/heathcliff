# ABOUTME: Communication tools for Telegram messaging and Google Drive file access
# ABOUTME: Integrates Telegram Bot API and Google Drive API

import io
from typing import TYPE_CHECKING, Any, List

from langchain.tools import tool

from config import Config
from logger import logger
from utils.google_auth import get_google_credentials

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
        config = Config
        token = config.TELEGRAM_BOT_TOKEN
        if not token:
            raise ValueError("Telegram bot token not configured")
        TelegramBot = _import_telegram_bot()
        _telegram_bot = TelegramBot(token=token)

    return _telegram_bot


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
        config = Config
        bot = _get_telegram_bot()
        chat_id = config.TELEGRAM_CHAT_ID

        if not chat_id:
            logger.warning("Telegram chat ID not configured")
            return "Telegram chat ID not configured"

        bot.send_message(chat_id=chat_id, text=message)
        logger.info("Telegram message sent successfully")
        return f"Message sent to Telegram successfully"

    except Exception as e:
        logger.error(f"Error sending Telegram message: {e}", exc_info=True)
        return f"Error sending Telegram message: {str(e)}"


def get_comm_tools() -> List[Any]:
    """
    Get all communication tools as a list for agent registration.

    Returns:
        List of LangChain tools
    """
    return [send_to_telegram]
