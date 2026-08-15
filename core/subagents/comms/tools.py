# ABOUTME: Communication tools for Telegram messaging
# ABOUTME: Integrates the asynchronous Telegram Bot API with sync LangChain tools

import asyncio
from typing import Any, List

from langchain.tools import tool
from telegram import Bot as TelegramBot

from config import Config
from logger import logger
from utils.outbound_signature import append_outbound_signature

_telegram_bot = None


def _get_telegram_bot() -> "TelegramBot":
    """Get Telegram Bot instance (singleton)."""
    global _telegram_bot

    if _telegram_bot is None:
        config = Config
        token = config.TELEGRAM_BOT_TOKEN
        if not token:
            raise ValueError("Telegram bot token not configured")
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

        asyncio.run(
            bot.send_message(chat_id=chat_id, text=append_outbound_signature(message))
        )
        logger.info("Telegram message sent successfully")
        return "Message sent to Telegram successfully"

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
