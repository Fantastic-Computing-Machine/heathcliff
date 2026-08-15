"""Standard signature for messages Heathcliff sends externally."""

from html import escape

from config import Config

_DISCLAIMER = (
    "This is sent by Heathcliff an Autonomous Intelligence system. "
    "It may make mistakes."
)


def append_outbound_signature(message: str, *, html: bool = False) -> str:
    """Append Heathcliff's signature and disclosure once to an outbound message."""
    profile = Config.MASTER_INFO if isinstance(Config.MASTER_INFO, dict) else {}
    master_name = str(profile.get("full_name") or profile.get("name") or "User").strip()
    signature = f"Heathcliff o.b.o {master_name}"
    if signature in message and _DISCLAIMER in message:
        return message
    if html:
        return f"{message}<br><br>{escape(signature)}<br><small>{_DISCLAIMER}</small>"
    return f"{message.rstrip()}\n\n{signature}\n{_DISCLAIMER}"
