"""Standard signature for messages Heathcliff sends externally."""

import re
from html import escape

from config import Config

_DISCLAIMER = (
    "This is sent by Heathcliff an Autonomous Intelligence system. "
    "It may make mistakes."
)

_HEADING = re.compile(r"^(#{1,3})\s+(.+)$")
_BULLET = re.compile(r"^[-*+]\s+(.+)$")
_NUMBERED = re.compile(r"^\d+[.)]\s+(.+)$")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITALIC = re.compile(r"(?<!\w)_(.+?)_(?!\w)")
_URL = re.compile(r"(https?://[^\s<]+)")


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


def format_outbound_email(message: str) -> str:
    """Render ordinary Markdown-style agent text as safe, readable email HTML."""
    lines: list[str] = []
    paragraph: list[str] = []
    list_tag: str | None = None

    def inline(text: str) -> str:
        rendered = escape(text)
        rendered = _BOLD.sub(r"<strong>\1</strong>", rendered)
        rendered = _ITALIC.sub(r"<em>\1</em>", rendered)
        return _URL.sub(r'<a href="\1">\1</a>', rendered)

    def flush_paragraph() -> None:
        if paragraph:
            lines.append(f"<p>{'<br>'.join(inline(line) for line in paragraph)}</p>")
            paragraph.clear()

    def close_list() -> None:
        nonlocal list_tag
        if list_tag:
            lines.append(f"</{list_tag}>")
            list_tag = None

    for raw_line in append_outbound_signature(message).splitlines():
        text = raw_line.strip()
        if not text:
            flush_paragraph()
            close_list()
            continue
        heading = _HEADING.match(text)
        bullet = _BULLET.match(text)
        numbered = _NUMBERED.match(text)
        if heading:
            flush_paragraph()
            close_list()
            level = min(len(heading.group(1)) + 1, 4)
            lines.append(f"<h{level}>{inline(heading.group(2))}</h{level}>")
        elif bullet or numbered:
            flush_paragraph()
            wanted_tag = "ul" if bullet else "ol"
            item = bullet or numbered
            assert item is not None
            if list_tag != wanted_tag:
                close_list()
                lines.append(f"<{wanted_tag}>")
                list_tag = wanted_tag
            lines.append(f"<li>{inline(item.group(1))}</li>")
        else:
            close_list()
            paragraph.append(text)
    flush_paragraph()
    close_list()
    return "\n".join(lines)
