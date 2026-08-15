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


def _outbound_identity() -> tuple[str, str]:
    profile = Config.MASTER_INFO if isinstance(Config.MASTER_INFO, dict) else {}
    master_name = str(profile.get("full_name") or profile.get("name") or "User").strip()
    return master_name, f"Heathcliff o.b.o {master_name}"


def append_outbound_signature(message: str, *, html: bool = False) -> str:
    """Append Heathcliff's signature and disclosure once to an outbound message."""
    _, signature = _outbound_identity()
    if signature in message and _DISCLAIMER in message:
        return message
    if html:
        return f"{message}<br><br>{escape(signature)}<br><small>{_DISCLAIMER}</small>"
    return f"{message.rstrip()}\n\n{signature}\n{_DISCLAIMER}"


def format_outbound_email(message: str) -> str:
    """Render ordinary Markdown-style agent text in Heathcliff's email card."""
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

    for raw_line in message.splitlines():
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
    _, signature = _outbound_identity()
    content = "\n".join(lines) or "<p>At your service.</p>"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    @media only screen and (max-width: 640px) {{
      .heathcliff-card {{ width: 100% !important; border-radius: 0 !important; }}
      .heathcliff-content {{ padding: 28px 24px !important; }}
    }}
    .heathcliff-copy h2 {{ color: #16243b; font-size: 23px; line-height: 1.3; margin: 0 0 18px; }}
    .heathcliff-copy h3 {{ color: #16243b; font-size: 18px; line-height: 1.35; margin: 24px 0 10px; }}
    .heathcliff-copy h4 {{ color: #56657a; font-size: 14px; letter-spacing: .04em; margin: 20px 0 8px; text-transform: uppercase; }}
    .heathcliff-copy p {{ color: #314158; font-size: 16px; line-height: 1.7; margin: 0 0 16px; }}
    .heathcliff-copy ul, .heathcliff-copy ol {{ color: #314158; font-size: 16px; line-height: 1.65; margin: 0 0 18px; padding-left: 24px; }}
    .heathcliff-copy li {{ margin: 7px 0; }}
    .heathcliff-copy a {{ color: #0f5d78; text-decoration: underline; }}
  </style>
</head>
<body style="margin:0; padding:0; background:#eef2f6;">
  <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="background:#eef2f6;">
    <tr><td align="center" style="padding:36px 16px;">
      <table role="presentation" class="heathcliff-card" width="620" cellspacing="0" cellpadding="0" border="0" style="width:620px; max-width:620px; background:#ffffff; border:1px solid #dbe2ea; border-radius:18px; overflow:hidden; box-shadow:0 12px 32px rgba(17,34,54,.10);">
        <tr><td style="background:#121d31; padding:28px 34px;">
          <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0"><tr>
            <td width="46" valign="middle"><div style="width:36px; height:36px; border:1px solid #cfad68; border-radius:50%; color:#f5dfab; font-family:Georgia,serif; font-size:22px; line-height:36px; text-align:center;">H</div></td>
            <td valign="middle" style="padding-left:12px;"><div style="color:#ffffff; font-family:Georgia,serif; font-size:22px; letter-spacing:.04em;">HEATHCLIFF</div><div style="color:#cfad68; font-family:Arial,sans-serif; font-size:10px; letter-spacing:.16em; margin-top:4px; text-transform:uppercase;">Autonomous Intelligence</div></td>
          </tr></table>
        </td></tr>
        <tr><td class="heathcliff-content" style="padding:38px 42px 30px;">
          <div class="heathcliff-copy" style="font-family:Arial,Helvetica,sans-serif;">{content}</div>
        </td></tr>
        <tr><td style="padding:0 42px;"><div style="height:1px; background:#e5e9ee;"></div></td></tr>
        <tr><td style="background:#faf8f3; padding:24px 42px 28px;">
          <div style="color:#16243b; font-family:Georgia,serif; font-size:16px;">{escape(signature)}</div>
          <div style="color:#718096; font-family:Arial,Helvetica,sans-serif; font-size:11px; line-height:1.55; margin-top:7px;">{_DISCLAIMER}</div>
        </td></tr>
      </table>
    </td></tr>
  </table>
</body>
</html>"""
