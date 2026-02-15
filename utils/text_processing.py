# ABOUTME: Text processing helpers for cleaning scraped web content
# ABOUTME: Provides clean_text and extract_main_content for search tools

from __future__ import annotations

from typing import Optional


def clean_text(text: str, max_length: int = 15_000) -> str:
    """Lightly clean scraped text: strip, drop duplicate lines, clamp length.

    Args:
        text: Raw scraped text.
        max_length: Maximum character length to return. 0 means no limit.

    Returns:
        Cleaned text string.
    """
    lines: list[str] = []
    seen: set[str] = set()
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            # keep single blank lines only
            if lines and lines[-1] != "":
                lines.append("")
            continue
        # drop common noisy strings
        if line.lower().startswith("there was an error while loading"):
            continue
        if line in seen:
            continue
        seen.add(line)
        lines.append(line)
    cleaned = "\n".join(lines)
    if max_length and len(cleaned) > max_length:
        cleaned = cleaned[:max_length]
    return cleaned


def extract_main_content(soup) -> Optional[str]:
    """Extract the most relevant content block from a BeautifulSoup tree.

    Prefers semantic containers (``<main>``, ``<article>``,
    ``role="main"``) over the full ``<body>``.

    Args:
        soup: A ``BeautifulSoup`` document.

    Returns:
        Extracted text, or ``None`` if no content was found.
    """
    # Remove noise elements first
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header"]):
        tag.decompose()

    # Try semantic containers in priority order
    container = (
        soup.find("main")
        or soup.find(attrs={"role": "main"})
        or soup.find("article")
        or soup.body
    )
    if container is None:
        return None

    text = container.get_text("\n", strip=True)
    return text if text.strip() else None
