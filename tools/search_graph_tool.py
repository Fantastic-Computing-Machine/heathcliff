# ABOUTME: Search-and-scrape tool using DuckDuckGo + Playwright with stealth headers
# ABOUTME: Uses LangGraph pipeline (Search -> Fetch) with concurrent fetching

from __future__ import annotations

import concurrent.futures
from typing import Any, Dict, List, Optional, TypedDict

from duckduckgo_search import DDGS
from langchain.tools import tool
from langgraph.graph import StateGraph
from pydantic import BaseModel, ConfigDict, Field

from logger import logger
from utils.http_client import StealthHttpClient
from utils.text_processing import clean_text, extract_main_content

# Module-level stealth client (shared across requests)
_http = StealthHttpClient(max_retries=3, base_timeout=20, backoff_factor=0.8)


class SearchState(TypedDict, total=False):
    """State flowing through the LangGraph search-scrape pipeline."""

    query: str
    max_results: int
    selector: Optional[str]
    wait_for: Optional[str]
    timeout: int
    urls: List[str]
    snippets: List[str]


# ---------------------------------------------------------------------------
# Search node
# ---------------------------------------------------------------------------


def _search_urls(query: str, max_results: int) -> List[str]:
    """Return top search-result URLs from DuckDuckGo (text API)."""
    try:
        with DDGS() as ddgs:
            hits = list(ddgs.text(query, max_results=max_results))
        return [h.get("href") for h in hits if h.get("href")]
    except Exception as exc:
        logger.warning("DuckDuckGo search failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------


def _fetch_with_requests(url: str) -> Optional[str]:
    """Fetch a page via ``StealthHttpClient`` (requests-based).

    Returns extracted text or ``None`` on failure.
    """
    try:
        from bs4 import BeautifulSoup

        resp = _http.get(url)

        # Non-success status
        if resp.status_code >= 400:
            logger.debug(
                "Stealth requests fetch for %s returned %d", url, resp.status_code
            )
            return None

        ctype = resp.headers.get("content-type", "")
        if "application/pdf" in ctype:
            return f"{url}\nDownloadable PDF detected; cannot render."

        soup = BeautifulSoup(resp.text, "html.parser")
        text = extract_main_content(soup)
        if not text:
            return None

        return clean_text(text)

    except Exception as exc:
        logger.debug("requests fetch failed for %s: %s", url, exc)
        return None


def _fetch_with_playwright(
    url: str,
    selector: Optional[str],
    wait_for: Optional[str],
    timeout: int,
) -> str:
    """Fetch a page with headless Chromium, falling back to requests.

    Uses randomised headers from ``StealthHttpClient`` in the browser
    context and in the requests fallback path.
    """
    try:
        from bs4 import BeautifulSoup
        from playwright.sync_api import sync_playwright
    except Exception as exc:  # pragma: no cover
        logger.warning("Playwright not available, trying requests fallback")
        result = _fetch_with_requests(url)
        if result:
            return result
        return f"{url}\nPlaywright/bs4 not available: {exc}"

    headers = _http.random_headers()
    user_agent = headers.pop("User-Agent")

    html = ""
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent=user_agent,
                extra_http_headers=headers,
                viewport={"width": 1920, "height": 1080},
                locale="en-US",
            )
            page = context.new_page()

            try:
                page.goto(url, wait_until="domcontentloaded", timeout=timeout)
                if wait_for:
                    try:
                        page.wait_for_selector(wait_for, timeout=timeout)
                    except Exception:
                        logger.debug(
                            "wait_for selector not found on %s; continuing", url
                        )

                # Scroll to trigger lazy-loaded content
                for _ in range(3):
                    page.evaluate("window.scrollBy(0, window.innerHeight)")
                    page.wait_for_timeout(300)
                page.evaluate("window.scrollTo(0, 0)")
                page.wait_for_timeout(200)

                html = page.content()

            except Exception as nav_exc:
                logger.warning(
                    "Playwright navigation failed for %s: %s — falling back to requests",
                    url,
                    nav_exc,
                )
                # Fallback: stealth requests fetch
                fallback = _fetch_with_requests(url)
                if fallback:
                    context.close()
                    browser.close()
                    return fallback

                context.close()
                browser.close()
                return (
                    f"{url}\nPlaywright navigation failed: {nav_exc}\n"
                    f"Requests fallback also failed."
                )

            context.close()
            browser.close()

        # Parse returned HTML
        soup = BeautifulSoup(html, "html.parser")

        if selector:
            nodes = soup.select(selector)
            if not nodes:
                return f"{url}\n(no elements matched selector '{selector}')"
            text = "\n\n".join(node.get_text(" ", strip=True) for node in nodes)
        else:
            text = extract_main_content(soup) or ""

        text = clean_text(text)
        if not text:
            return f"{url}\n(page returned no extractable content)"
        return f"{url}\n{text}"

    except Exception as exc:  # pragma: no cover
        logger.error("Playwright fetch failed for %s", url, exc_info=True)
        # Last-resort requests fallback
        fallback = _fetch_with_requests(url)
        if fallback:
            return fallback
        return f"{url}\nError fetching page: {exc}"


def _fetch_url(
    url: str,
    selector: Optional[str],
    wait_for: Optional[str],
    timeout: int,
) -> str:
    """Wrapper that tries Playwright first, then pure requests."""
    return _fetch_with_playwright(url, selector, wait_for, timeout)


# ---------------------------------------------------------------------------
# LangGraph pipeline
# ---------------------------------------------------------------------------


def _build_search_scrape_graph():
    """Build a LangGraph pipeline: Search -> Fetch (concurrent)."""

    def search_node(state: SearchState) -> SearchState:
        query = state["query"]
        max_results = state.get("max_results", 3)
        urls = _search_urls(query, max_results)
        return {"urls": urls}

    def fetch_node(state: SearchState) -> SearchState:
        selector = state.get("selector")
        wait_for = state.get("wait_for")
        timeout = state.get("timeout", 12000)
        urls: List[str] = state.get("urls", [])

        if not urls:
            return {"snippets": []}

        # Concurrent fetching for faster results
        snippets: List[str] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = {
                pool.submit(_fetch_url, url, selector, wait_for, timeout): url
                for url in urls
            }
            for future in concurrent.futures.as_completed(futures):
                url = futures[future]
                try:
                    snippets.append(future.result())
                except Exception as exc:
                    logger.error("Fetch failed for %s: %s", url, exc)
                    snippets.append(f"{url}\nFetch error: {exc}")

        return {"snippets": snippets}

    graph: StateGraph[SearchState] = StateGraph(SearchState)
    graph.add_node("search", search_node)
    graph.add_node("fetch", fetch_node)

    graph.set_entry_point("search")
    graph.add_edge("search", "fetch")
    graph.set_finish_point("fetch")

    return graph.compile()


_SEARCH_SCRAPE_GRAPH = _build_search_scrape_graph()


class SearchAndScrapeArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(..., description="Search query")
    max_results: int = Field(3, description="Max results (default 3)")
    selector: Optional[str] = Field(
        None, description="CSS selector (optional, send null if none)"
    )
    wait_for: Optional[str] = Field(
        None, description="Wait for selector (optional, send null if none)"
    )
    timeout_ms: int = Field(12000, description="Timeout in ms (default 12000)")


@tool(
    "search_and_scrape",
    args_schema=SearchAndScrapeArgs,
    description=(
        "Search DuckDuckGo for a query, then scrape top results with headless Chromium. "
        "Uses randomised browser identities to avoid bot detection. "
        "Retries automatically on 403/429 with fresh headers. "
        "Fetches URLs concurrently for speed."
    ),
)
def search_and_scrape(
    query: str,
    max_results: int = 3,
    selector: Optional[str] = None,
    wait_for: Optional[str] = None,
    timeout_ms: int = 12000,
) -> str:
    """
    Search DuckDuckGo for a query, then scrape top results with headless Chromium.

    Uses randomised browser identities (User-Agent, Referer, client hints)
    to avoid bot detection. Retries automatically on 403/429 with fresh headers.
    Fetches URLs concurrently for speed.

    Args:
        query: Full search query with all relevant keywords
        max_results: Maximum number of search results to fetch (default 3)
        selector: Optional CSS selector to extract specific elements
        wait_for: Optional CSS selector to wait for before extracting content
        timeout_ms: Timeout in milliseconds for page loads (default 12000)

    Returns:
        Scraped content from top search results, separated by dividers
    """

    state: SearchState = {
        "query": query,
        "max_results": max_results,
        "selector": selector,
        "wait_for": wait_for,
        "timeout": max(timeout_ms, 1000),
    }

    result = _SEARCH_SCRAPE_GRAPH.invoke(state)
    snippets = result.get("snippets") or []
    if not snippets:
        return "No content returned from search_and_scrape."
    return "\n\n---\n\n".join(snippets)
