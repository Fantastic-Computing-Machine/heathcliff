# ABOUTME: Information retrieval tools for weather, news, web search, Wikipedia, Wikidata, StackExchange, and NASA
# ABOUTME: Integrates LangChain community wrappers/toolkits with recent-context capture

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import requests
from bs4 import BeautifulSoup
from langchain.tools import tool
from langchain_community.agent_toolkits.nasa.toolkit import NasaToolkit
from langchain_community.tools import YouTubeSearchTool
from langchain_community.tools.ddg_search.tool import (
    DuckDuckGoSearchResults as DuckDuckGoSearchTool,
)
from langchain_community.tools.google_search import GoogleSearchResults
from langchain_community.tools.stackexchange.tool import StackExchangeTool
from langchain_community.tools.wikidata.tool import WikidataAPIWrapper, WikidataQueryRun
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain_community.utilities import (
    GoogleSearchAPIWrapper,
    OpenWeatherMapAPIWrapper,
    StackExchangeAPIWrapper,
)
from langchain_community.utilities.nasa import NasaAPIWrapper

from config import Config
from core.subagents.info.recent_context import _capture_recent_result, recent_context
from logger import logger

_google_tool: Optional[Any] = None
_duck_tool: Optional[Any] = None
_wikidata_tool: Optional[Any] = None
_stackexchange_tool: Optional[Any] = None
_nasa_tools: Optional[Dict[str, Any]] = None


def _init_google_tool(config) -> Optional[Any]:
    """Lazily instantiate the Google search tool."""

    global _google_tool

    if _google_tool is None:
        try:
            api_wrapper = GoogleSearchAPIWrapper(
                google_api_key=config.GOOGLE_CSE_API_KEY,
                google_cse_id=config.GOOGLE_CSE_ID,
            )
            _google_tool = GoogleSearchResults(api_wrapper=api_wrapper)
            logger.info("Google search tool initialized successfully")
        except Exception as exc:
            logger.error(f"Failed to initialize Google search: {exc}", exc_info=True)
            return None

    return _google_tool


def _init_duck_tool() -> Optional[Any]:
    """Lazily instantiate the DuckDuckGo search tool."""

    global _duck_tool

    if _duck_tool is None:
        _duck_tool = DuckDuckGoSearchTool()

    return _duck_tool


def _init_wikidata_tool() -> Optional[Any]:
    """Lazily instantiate the LangChain Wikidata tool."""

    global _wikidata_tool

    if _wikidata_tool is None:
        try:
            _wikidata_tool = WikidataQueryRun(
                api_wrapper=WikidataAPIWrapper(
                    top_k_results=2,
                    doc_content_chars_max=5000,
                    lang="en",
                )
            )
            logger.info("Wikidata tool initialized successfully")
        except Exception as exc:
            logger.warning(
                "Wikidata tool unavailable: %s. Install `mediawikiapi` and "
                "`wikibase-rest-api-client`.",
                exc,
            )
            return None

    return _wikidata_tool


def _init_stackexchange_tool() -> Optional[Any]:
    """Lazily instantiate the LangChain StackExchange tool."""

    global _stackexchange_tool

    if _stackexchange_tool is None:
        try:
            _stackexchange_tool = StackExchangeTool(
                api_wrapper=StackExchangeAPIWrapper(max_results=3, query_type="all")
            )
            logger.info("StackExchange tool initialized successfully")
        except Exception as exc:
            logger.warning(
                "StackExchange tool unavailable: %s. Install `stackapi`.",
                exc,
            )
            return None

    return _stackexchange_tool


def _init_nasa_tools() -> Optional[Dict[str, Any]]:
    """Lazily instantiate NASA toolkit tools keyed by operation mode."""

    global _nasa_tools

    if _nasa_tools is None:
        try:
            toolkit = NasaToolkit.from_nasa_api_wrapper(NasaAPIWrapper())
            _nasa_tools = {}
            for nasa_tool in toolkit.get_tools():
                mode = getattr(nasa_tool, "mode", None)
                if mode:
                    _nasa_tools[mode] = nasa_tool
            logger.info("NASA toolkit initialized with %d tools", len(_nasa_tools))
        except Exception as exc:
            logger.warning(f"NASA toolkit unavailable: {exc}")
            return None

    return _nasa_tools


def _format_search_result(result: Any) -> str:
    """Normalize tool responses into a human-friendly string."""

    if not result:
        return ""

    if isinstance(result, str):
        return result.strip()

    if isinstance(result, list):
        lines = []
        for item in result:
            if isinstance(item, dict):
                title = item.get("title") or item.get("source") or "Result"
                snippet = item.get("snippet") or item.get("content") or ""
                url = item.get("url") or item.get("link") or ""
                fragments = [part for part in [title, snippet, url] if part]
                if fragments:
                    lines.append(" - " + " | ".join(fragments))
            else:
                lines.append(f" - {item}")
        return "\n".join(lines)

    if isinstance(result, dict):
        return json.dumps(result, indent=2)

    return str(result)


def _search_wikipedia(query: str) -> str:
    """Search Wikipedia through its supported REST API."""
    response = requests.get(
        "https://en.wikipedia.org/w/rest.php/v1/search/page",
        params={"q": query, "limit": 3},
        headers={"User-Agent": "Heathcliff/0.1"},
        timeout=10,
    )
    response.raise_for_status()
    pages = response.json().get("pages", [])
    lines = []
    for page in pages:
        title = page.get("title", "Wikipedia result")
        description = page.get("description", "")
        excerpt = BeautifulSoup(page.get("excerpt", ""), "html.parser").get_text(
            " ", strip=True
        )
        lines.append(" — ".join(part for part in [title, description, excerpt] if part))
    return "\n".join(lines)


def _dispatch_search(provider: str, query: str, max_results: int, config) -> str:
    """Execute the configured search provider via LangChain tools."""

    provider = (provider or "").lower()

    if provider == "google":
        google_tool = _init_google_tool(config)
        if not google_tool:
            return ""
        try:
            return _format_search_result(google_tool.run(query))
        except Exception as exc:  # pragma: no cover - network errors
            return f"Google search error: {exc}"

    if provider == "duckduckgo":
        duck_tool = _init_duck_tool()
        if not duck_tool:
            return ""
        try:
            return _format_search_result(duck_tool.run(query))
        except Exception as exc:  # pragma: no cover - network errors
            return f"DuckDuckGo search error: {exc}"

    return ""


def _truncate(text: str, max_chars: int = 260) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _format_nasa_search(raw: str, max_results: int = 5) -> str:
    """Summarize NASA search JSON payload into compact, readable results."""

    try:
        payload = json.loads(raw)
    except Exception:
        return _format_search_result(raw)

    collection = payload.get("collection", {}) if isinstance(payload, dict) else {}
    items = collection.get("items", []) if isinstance(collection, dict) else []

    if not items:
        return "No NASA media results found."

    lines: List[str] = []
    for item in items[:max_results]:
        data = item.get("data", [{}])
        info = data[0] if data and isinstance(data[0], dict) else {}
        title = info.get("title") or "Untitled"
        nasa_id = info.get("nasa_id") or "N/A"
        created = info.get("date_created") or "Unknown date"
        media_type = info.get("media_type") or "unknown"
        description = _truncate(info.get("description", ""), 220)

        media_url = ""
        links = item.get("links", [])
        if links and isinstance(links[0], dict):
            media_url = links[0].get("href", "")

        parts = [
            f"Title: {title}",
            f"NASA ID: {nasa_id}",
            f"Type: {media_type}",
            f"Created: {created}",
        ]
        if description:
            parts.append(f"Description: {description}")
        if media_url:
            parts.append(f"Link: {media_url}")

        lines.append("\n".join(parts))

    return "\n\n".join(lines)


def _format_nasa_manifest(raw: str, nasa_id: str) -> str:
    """Format NASA asset manifest links for readability."""

    try:
        payload = json.loads(raw)
    except Exception:
        return _format_search_result(raw)

    collection = payload.get("collection", {}) if isinstance(payload, dict) else {}
    items = collection.get("items", []) if isinstance(collection, dict) else []

    if not items:
        return f"No manifest entries found for NASA asset: {nasa_id}"

    links: List[str] = []
    for item in items[:15]:
        if isinstance(item, dict) and item.get("href"):
            links.append(f" - {item['href']}")

    if not links:
        return f"No manifest links found for NASA asset: {nasa_id}"

    return f"Manifest links for NASA asset {nasa_id}:\n" + "\n".join(links)


def _format_nasa_location(raw: str, label: str, nasa_id: str) -> str:
    """Format single-location NASA responses (metadata/captions)."""

    try:
        payload = json.loads(raw)
    except Exception:
        return _format_search_result(raw)

    location = payload.get("location") if isinstance(payload, dict) else None
    if not location:
        return f"No {label} location found for NASA asset: {nasa_id}"

    return f"{label.capitalize()} location for NASA asset {nasa_id}: {location}"


@tool
def get_weather(location: str | None = None) -> str:
    """
    Get current weather for a location. Use this when user asks about weather.

    Args:
        location: City or location name. MUST be in the format "City,CountryCode" (e.g. "Paris,FR" or "Jersey City,US"). Do NOT use US state codes like "NJ" as OpenWeatherMap uses country codes. If None, uses default from config.

    Returns:
        Weather description with temperature, conditions, and humidity
    """
    try:
        api_key = Config.OPENWEATHERMAP_API_KEY

        if not api_key:
            return "Weather API key not configured"

        if "OPENWEATHERMAP_API_KEY" not in os.environ:
            os.environ["OPENWEATHERMAP_API_KEY"] = api_key

        if location is None:
            logger.debug("No location provided for weather; using default from config")
            location = Config.DEFAULT_CITY

        logger.debug(f"Fetching weather for location: {location}")
        weather_wrapper = OpenWeatherMapAPIWrapper()
        weather_data = weather_wrapper.run(location)
        _capture_recent_result("get_weather", weather_data)

        logger.info(f"Weather retrieved for {location} using OpenWeatherMapAPIWrapper")
        return weather_data

    except Exception as exc:
        logger.error(
            f"Error fetching weather with wrapper for {location}: {exc}",
            exc_info=True,
        )
        return f"Error fetching weather data: {str(exc)}"


@tool
def get_news(category: str = "technology") -> str:
    """
    Get latest news headlines. Use this when user asks about news.

    Args:
        category: News category (technology, business, science, etc.)

    Returns:
        String with news headlines and descriptions
    """
    try:
        config = Config
        api_key = config.NEWS_API_KEY

        if not api_key:
            return "News API key not configured"

        max_articles = config.MAX_ARTICLES

        url = "https://newsapi.org/v2/top-headlines"
        params = {
            "apiKey": api_key,
            "category": category,
            "language": "en",
            "pageSize": max_articles,
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        articles = data.get("articles", [])

        if not articles:
            return f"No {category} news found"

        news_list = []
        for article in articles[:max_articles]:
            title = article.get("title", "No title")
            description = article.get("description", "")
            source = article.get("source", {}).get("name", "Unknown")
            news_list.append(f"{source}: {title}\n{description}")

        result = "\n\n".join(news_list)
        _capture_recent_result("get_news", result)
        return result

    except requests.exceptions.RequestException as exc:
        return f"Error fetching news: {str(exc)}"
    except Exception as exc:
        return f"Error: {str(exc)}"


@tool
def search_web(query: str, provider: Optional[str] = None) -> str:
    """
    Search the web for information. Use this when user asks to search online.

    Args:
        query: Search query
        provider: Override the configured search provider (google or duckduckgo)

    Returns:
        Search results summary
    """
    try:
        config = Config
        max_results = 5
        primary_provider = provider or "duckduckgo"
        response = _dispatch_search(primary_provider, query, max_results, config)

        if response:
            _capture_recent_result("search_web", response)
            return response

        for fallback in ["google"]:
            if fallback != primary_provider:
                fallback_response = _dispatch_search(
                    fallback, query, max_results, config
                )
                if fallback_response:
                    _capture_recent_result("search_web", fallback_response)
                    return fallback_response

        try:
            wiki_response = _search_wikipedia(query)
            if wiki_response:
                _capture_recent_result("search_web", wiki_response)
                return wiki_response
        except requests.RequestException as exc:
            logger.warning("Wikipedia fallback failed for '%s': %s", query, exc)

        return f"No search results found for: {query}"

    except Exception as exc:
        return f"Error searching web: {str(exc)}"


@tool
def wikipedia_search(query: str) -> str:
    """
    Search Wikipedia for information. Use this for factual queries.

    Args:
        query: Wikipedia search query

    Returns:
        Encyclopedia summaries and page snippets
    """
    try:
        logger.debug(f"Searching Wikipedia for: {query}")
        result = _search_wikipedia(query)
        if not result:
            return f"No Wikipedia page found for: {query}"
        _capture_recent_result("wikipedia_search", result)
        return result
    except requests.RequestException as exc:
        logger.warning("Wikipedia search failed for '%s': %s", query, exc)
        return "Wikipedia is temporarily unavailable. Please try another source."


@tool
def wikidata_search(query: str) -> str:
    """
    Search Wikidata for structured entity facts.

    Args:
        query: Entity name or Wikidata QID (e.g. "Alan Turing" or "Q7251")

    Returns:
        Structured Wikidata facts for top matching entities
    """
    wikidata_tool = _init_wikidata_tool()
    if not wikidata_tool:
        return (
            "Wikidata tool unavailable. Install `mediawikiapi` and "
            "`wikibase-rest-api-client` to enable it."
        )

    try:
        logger.debug(f"Searching Wikidata for: {query}")
        result = _format_search_result(wikidata_tool.run(query))
        if not result:
            return f"No Wikidata result found for: {query}"
        _capture_recent_result("wikidata_search", result)
        return result
    except Exception as exc:
        logger.error(f"Error searching Wikidata for '{query}': {exc}", exc_info=True)
        return f"Error searching Wikidata: {str(exc)}"


@tool
def stackexchange_search(query: str) -> str:
    """
    Search Stack Overflow for programming Q&A.

    Args:
        query: Coding problem or debugging question

    Returns:
        Relevant Stack Overflow question/answer excerpts
    """
    stack_tool = _init_stackexchange_tool()
    if not stack_tool:
        return "StackExchange tool unavailable. Install `stackapi` to enable it."

    try:
        logger.debug(f"Searching StackExchange for: {query}")
        result = _format_search_result(stack_tool.run(query))
        if not result:
            return f"No Stack Overflow results found for: {query}"
        _capture_recent_result("stackexchange_search", result)
        return result
    except Exception as exc:
        logger.error(
            f"Error searching StackExchange for '{query}': {exc}",
            exc_info=True,
        )
        return f"Error searching StackExchange: {str(exc)}"


@tool
def nasa_media_search(
    query: str,
    media_type: Optional[str] = None,
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
    page_size: int = 5,
) -> str:
    """
    Search NASA's Image and Video Library.

    Args:
        query: Free-text search term (e.g. "moon", "Apollo 11")
        media_type: Optional filter (image, video, audio)
        year_start: Optional start year (YYYY)
        year_end: Optional end year (YYYY)
        page_size: Number of results to request/summarize (1-10)

    Returns:
        Summarized NASA media results with IDs and links
    """
    nasa_tools = _init_nasa_tools()
    if not nasa_tools or "search_media" not in nasa_tools:
        return "NASA media search tool is currently unavailable."

    if not query.strip():
        return "Please provide a NASA media search query."

    safe_page_size = max(1, min(page_size, 10))
    params: Dict[str, Any] = {"q": query.strip(), "page_size": safe_page_size}

    if media_type:
        normalized_media_type = media_type.strip().lower()
        allowed_media_types = {"image", "video", "audio"}
        if normalized_media_type not in allowed_media_types:
            return "media_type must be one of: image, video, audio"
        params["media_type"] = normalized_media_type

    if year_start is not None:
        params["year_start"] = str(year_start)
    if year_end is not None:
        params["year_end"] = str(year_end)

    payload = json.dumps(params)

    try:
        raw_result = nasa_tools["search_media"].invoke(payload)
        result = _format_nasa_search(str(raw_result), max_results=safe_page_size)
        _capture_recent_result("nasa_media_search", result)
        return result
    except Exception as exc:
        logger.error(f"NASA media search failed for '{query}': {exc}", exc_info=True)
        return f"Error searching NASA media: {str(exc)}"


@tool
def nasa_media_manifest(nasa_id: str) -> str:
    """
    Get NASA asset manifest links for a NASA media ID.

    Args:
        nasa_id: NASA media identifier

    Returns:
        Download/asset manifest links for the media item
    """
    nasa_tools = _init_nasa_tools()
    if not nasa_tools or "get_media_metadata_manifest" not in nasa_tools:
        return "NASA manifest tool is currently unavailable."

    clean_id = nasa_id.strip()
    if not clean_id:
        return "Please provide a NASA media ID."

    try:
        raw_result = nasa_tools["get_media_metadata_manifest"].invoke(clean_id)
        result = _format_nasa_manifest(str(raw_result), clean_id)
        _capture_recent_result("nasa_media_manifest", result)
        return result
    except Exception as exc:
        logger.error(
            f"NASA manifest lookup failed for '{clean_id}': {exc}",
            exc_info=True,
        )
        return f"Error fetching NASA manifest: {str(exc)}"


@tool
def nasa_media_metadata(nasa_id: str) -> str:
    """
    Get NASA metadata location URL for a NASA media ID.

    Args:
        nasa_id: NASA media identifier

    Returns:
        URL to metadata JSON for the media item
    """
    nasa_tools = _init_nasa_tools()
    if not nasa_tools or "get_media_metadata_location" not in nasa_tools:
        return "NASA metadata tool is currently unavailable."

    clean_id = nasa_id.strip()
    if not clean_id:
        return "Please provide a NASA media ID."

    try:
        raw_result = nasa_tools["get_media_metadata_location"].invoke(clean_id)
        result = _format_nasa_location(str(raw_result), "metadata", clean_id)
        _capture_recent_result("nasa_media_metadata", result)
        return result
    except Exception as exc:
        logger.error(
            f"NASA metadata lookup failed for '{clean_id}': {exc}",
            exc_info=True,
        )
        return f"Error fetching NASA metadata location: {str(exc)}"


@tool
def nasa_video_captions(nasa_id: str) -> str:
    """
    Get NASA video captions location URL for a NASA media ID.

    Args:
        nasa_id: NASA video media identifier

    Returns:
        URL to captions for the media item
    """
    nasa_tools = _init_nasa_tools()
    if not nasa_tools or "get_video_captions_location" not in nasa_tools:
        return "NASA captions tool is currently unavailable."

    clean_id = nasa_id.strip()
    if not clean_id:
        return "Please provide a NASA media ID."

    try:
        raw_result = nasa_tools["get_video_captions_location"].invoke(clean_id)
        result = _format_nasa_location(str(raw_result), "captions", clean_id)
        _capture_recent_result("nasa_video_captions", result)
        return result
    except Exception as exc:
        logger.error(
            f"NASA captions lookup failed for '{clean_id}': {exc}",
            exc_info=True,
        )
        return f"Error fetching NASA captions location: {str(exc)}"


@tool
def read_website(url: str) -> str:
    """
    Read and extract the text content from a specific webpage URL.
    Use this after search_web to get in-depth information from a specific article/page.

    Args:
        url: The full URL to read (e.g., 'https://en.wikipedia.org/wiki/Tsunami')

    Returns:
        The extracted text content of the webpage, up to 15,000 characters.
    """
    try:
        logger.debug(f"Reading website: {url}")
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            )
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
            script.extract()

        text = soup.get_text(separator=" ", strip=True)

        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = "\n".join(chunk for chunk in chunks if chunk)

        logger.info(f"Successfully extracted {len(text)} characters from {url}")

        if len(text) > 15000:
            text = text[:15000] + "... [Content truncated due to length]"

        _capture_recent_result("read_website", text)

        return (
            text
            if text
            else "The page was successfully fetched but no readable text was found."
        )

    except requests.exceptions.Timeout:
        return f"Error: Request timed out while trying to fetch {url}"
    except requests.exceptions.RequestException as exc:
        return f"Error fetching website {url}: {str(exc)}"
    except Exception as exc:
        logger.error(f"Unexpected error reading {url}: {exc}", exc_info=True)
        return f"Error processing website: {str(exc)}"


def finance_news_tool() -> List[Any]:
    """Return the Yahoo Finance News Tool."""
    try:
        return [YahooFinanceNewsTool()]
    except Exception as exc:
        logger.warning(f"Yahoo Finance tool unavailable: {exc}")
        return []


def yt_search_tool() -> List[Any]:
    """Return the YouTube Search Tool."""
    try:
        return [YouTubeSearchTool()]
    except Exception as exc:
        logger.warning(f"YouTube search tool unavailable: {exc}")
        return []


def get_info_tools() -> List[Any]:
    """
    Get all info tools as a list for agent registration.

    Returns:
        List of LangChain tools
    """
    tools: List[Any] = [
        get_weather,
        get_news,
        search_web,
        wikipedia_search,
        wikidata_search,
        stackexchange_search,
        nasa_media_search,
        nasa_media_manifest,
        nasa_media_metadata,
        nasa_video_captions,
        read_website,
        recent_context,
    ]
    tools.extend(finance_news_tool())
    tools.extend(yt_search_tool())
    return tools
