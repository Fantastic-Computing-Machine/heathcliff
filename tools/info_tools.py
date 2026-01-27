# ABOUTME: Information retrieval tools for weather, news, web search, and Wikipedia
# ABOUTME: Integrates OpenWeatherMap, NewsAPI, LangChain search tools, and Wikipedia APIs

from __future__ import annotations

import inspect
import json
import os
from typing import Any, Dict, List, Optional

import requests
import wikipedia
from bs4 import BeautifulSoup
from langchain.tools import tool
from playwright.sync_api import TimeoutError as PlaywrightTimeout
from playwright.sync_api import sync_playwright

from config import Config
from logger import logger

_google_tool: Optional[Any] = None
_duck_tool: Optional[Any] = None


def _filter_kwargs_for_cls(cls: Any, candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Filter keyword arguments so we only pass parameters the class expects."""

    if cls is None:
        return {}

    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return {}

    return {k: v for k, v in candidate.items() if k in sig.parameters}


def _ensure_google_search_env(config) -> None:
    """Make sure Google Custom Search keys are available to LangChain."""

    if config.GOOGLE_CSE_API_KEY and not os.getenv("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = config.GOOGLE_CSE_API_KEY

    if config.GOOGLE_CSE_ID and not os.getenv("GOOGLE_CSE_ID"):
        os.environ["GOOGLE_CSE_ID"] = config.GOOGLE_CSE_ID


def _init_google_tool(max_results: int) -> Optional[Any]:
    """Lazily instantiate the Google search tool."""

    global _google_tool

    if _google_tool is None:
        try:
            from langchain_community.tools.google_search import GoogleSearchResults
            from langchain_community.utilities import GoogleSearchAPIWrapper
        except ImportError:  # pragma: no cover - optional dependency
            logger.warning("Google search dependencies not installed")
            return None

        try:
            # Create the API wrapper with environment credentials
            api_wrapper = GoogleSearchAPIWrapper()

            # Pass the wrapper to GoogleSearchResults
            _google_tool = GoogleSearchResults(api_wrapper=api_wrapper)
            logger.info("Google search tool initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Google search: {e}", exc_info=True)
            return None

    return _google_tool


def _init_duck_tool() -> Optional[Any]:
    """Lazily instantiate the DuckDuckGo search tool."""

    global _duck_tool

    if _duck_tool is None:
        try:
            from langchain_community.tools.ddg_search.tool import (
                DuckDuckGoSearchRun as DuckDuckGoSearchTool,
            )

        except ImportError:  # pragma: no cover - optional dependency
            try:
                from langchain_community.tools.ddg_search.tool import (
                    DuckDuckGoSearchResults as DuckDuckGoSearchTool,
                )
            except ImportError:
                return None

        _duck_tool = DuckDuckGoSearchTool()

    return _duck_tool


def _format_search_result(result: Any) -> str:
    """Normalize LangChain search responses into a human-friendly string."""

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


def _dispatch_search(provider: str, query: str, max_results: int, config) -> str:
    """Execute the configured search provider via LangChain tools."""

    provider = (provider or "").lower()

    if provider == "google":
        _ensure_google_search_env(config)
        tool = _init_google_tool(max_results)
        if not tool:
            return ""
        try:
            return _format_search_result(tool.run(query))
        except Exception as exc:  # pragma: no cover - network errors
            return f"Google search error: {exc}"

    if provider == "duckduckgo":
        tool = _init_duck_tool()
        if not tool:
            return ""
        try:
            return _format_search_result(tool.run(query))
        except Exception as exc:  # pragma: no cover - network errors
            return f"DuckDuckGo search error: {exc}"

    return ""


@tool
def get_weather(location: str | None = None) -> str:
    """
    Get current weather for a location.

    If user asks about weather "around me", "here", or without specifying a location,
    call this with NO location argument - it will automatically use the user's
    configured home location.

    Only pass a location if user explicitly asks for weather in a DIFFERENT city.

    Args:
        location: Optional city name. Leave empty/None for user's default location.

    Returns:
        Weather description with temperature, conditions, and humidity
    """
    try:
        config = Config
        api_key = config.OPENWEATHERMAP_API_KEY

        if not api_key:
            return "Weather API key not configured"

        if location is None:
            logger.debug("No location provided for weather; using default from config")
            location = config.DEFAULT_CITY

        logger.debug(f"Fetching weather for location: {location}")

        units = config.UNITS
        temp_unit = "°C" if units == "metric" else "°F"

        url = f"http://api.openweathermap.org/data/2.5/weather"
        params = {"q": location, "appid": api_key, "units": units}

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Get actual city name from API response
        city_name = data.get("name", location)
        temp = data["main"]["temp"]
        feels_like = data["main"]["feels_like"]
        humidity = data["main"]["humidity"]
        description = data["weather"][0]["description"]

        logger.info(
            f"Weather retrieved for {city_name}: {description}, {temp}{temp_unit}"
        )

        return (
            f"Weather in {city_name}: {description.capitalize()}\n"
            f"Temperature: {temp}{temp_unit} (feels like {feels_like}{temp_unit})\n"
            f"Humidity: {humidity}%"
        )

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching weather for {location}: {e}", exc_info=True)
        return f"Error fetching weather data: {str(e)}"
    except KeyError as e:
        logger.error(f"Unexpected weather API response format: {e}", exc_info=True)
        return f"Error parsing weather data for {location}"
    except Exception as e:
        logger.error(f"Unexpected error in get_weather: {e}", exc_info=True)
        return f"Error: {str(e)}"


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

        sources = config.DEFAULT_SOURCES
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

        return "\n\n".join(news_list)

    except requests.exceptions.RequestException as e:
        return f"Error fetching news: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def search_web(query: str, provider: Optional[str] = None) -> str:
    """
    Search the web for current information.

    IMPORTANT: Use COMPLETE search queries with all relevant keywords.
    Examples:
    - "Y Combinator Winter 2025 batch startups" ✓
    - "latest AI developments January 2026" ✓
    - "latest startups" ✗ (too vague)

    Args:
        query: Full search query with all keywords
        provider: Override search provider (google or duckduckgo)

    Returns:
        Search results summary
    """
    try:
        config = Config
        max_results = 5
        primary_provider = provider or "google"
        response = _dispatch_search(primary_provider, query, max_results, config)

        if response:
            return response

        fallback_provider = "duckduckgo"
        if fallback_provider and fallback_provider != primary_provider:
            fallback_response = _dispatch_search(
                fallback_provider, query, max_results, config
            )
            if fallback_response:
                return fallback_response

        # Final fallback: Wikipedia summary search
        wiki_results = wikipedia.search(query, results=3)
        if not wiki_results:
            return f"No search results found for: {query}"

        summaries = []
        for result in wiki_results:
            try:
                summary = wikipedia.summary(result, sentences=2)
                summaries.append(f"{result}:\n{summary}")
            except Exception:
                continue

        if summaries:
            return "\n\n".join(summaries)

        return f"No search results found for: {query}"

    except Exception as e:
        return f"Error searching web: {str(e)}"


@tool
def wikipedia_search(query: str) -> str:
    """
    Search Wikipedia for factual information about topics, people, places, concepts.

    IMPORTANT: Use the COMPLETE topic name for accurate results.
    Examples:
    - "Mount Fuji" ✓
    - "Albert Einstein physicist" ✓
    - "Mount" ✗ (incomplete)

    Args:
        query: Complete topic or entity name

    Returns:
        Wikipedia article summary
    """
    try:
        logger.debug(f"Searching Wikipedia for: {query}")

        # Search for the query
        results = wikipedia.search(query, results=3)  # Get top 3 results

        if not results:
            logger.warning(f"No Wikipedia search results for: {query}")
            return f"No Wikipedia articles found for: {query}"

        logger.debug(f"Wikipedia search results: {results}")

        # Try each result until we find one that works
        for result_title in results:
            try:
                logger.debug(f"Attempting to fetch summary for: {result_title}")
                summary = wikipedia.summary(
                    result_title, sentences=5, auto_suggest=False
                )
                logger.info(
                    f"Successfully retrieved Wikipedia summary for: {result_title}"
                )
                return f"{result_title}:\n{summary}"
            except wikipedia.exceptions.PageError:
                logger.debug(f"PageError for {result_title}, trying next result")
                continue
            except wikipedia.exceptions.DisambiguationError as e:
                logger.debug(
                    f"DisambiguationError for {result_title}, trying first option"
                )
                # Try the first disambiguation option
                if e.options:
                    try:
                        summary = wikipedia.summary(
                            e.options[0], sentences=5, auto_suggest=False
                        )
                        logger.info(
                            f"Retrieved disambiguated summary for: {e.options[0]}"
                        )
                        return f"{e.options[0]}:\n{summary}"
                    except:
                        continue

        # If we exhausted all results
        logger.warning(
            f"Failed to retrieve summary for any Wikipedia result for: {query}"
        )
        return f"No Wikipedia page found for: {query}"

    except wikipedia.exceptions.DisambiguationError as e:
        # Handle top-level disambiguation
        options = ", ".join(e.options[:5])
        logger.debug(f"Top-level disambiguation for '{query}': {options}")
        return f"Multiple results found for '{query}'. Please be more specific. Options: {options}"
    except Exception as e:
        logger.error(
            f"Unexpected error searching Wikipedia for '{query}': {e}", exc_info=True
        )
        return f"Error searching Wikipedia: {str(e)}"


@tool
def fetch_webpage(url: str, max_length: int = 8000) -> str:
    """
    Fetch and extract detailed text content from any webpage URL, including company directories,
    startup lists, and article pages. Use this whenever you have a URL and need to read its full content.
    This is MUCH better than search_web when you already know the URL.

    Examples of when to use:
    - User wants companies from https://www.ycombinator.com/companies
    - User wants to read a specific article or blog post
    - User provides a direct link and asks for information from it
    - You found a URL via search and need to extract data from it

    Args:
        url: The full URL to fetch content from
        max_length: Maximum character length to return (default 8000)

    Returns:
        Extracted text content from the webpage
    """
    try:
        logger.debug(f"Fetching webpage: {url}")

        # Add headers to avoid being blocked
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        response = requests.get(url, headers=headers, timeout=15, verify=True)
        response.raise_for_status()

        # Use lxml parser which is more stable than html.parser
        # Fallback to html.parser if lxml not available
        try:
            soup = BeautifulSoup(response.content, "lxml")
        except:
            soup = BeautifulSoup(response.content, "html.parser")

        # Remove script and style elements safely
        try:
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
        except:
            pass  # Continue even if removal fails

        # Get text content
        text = soup.get_text(separator="\n", strip=True)

        # Clean up whitespace
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text = "\n".join(lines)

        # Truncate if too long
        if len(text) > max_length:
            text = (
                text[:max_length]
                + f"\n\n[Content truncated to {max_length} characters]"
            )

        logger.info(f"Successfully fetched {len(text)} characters from {url}")
        return text

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching webpage {url}: {e}", exc_info=True)
        return f"Error fetching webpage: {str(e)}"
    except MemoryError as e:
        logger.error(f"Memory error parsing {url}: {e}", exc_info=True)
        return f"Webpage too large to parse: {url}"
    except Exception as e:
        logger.error(f"Unexpected error fetching {url}: {e}", exc_info=True)
        return f"Error: {str(e)}"


@tool
def fetch_dynamic_webpage(
    url: str, wait_seconds: int = 5, max_length: int = 30000
) -> str:
    """
    Fetch FULL content from ANY webpage URL using a real browser. Works for both static and
    JavaScript-rendered pages (React/Vue/Angular). Returns complete rendered text that you
    MUST parse yourself to extract data, companies, tables, or structured information.

    Use this tool whenever user asks to:
    - Get companies from a URL (e.g., YC directory, startup lists)
    - Fetch content from any webpage
    - Read articles, blog posts, or documentation
    - Extract data from websites

    After fetching, YOU MUST parse the returned text to create:
    - Tables with company names, descriptions, links
    - Lists of items extracted from the page
    - Structured data in the format user requested

    Args:
        url: The URL to fetch (any webpage - static or dynamic)
        wait_seconds: Seconds to wait for page to fully load (default 5)
        max_length: Maximum text length to return (default 30000 chars)

    Returns:
        Full rendered page text - YOU extract and format the data from this
    """
    try:
        logger.info(f"Launching browser to fetch: {url}")

        with sync_playwright() as p:
            # Launch headless browser
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            # Navigate and wait for content
            page.goto(url, wait_until="networkidle", timeout=30000)

            # Wait additional time for JavaScript to render
            page.wait_for_timeout(wait_seconds * 1000)

            # Scroll down to trigger lazy loading (e.g., YC directory)
            # Scroll in increments to trigger infinite scroll
            for _ in range(5):
                page.evaluate("window.scrollBy(0, window.innerHeight)")
                page.wait_for_timeout(500)  # Wait 500ms between scrolls

            # Scroll back to top to capture all content
            page.evaluate("window.scrollTo(0, 0)")
            page.wait_for_timeout(1000)

            # Get rendered content
            content = page.content()
            browser.close()

        # Parse with BeautifulSoup
        soup = BeautifulSoup(content, "lxml")

        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()

        # Get text
        text = soup.get_text(separator="\n", strip=True)

        # Clean up whitespace
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text = "\n".join(lines)

        # Truncate if too long
        if len(text) > max_length:
            text = (
                text[:max_length]
                + f"\n\n[Content truncated to {max_length} characters]"
            )

        logger.info(
            f"Successfully fetched {len(text)} characters from {url} using browser"
        )
        return text

    except PlaywrightTimeout as e:
        logger.error(f"Timeout fetching {url}: {e}", exc_info=True)
        return f"Page took too long to load: {url}"
    except Exception as e:
        logger.error(f"Error with browser automation for {url}: {e}", exc_info=True)
        return f"Browser automation error: {str(e)}"


def get_info_tools() -> List[Any]:
    """
    Get all info tools as a list for agent registration.

    Returns:
        List of LangChain tools
    """
    # Removed fetch_webpage - use only fetch_dynamic_webpage for all URL fetching
    # fetch_dynamic_webpage handles both static and JS-rendered pages
    return [get_weather, get_news, search_web, wikipedia_search, fetch_dynamic_webpage]
