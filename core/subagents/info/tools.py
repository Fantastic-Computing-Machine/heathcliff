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
from langchain_community.tools import YouTubeSearchTool
from langchain_community.tools.ddg_search.tool import (
    DuckDuckGoSearchResults as DuckDuckGoSearchTool,
)
from langchain_community.tools.google_search import GoogleSearchResults
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain_community.utilities import (
    GoogleSearchAPIWrapper,
    OpenWeatherMapAPIWrapper,
)

from config import Config
from core.subagents.info.recent_context import _capture_recent_result, recent_context
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

    if config.google_search_api_key and not os.getenv("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = config.google_search_api_key

    if config.google_search_cse_id and not os.getenv("GOOGLE_CSE_ID"):
        os.environ["GOOGLE_CSE_ID"] = config.google_search_cse_id


def _init_google_tool(max_results: int) -> Optional[Any]:
    """Lazily instantiate the Google search tool."""

    global _google_tool

    if _google_tool is None:
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

        # The wrapper uses os.environ for the API key
        if "OPENWEATHERMAP_API_KEY" not in os.environ:
            os.environ["OPENWEATHERMAP_API_KEY"] = api_key

        if location is None:
            logger.debug("No location provided for weather; using default from config")
            location = Config.DEFAULT_CITY

        logger.debug(f"Fetching weather for location: {location}")

        # The LangChain wrapper abstracts away the url and parameters
        weather_wrapper = OpenWeatherMapAPIWrapper()

        # It returns a string containing weather information
        weather_data = weather_wrapper.run(location)
        _capture_recent_result("get_weather", weather_data)

        logger.info(f"Weather retrieved for {location} using OpenWeatherMapAPIWrapper")
        return weather_data

    except Exception as e:
        logger.error(
            f"Error fetching weather with wrapper for {location}: {e}", exc_info=True
        )
        return f"Error fetching weather data: {str(e)}"


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

        result = "\n\n".join(news_list)
        _capture_recent_result("get_news", result)
        return result

    except requests.exceptions.RequestException as e:
        return f"Error fetching news: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"


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
        # Use DuckDuckGo as primary search, fallback gracefully
        primary_provider = provider or "duckduckgo"
        response = _dispatch_search(primary_provider, query, max_results, config)

        if response:
            _capture_recent_result("search_web", response)
            return response

        # Automatic chained fallback
        fallbacks = ["google"]
        for fallback in fallbacks:
            if fallback != primary_provider:
                fb_resp = _dispatch_search(fallback, query, max_results, config)
                if fb_resp:
                    _capture_recent_result("search_web", fb_resp)
                    return fb_resp

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
            result = "\n\n".join(summaries)
            _capture_recent_result("search_web", result)
            return result

        return f"No search results found for: {query}"

    except Exception as e:
        return f"Error searching web: {str(e)}"


@tool
def wikipedia_search(query: str) -> str:
    """
    Search Wikipedia for information. Use this for factual queries.

    Args:
        query: Wikipedia search query

    Returns:
        Full Wikipedia article content
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
                logger.debug(f"Attempting to fetch full article for: {result_title}")
                page = wikipedia.page(result_title, auto_suggest=False)
                logger.info(
                    f"Successfully retrieved Wikipedia article for: {result_title}"
                )
                result = f"{result_title} (Full Text):\n{page.content}"
                _capture_recent_result("wikipedia_search", result)
                return result
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
                        page = wikipedia.page(e.options[0], auto_suggest=False)
                        logger.info(
                            f"Retrieved disambiguated full article for: {e.options[0]}"
                        )
                        result = f"{e.options[0]} (Full Text):\n{page.content}"
                        _capture_recent_result("wikipedia_search", result)
                        return result
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
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
            script.extract()

        # Get text
        text = soup.get_text(separator=" ", strip=True)

        # Clean up excessive whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = "\n".join(chunk for chunk in chunks if chunk)

        logger.info(f"Successfully extracted {len(text)} characters from {url}")

        # Guard against massive pages choking the context window
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
    except requests.exceptions.RequestException as e:
        return f"Error fetching website {url}: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error reading {url}: {e}", exc_info=True)
        return f"Error processing website: {str(e)}"


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
        read_website,
        recent_context,
    ]
    tools.extend(finance_news_tool())
    tools.extend(yt_search_tool())
    return tools
