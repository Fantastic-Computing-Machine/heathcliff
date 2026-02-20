# ABOUTME: Information retrieval tools for weather, news, web search, and Wikipedia
# ABOUTME: Integrates OpenWeatherMap, NewsAPI, LangChain search tools, and Wikipedia APIs

from __future__ import annotations

import inspect
import json
import os
from typing import Any, Dict, List, Optional

import requests
import wikipedia
from langchain.tools import tool

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

    if config.google_search_api_key and not os.getenv("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = config.google_search_api_key

    if config.google_search_cse_id and not os.getenv("GOOGLE_CSE_ID"):
        os.environ["GOOGLE_CSE_ID"] = config.google_search_cse_id


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
    Get current weather for a location. Use this when user asks about weather.

    Args:
        location: City or location name (if None, uses default from config)

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

        logger.info(f"Weather retrieved for {city_name}: {description}, {temp}{temp_unit}")

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
    Search Wikipedia for information. Use this for factual queries.

    Args:
        query: Wikipedia search query

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
                summary = wikipedia.summary(result_title, sentences=5, auto_suggest=False)
                logger.info(f"Successfully retrieved Wikipedia summary for: {result_title}")
                return f"{result_title}:\n{summary}"
            except wikipedia.exceptions.PageError:
                logger.debug(f"PageError for {result_title}, trying next result")
                continue
            except wikipedia.exceptions.DisambiguationError as e:
                logger.debug(f"DisambiguationError for {result_title}, trying first option")
                # Try the first disambiguation option
                if e.options:
                    try:
                        summary = wikipedia.summary(e.options[0], sentences=5, auto_suggest=False)
                        logger.info(f"Retrieved disambiguated summary for: {e.options[0]}")
                        return f"{e.options[0]}:\n{summary}"
                    except:
                        continue

        # If we exhausted all results
        logger.warning(f"Failed to retrieve summary for any Wikipedia result for: {query}")
        return f"No Wikipedia page found for: {query}"

    except wikipedia.exceptions.DisambiguationError as e:
        # Handle top-level disambiguation
        options = ", ".join(e.options[:5])
        logger.debug(f"Top-level disambiguation for '{query}': {options}")
        return f"Multiple results found for '{query}'. Please be more specific. Options: {options}"
    except Exception as e:
        logger.error(f"Unexpected error searching Wikipedia for '{query}': {e}", exc_info=True)
        return f"Error searching Wikipedia: {str(e)}"


def get_info_tools() -> List[Any]:
    """
    Get all info tools as a list for agent registration.

    Returns:
        List of LangChain tools
    """
    return [get_weather, get_news, search_web, wikipedia_search]
