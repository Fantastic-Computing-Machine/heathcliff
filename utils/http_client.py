# ABOUTME: Stealth HTTP client with randomized User-Agent, Referer, and headers
# ABOUTME: Retries with exponential backoff on 403/429/503 using fresh identities

from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from logger import logger

# ---------------------------------------------------------------------------
# User-Agent pool — current, diverse browser strings
# ---------------------------------------------------------------------------
_USER_AGENTS: List[str] = [
    # Chrome – Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    # Chrome – macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    # Chrome – Linux
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    # Firefox – Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
    # Firefox – macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:125.0) Gecko/20100101 Firefox/125.0",
    # Firefox – Linux
    "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
    # Safari – macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15",
    # Edge – Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0",
    # Edge – macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0",
]

# ---------------------------------------------------------------------------
# Referer pool — plausible origins a real user might come from
# ---------------------------------------------------------------------------
_REFERERS: List[Optional[str]] = [
    "https://www.google.com/",
    "https://www.google.com/search?q=",
    "https://duckduckgo.com/",
    "https://www.bing.com/search?q=",
    "https://search.yahoo.com/",
    "https://www.reddit.com/",
    "https://news.ycombinator.com/",
    "https://t.co/",
    None,  # some requests legitimately have no referrer
    None,
]

# ---------------------------------------------------------------------------
# Accept-Language variants
# ---------------------------------------------------------------------------
_ACCEPT_LANGUAGES: List[str] = [
    "en-US,en;q=0.9",
    "en-US,en;q=0.9,es;q=0.8",
    "en-GB,en;q=0.9,en-US;q=0.8",
    "en-US,en;q=0.9,fr;q=0.7",
    "en-US,en;q=0.9,de;q=0.7",
    "en,en-US;q=0.9",
]

# Status codes that should trigger a retry with a fresh identity
_RETRYABLE_STATUS_CODES = frozenset({403, 429, 503})


class StealthHttpClient:
    """HTTP client that randomizes identity headers per-request.

    Each call to ``get()`` picks a fresh User-Agent, Referer, and
    Accept-Language to minimise fingerprinting. On 403/429/503 responses
    it retries with a completely new identity up to ``max_retries`` times
    with exponential back-off.

    Args:
        max_retries: Number of identity-swap retries on rejection.
        base_timeout: Default request timeout in seconds.
        backoff_factor: Multiplier for exponential back-off sleep.
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_timeout: int = 20,
        backoff_factor: float = 1.0,
    ) -> None:
        self.max_retries = max_retries
        self.base_timeout = base_timeout
        self.backoff_factor = backoff_factor

    # ---- public API -------------------------------------------------------

    def get(
        self,
        url: str,
        *,
        timeout: Optional[int] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> requests.Response:
        """Fetch *url* with a randomised identity, retrying on rejection.

        Args:
            url: URL to GET.
            timeout: Per-request timeout (falls back to ``base_timeout``).
            extra_headers: Additional headers merged on top of generated ones.

        Returns:
            The final ``requests.Response`` (may still be non-200 if all
            retries are exhausted — caller should check ``.status_code``).

        Raises:
            requests.exceptions.RequestException: On network-level failures
            after all retries.
        """
        timeout = timeout or self.base_timeout
        last_exc: Optional[Exception] = None
        last_resp: Optional[requests.Response] = None

        for attempt in range(1 + self.max_retries):
            headers = self._build_headers(extra_headers)
            try:
                session = self._build_session()
                resp = session.get(url, headers=headers, timeout=timeout, verify=True)

                if resp.status_code not in _RETRYABLE_STATUS_CODES:
                    return resp

                # Retryable status — log and sleep before next attempt
                last_resp = resp
                logger.debug(
                    "Stealth GET %s returned %d (attempt %d/%d)",
                    url,
                    resp.status_code,
                    attempt + 1,
                    1 + self.max_retries,
                )

            except requests.exceptions.RequestException as exc:
                last_exc = exc
                logger.debug(
                    "Stealth GET %s failed (attempt %d/%d): %s",
                    url,
                    attempt + 1,
                    1 + self.max_retries,
                    exc,
                )

            if attempt < self.max_retries:
                sleep_time = self.backoff_factor * (2**attempt) + random.uniform(0, 0.5)
                time.sleep(sleep_time)

        # All retries exhausted — return whatever we have
        if last_resp is not None:
            return last_resp
        if last_exc is not None:
            raise last_exc
        # Should never get here, but satisfy the type checker
        raise requests.exceptions.ConnectionError(  # pragma: no cover
            f"All {self.max_retries} retries exhausted for {url}"
        )

    def random_headers(
        self,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """Return a set of randomised browser-like headers.

        Useful for passing to Playwright browser contexts or other HTTP
        libraries that don't use ``requests``.
        """
        return self._build_headers(extra_headers)

    def random_user_agent(self) -> str:
        """Return a randomly selected User-Agent string."""
        return random.choice(_USER_AGENTS)

    # ---- internals --------------------------------------------------------

    def _build_headers(
        self,
        extra: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """Assemble a plausible, randomised header set."""
        ua = random.choice(_USER_AGENTS)
        referer = random.choice(_REFERERS)

        headers: Dict[str, str] = {
            "User-Agent": ua,
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;"
                "q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8"
            ),
            "Accept-Language": random.choice(_ACCEPT_LANGUAGES),
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none" if referer is None else "cross-site",
            "Sec-Fetch-User": "?1",
        }

        # Add Sec-CH-UA hints for Chrome/Edge UAs
        if "Chrome" in ua and "Firefox" not in ua and "Safari/605" not in ua:
            # Extract major version from UA string
            chrome_ver = "124"
            for part in ua.split("Chrome/"):
                if len(part) > 1:
                    chrome_ver = part.split(".")[0]
                    break
            brand = "Google Chrome"
            if "Edg/" in ua:
                brand = "Microsoft Edge"
            headers["Sec-CH-UA"] = (
                f'"{brand}";v="{chrome_ver}", '
                f'"Chromium";v="{chrome_ver}", '
                f'"Not_A Brand";v="8"'
            )
            headers["Sec-CH-UA-Mobile"] = "?0"
            # Pick a platform consistent with the UA
            if "Windows" in ua:
                headers["Sec-CH-UA-Platform"] = '"Windows"'
            elif "Macintosh" in ua:
                headers["Sec-CH-UA-Platform"] = '"macOS"'
            else:
                headers["Sec-CH-UA-Platform"] = '"Linux"'

        if referer is not None:
            headers["Referer"] = referer

        if extra:
            headers.update(extra)

        return headers

    @staticmethod
    def _build_session() -> requests.Session:
        """Create a ``requests.Session`` with connection-level retries."""
        session = requests.Session()
        adapter = HTTPAdapter(
            max_retries=Retry(
                total=2,
                backoff_factor=0.3,
                status_forcelist=[500, 502, 504],
                allowed_methods=["GET", "HEAD"],
            )
        )
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session
