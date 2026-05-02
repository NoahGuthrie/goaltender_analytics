"""
Centralized NHL API HTTP client.

Provides a single, shared HTTP client with:
- Connection pooling via requests.Session
- Rate limiting (1 request/second by default)
- Exponential backoff with jitter on 429/5xx errors (up to 5 retries)
- Configurable timeouts
- Polite User-Agent identification

All ingestion scrapers should use this client rather than making raw requests.
"""

import logging
import random
import time

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_WEB = "https://api-web.nhle.com/v1"
BASE_STATS = "https://api.nhle.com/stats/rest/en"

_DEFAULT_RATE_LIMIT = 1.0  # minimum seconds between consecutive requests
_DEFAULT_MAX_RETRIES = 5
_DEFAULT_CONNECT_TIMEOUT = 30  # seconds
_DEFAULT_READ_TIMEOUT = 60  # seconds
_USER_AGENT = "GoaltenderAnalytics/0.1 (research)"

# HTTP status codes that trigger a retry
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class NHLAPIClient:
    """Rate-limited, retry-capable HTTP client for the NHL API.

    Parameters
    ----------
    rate_limit : float
        Minimum seconds between consecutive requests.  Default 1.0.
    max_retries : int
        Maximum retry attempts on transient errors.  Default 5.
    connect_timeout : float
        TCP connect timeout in seconds.
    read_timeout : float
        Socket read timeout in seconds.
    """

    def __init__(
        self,
        rate_limit: float = _DEFAULT_RATE_LIMIT,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        connect_timeout: float = _DEFAULT_CONNECT_TIMEOUT,
        read_timeout: float = _DEFAULT_READ_TIMEOUT,
    ) -> None:
        self.rate_limit = rate_limit
        self.max_retries = max_retries
        self.timeout = (connect_timeout, read_timeout)

        self._session = requests.Session()
        self._session.headers.update({"User-Agent": _USER_AGENT})
        self._last_request_time: float = 0.0

    # -- public helpers -----------------------------------------------------

    def get_schedule(self, date: str) -> dict:
        """Fetch the schedule page for a given date (YYYY-MM-DD).

        Returns the full JSON response including ``gameWeek``,
        ``nextStartDate``, season boundaries, etc.
        """
        url = f"{BASE_WEB}/schedule/{date}"
        return self._get_json(url)

    def get_pbp(self, game_id: int) -> dict:
        """Fetch play-by-play data for a single game."""
        url = f"{BASE_WEB}/gamecenter/{game_id}/play-by-play"
        return self._get_json(url)

    def get_shifts(self, game_id: int) -> dict:
        """Fetch shift chart data for a single game."""
        url = f"{BASE_STATS}/shiftcharts?cayenneExp=gameId={game_id}"
        return self._get_json(url)

    def close(self) -> None:
        """Close the underlying ``requests.Session``."""
        self._session.close()

    # -- context manager ----------------------------------------------------

    def __enter__(self) -> "NHLAPIClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- internals ----------------------------------------------------------

    def _throttle(self) -> None:
        """Enforce the minimum inter-request gap."""
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self.rate_limit:
            sleep_time = self.rate_limit - elapsed
            logger.debug("Rate-limiter sleeping %.2fs", sleep_time)
            time.sleep(sleep_time)

    def _get_json(self, url: str) -> dict:
        """GET *url*, returning parsed JSON.

        Retries on transient HTTP errors with exponential backoff + jitter.
        """
        for attempt in range(1, self.max_retries + 1):
            self._throttle()
            try:
                logger.debug("GET %s (attempt %d/%d)", url, attempt, self.max_retries)
                self._last_request_time = time.monotonic()
                resp = self._session.get(url, timeout=self.timeout)

                if resp.status_code == 200:
                    return resp.json()

                if resp.status_code in _RETRYABLE_STATUS_CODES:
                    backoff = self._backoff_seconds(attempt)
                    logger.warning(
                        "HTTP %d from %s — retrying in %.1fs (attempt %d/%d)",
                        resp.status_code,
                        url,
                        backoff,
                        attempt,
                        self.max_retries,
                    )
                    time.sleep(backoff)
                    continue

                # Non-retryable HTTP error
                resp.raise_for_status()

            except requests.exceptions.ConnectionError as exc:
                backoff = self._backoff_seconds(attempt)
                logger.warning(
                    "Connection error for %s — retrying in %.1fs (attempt %d/%d): %s",
                    url,
                    backoff,
                    attempt,
                    self.max_retries,
                    exc,
                )
                time.sleep(backoff)

            except requests.exceptions.Timeout as exc:
                backoff = self._backoff_seconds(attempt)
                logger.warning(
                    "Timeout for %s — retrying in %.1fs (attempt %d/%d): %s",
                    url,
                    backoff,
                    attempt,
                    self.max_retries,
                    exc,
                )
                time.sleep(backoff)

        # Exhausted all retries
        msg = f"Failed to fetch {url} after {self.max_retries} attempts"
        logger.error(msg)
        raise requests.exceptions.RetryError(msg)

    @staticmethod
    def _backoff_seconds(attempt: int) -> float:
        """Exponential backoff with full jitter: ``uniform(0, 2^attempt)``."""
        ceiling = min(2**attempt, 64)  # cap at 64 seconds
        return random.uniform(0, ceiling)
