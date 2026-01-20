"""Adaptive rate limiter with header parsing and exponential backoff."""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger()


@dataclass
class RateLimitState:
    """Tracks rate limit state for a specific API."""

    requests_made: int = 0
    window_start: float = field(default_factory=time.monotonic)
    window_seconds: float = 60.0

    # Discovered limits (learned from headers or 429 responses)
    discovered_limit: int | None = None
    discovered_remaining: int | None = None
    discovered_reset_time: float | None = None

    # Backoff state
    consecutive_429s: int = 0
    last_429_time: float | None = None
    current_backoff_seconds: float = 1.0

    # Default conservative limits (used when no headers available)
    default_requests_per_minute: int = 30

    def reset_window_if_expired(self) -> None:
        """Reset the request window if it has expired."""
        now = time.monotonic()
        if now - self.window_start >= self.window_seconds:
            self.requests_made = 0
            self.window_start = now

    def record_request(self) -> None:
        """Record that a request was made."""
        self.reset_window_if_expired()
        self.requests_made += 1

    def should_wait(self) -> tuple[bool, float]:
        """Check if we should wait before making a request.

        Returns:
            Tuple of (should_wait, wait_seconds)
        """
        now = time.monotonic()

        # Check if we're in a backoff period from a 429
        if self.last_429_time and self.current_backoff_seconds > 0:
            elapsed = now - self.last_429_time
            if elapsed < self.current_backoff_seconds:
                return True, self.current_backoff_seconds - elapsed

        # Check discovered reset time
        if self.discovered_reset_time and now < self.discovered_reset_time:
            if self.discovered_remaining is not None and self.discovered_remaining <= 0:
                return True, self.discovered_reset_time - now

        # Check our own accounting against discovered or default limits
        self.reset_window_if_expired()
        limit = self.discovered_limit or self.default_requests_per_minute

        if self.requests_made >= limit:
            wait_time = self.window_seconds - (now - self.window_start)
            if wait_time > 0:
                return True, wait_time

        return False, 0.0

    def update_from_headers(self, headers: dict[str, str]) -> None:
        """Update state from rate limit response headers.

        Handles common header formats:
        - X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset
        - RateLimit-Limit, RateLimit-Remaining, RateLimit-Reset
        - Retry-After
        """
        # Try different header naming conventions
        limit_keys = ["X-RateLimit-Limit", "RateLimit-Limit", "x-ratelimit-limit"]
        remaining_keys = ["X-RateLimit-Remaining", "RateLimit-Remaining", "x-ratelimit-remaining"]
        reset_keys = ["X-RateLimit-Reset", "RateLimit-Reset", "x-ratelimit-reset"]

        # Case-insensitive header lookup
        lower_headers = {k.lower(): v for k, v in headers.items()}

        for key in limit_keys:
            if key.lower() in lower_headers:
                try:
                    self.discovered_limit = int(lower_headers[key.lower()])
                    logger.debug("discovered_rate_limit", limit=self.discovered_limit)
                except ValueError:
                    pass
                break

        for key in remaining_keys:
            if key.lower() in lower_headers:
                try:
                    self.discovered_remaining = int(lower_headers[key.lower()])
                except ValueError:
                    pass
                break

        for key in reset_keys:
            if key.lower() in lower_headers:
                try:
                    reset_value = lower_headers[key.lower()]
                    # Could be Unix timestamp or seconds until reset
                    reset_int = int(reset_value)
                    if reset_int > 1000000000:  # Looks like Unix timestamp
                        self.discovered_reset_time = float(reset_int)
                    else:  # Seconds until reset
                        self.discovered_reset_time = time.time() + reset_int
                except ValueError:
                    pass
                break

    def handle_429(self, headers: dict[str, str] | None = None) -> float:
        """Handle a 429 response and return wait time.

        Args:
            headers: Response headers (may contain Retry-After)

        Returns:
            Seconds to wait before retrying
        """
        self.consecutive_429s += 1
        self.last_429_time = time.monotonic()

        # Check for Retry-After header
        if headers:
            lower_headers = {k.lower(): v for k, v in headers.items()}
            retry_after = lower_headers.get("retry-after")
            if retry_after:
                try:
                    wait_time = float(retry_after)
                    self.current_backoff_seconds = wait_time
                    logger.info("rate_limited_with_retry_after", wait_seconds=wait_time)
                    return wait_time
                except ValueError:
                    pass

        # Exponential backoff: 1s, 2s, 4s, 8s, 16s, 32s, 60s max
        self.current_backoff_seconds = min(60.0, 2 ** (self.consecutive_429s - 1))
        logger.warning(
            "rate_limited_exponential_backoff",
            consecutive_429s=self.consecutive_429s,
            backoff_seconds=self.current_backoff_seconds,
        )
        return self.current_backoff_seconds

    def handle_success(self) -> None:
        """Handle a successful response - reset backoff state."""
        if self.consecutive_429s > 0:
            logger.debug("rate_limit_backoff_cleared", previous_429s=self.consecutive_429s)
        self.consecutive_429s = 0
        self.current_backoff_seconds = 1.0


class AdaptiveRateLimiter:
    """Manages rate limiting across multiple API endpoints."""

    def __init__(self) -> None:
        self._states: dict[str, RateLimitState] = {}
        self._lock = asyncio.Lock()

    def _get_state(self, api_name: str, default_rpm: int = 30) -> RateLimitState:
        """Get or create rate limit state for an API."""
        if api_name not in self._states:
            self._states[api_name] = RateLimitState(default_requests_per_minute=default_rpm)
        return self._states[api_name]

    async def wait_if_needed(self, api_name: str, default_rpm: int = 30) -> None:
        """Wait if rate limit requires it.

        Args:
            api_name: Identifier for the API (e.g., "ethresearch", "arxiv")
            default_rpm: Default requests per minute if no limit discovered
        """
        async with self._lock:
            state = self._get_state(api_name, default_rpm)
            should_wait, wait_time = state.should_wait()

        if should_wait and wait_time > 0:
            logger.info("rate_limit_waiting", api=api_name, seconds=round(wait_time, 1))
            await asyncio.sleep(wait_time)

    async def record_request(self, api_name: str) -> None:
        """Record that a request was made."""
        async with self._lock:
            state = self._get_state(api_name)
            state.record_request()

    async def update_from_response(
        self,
        api_name: str,
        status_code: int,
        headers: dict[str, str] | None = None,
    ) -> float:
        """Update rate limit state from response.

        Args:
            api_name: API identifier
            status_code: HTTP response status code
            headers: Response headers

        Returns:
            Wait time in seconds (0 if no wait needed, >0 if rate limited)
        """
        async with self._lock:
            state = self._get_state(api_name)

            if headers:
                state.update_from_headers(headers)

            if status_code == 429:
                return state.handle_429(headers)
            else:
                state.handle_success()
                return 0.0

    async def execute_with_retry(
        self,
        api_name: str,
        func: Any,
        *args: Any,
        max_retries: int = 5,
        default_rpm: int = 30,
        **kwargs: Any,
    ) -> Any:
        """Execute a function with automatic rate limit handling.

        Args:
            api_name: API identifier for rate limiting
            func: Async function to execute (should return (status_code, headers, result))
            max_retries: Maximum retry attempts on 429
            default_rpm: Default requests per minute

        Returns:
            Result from the function

        Raises:
            Exception: If max retries exceeded or non-429 error
        """
        for attempt in range(max_retries + 1):
            await self.wait_if_needed(api_name, default_rpm)
            await self.record_request(api_name)

            try:
                status_code, headers, result = await func(*args, **kwargs)

                wait_time = await self.update_from_response(api_name, status_code, headers)

                if status_code == 429:
                    if attempt < max_retries:
                        logger.warning(
                            "rate_limit_retry",
                            api=api_name,
                            attempt=attempt + 1,
                            max_retries=max_retries,
                            wait_seconds=wait_time,
                        )
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        raise RuntimeError(f"Rate limit exceeded after {max_retries} retries")

                return result

            except Exception as e:
                if "429" in str(e) or "rate limit" in str(e).lower():
                    wait_time = await self.update_from_response(api_name, 429, None)
                    if attempt < max_retries:
                        await asyncio.sleep(wait_time)
                        continue
                raise

        raise RuntimeError(f"Rate limit exceeded after {max_retries} retries")


# Global rate limiter instance for shared use
_global_limiter: AdaptiveRateLimiter | None = None


def get_rate_limiter() -> AdaptiveRateLimiter:
    """Get the global rate limiter instance."""
    global _global_limiter
    if _global_limiter is None:
        _global_limiter = AdaptiveRateLimiter()
    return _global_limiter
