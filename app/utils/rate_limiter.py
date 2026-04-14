"""
app/utils/rate_limiter.py — Simple in-memory rate limiter using a sliding window.

Design choice: Token-bucket / fixed-window counters stored in a dict.
- No Redis dependency — acceptable for single-instance deployments.
- For multi-instance production, replace with Redis + sliding window.

Each client is identified by IP address (extracted from the request).
"""

import logging
import time
import threading
from collections import defaultdict, deque
from typing import Dict, Deque

from fastapi import Request, HTTPException, status

logger = logging.getLogger(__name__)


class SlidingWindowRateLimiter:
    """
    Sliding window rate limiter.

    Tracks request timestamps per client in a deque.
    A request is allowed if the number of requests in the past `window_seconds`
    is below `max_requests`.

    Thread-safe via a per-client lock.
    """

    def __init__(self, max_requests: int = 30, window_seconds: int = 60):
        """
        Args:
            max_requests:   Max allowed requests per window per client.
            window_seconds: Size of the sliding window in seconds.
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        # client_ip → deque of timestamps
        self._windows: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def is_allowed(self, client_ip: str) -> bool:
        """
        Returns True if the request should be allowed, False if rate-limited.
        Automatically evicts timestamps outside the current window.
        """
        now = time.monotonic()
        window_start = now - self.window_seconds

        with self._lock:
            dq = self._windows[client_ip]

            # Remove timestamps older than the window
            while dq and dq[0] < window_start:
                dq.popleft()

            if len(dq) >= self.max_requests:
                logger.warning(
                    "Rate limit exceeded for client: %s (%d requests in %ds window)",
                    client_ip, len(dq), self.window_seconds,
                )
                return False

            dq.append(now)
            return True

    def check(self, request: Request) -> None:
        """
        Convenience method: raises HTTP 429 if the client is rate-limited.
        Use as a FastAPI dependency.
        """
        client_ip = self._get_client_ip(request)
        if not self.is_allowed(client_ip):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=(
                    f"Rate limit exceeded: max {self.max_requests} requests "
                    f"per {self.window_seconds} seconds."
                ),
                headers={"Retry-After": str(self.window_seconds)},
            )

    @staticmethod
    def _get_client_ip(request: Request) -> str:
        """
        Extract client IP, respecting X-Forwarded-For for reverse proxy setups.
        Falls back to the direct connection IP.
        """
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"


# ── Singleton instances for different rate limits ─────────────────────────────

# Document upload: stricter — 10 uploads per minute
upload_limiter = SlidingWindowRateLimiter(max_requests=10, window_seconds=60)

# Question answering: moderate — 30 questions per minute
question_limiter = SlidingWindowRateLimiter(max_requests=30, window_seconds=60)


# ── FastAPI dependency functions ──────────────────────────────────────────────

def check_upload_limit(request: Request):
    upload_limiter.check(request)


def check_question_limit(request: Request):
    question_limiter.check(request)
