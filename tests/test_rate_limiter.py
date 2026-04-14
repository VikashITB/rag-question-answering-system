"""
tests/test_rate_limiter.py — Unit tests for the sliding window rate limiter.
"""

import time
import pytest

from app.utils.rate_limiter import SlidingWindowRateLimiter


def test_allows_requests_within_limit():
    limiter = SlidingWindowRateLimiter(max_requests=5, window_seconds=60)
    for _ in range(5):
        assert limiter.is_allowed("client_a") is True


def test_blocks_after_limit_exceeded():
    limiter = SlidingWindowRateLimiter(max_requests=3, window_seconds=60)
    for _ in range(3):
        limiter.is_allowed("client_b")
    assert limiter.is_allowed("client_b") is False


def test_different_clients_are_independent():
    limiter = SlidingWindowRateLimiter(max_requests=2, window_seconds=60)
    limiter.is_allowed("client_x")
    limiter.is_allowed("client_x")
    # client_x is exhausted, but client_y should still be allowed
    assert limiter.is_allowed("client_y") is True


def test_window_expires_and_allows_again():
    """After the window expires, the counter should reset."""
    limiter = SlidingWindowRateLimiter(max_requests=2, window_seconds=1)
    limiter.is_allowed("client_c")
    limiter.is_allowed("client_c")
    assert limiter.is_allowed("client_c") is False

    time.sleep(1.1)  # Wait for window to expire

    assert limiter.is_allowed("client_c") is True
