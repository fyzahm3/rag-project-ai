"""API-key auth and a per-process sliding-window rate limiter for the HTTP layer."""
from __future__ import annotations

import time
from collections import defaultdict, deque
from threading import Lock

from fastapi import Header, HTTPException, Request


async def require_api_key(request: Request, x_api_key: str | None = Header(default=None)) -> None:
    """No-op when API_KEY is unset (local/dev default); enforced once an operator sets it."""
    settings = request.app.state.ctx.settings
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Missing or invalid API key")


class RateLimiter:
    """In-memory sliding-window limiter keyed by client address.

    Per-process only (does not coordinate across workers/replicas) — enough to blunt
    accidental hammering and single-client abuse, not a substitute for an edge limiter
    in a multi-instance deployment.
    """

    def __init__(self, max_requests: int, window_seconds: float = 60.0) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def check(self, key: str) -> None:
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] > self.window_seconds:
                hits.popleft()
            if len(hits) >= self.max_requests:
                raise HTTPException(status_code=429, detail="Rate limit exceeded; slow down")
            hits.append(now)


def client_key(request: Request) -> str:
    if request.client:
        return request.client.host
    return "unknown"
