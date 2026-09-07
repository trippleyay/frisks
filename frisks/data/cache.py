"""
Minimal in-process TTL cache.

Decision (open item — caching TTL/strategy not specified): a simple
dict-based TTL cache keyed by (endpoint, params) is sufficient for an MVP
serving synchronous strategy-search requests. It avoids re-fetching
exchangeInfo/ticker/mark/openInterest multiple times within the same
request (the generator and scorer both need market data) and across
back-to-back requests for the same underlying/expiry, without the
operational overhead of an external cache (Redis, etc.) that a single-
process MVP doesn't need yet. If Frisks is horizontally scaled, swap this
for a shared cache — the interface (get/set with TTL) is intentionally
small so that's a drop-in change.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable, TypeVar

T = TypeVar("T")


class TTLCache:
    def __init__(self) -> None:
        self._store: dict[str, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if time.monotonic() >= expires_at:
                del self._store[key]
                return None
            return value

    def set(self, key: str, value: Any, ttl_s: float) -> None:
        with self._lock:
            self._store[key] = (time.monotonic() + ttl_s, value)

    def get_or_set(self, key: str, ttl_s: float, factory: Callable[[], T]) -> T:
        cached = self.get(key)
        if cached is not None:
            return cached
        value = factory()
        self.set(key, value, ttl_s)
        return value

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
