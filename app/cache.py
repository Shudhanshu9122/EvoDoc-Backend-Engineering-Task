"""
EvoDoc — Clinical Drug Safety Engine
Cache Layer

SHA-256 keyed, TTL-based in-memory cache with lazy eviction,
hit/miss stats, and a background sweep thread.
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from app.models import CacheStats

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_TTL_SECONDS: int = 3600          # 1 hour
SWEEP_INTERVAL_SECONDS: int = 300        # eviction sweep every 5 minutes


# ---------------------------------------------------------------------------
# Internal entry type
# ---------------------------------------------------------------------------


@dataclass
class _CacheEntry:
    value: Any
    created_at: float = field(default_factory=time.monotonic)
    ttl: int = DEFAULT_TTL_SECONDS

    @property
    def is_expired(self) -> bool:
        return (time.monotonic() - self.created_at) > self.ttl

    @property
    def age_seconds(self) -> float:
        return time.monotonic() - self.created_at


# ---------------------------------------------------------------------------
# Cache implementation
# ---------------------------------------------------------------------------


class DrugSafetyCache:
    """
    Thread-safe, TTL-aware in-memory cache for drug evaluation results.

    Cache key is derived from the sorted list of drugs and current
    medications — providing patient-agnostic interaction reuse.
    """

    def __init__(self, ttl: int = DEFAULT_TTL_SECONDS) -> None:
        self._ttl = ttl
        self._store: dict[str, _CacheEntry] = {}
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

        # Start background eviction sweep
        self._sweep_thread = threading.Thread(
            target=self._background_sweep,
            daemon=True,
            name="cache-sweep",
        )
        self._sweep_thread.start()

    # ------------------------------------------------------------------
    # Key generation
    # ------------------------------------------------------------------

    @staticmethod
    def make_key(drugs: list[str], current_medications: list[str]) -> str:
        """
        Produce a deterministic SHA-256 cache key from drug lists.

        Sorted to ensure order-independence.
        Example:
            drugs = ["Warfarin", "Aspirin"]
            current_medications = ["Metformin"]
            → SHA-256("aspirin|warfarin::metformin")
        """
        drugs_part = "|".join(sorted(d.lower() for d in drugs))
        meds_part = "|".join(sorted(m.lower() for m in current_medications))
        raw = f"{drugs_part}::{meds_part}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def get(self, key: str) -> Any | None:
        """Return cached value or None if miss/expired."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            if entry.is_expired:
                del self._store[key]
                self._misses += 1
                return None
            self._hits += 1
            return entry.value

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Store a value with the given (or default) TTL."""
        with self._lock:
            self._store[key] = _CacheEntry(
                value=value,
                ttl=ttl if ttl is not None else self._ttl,
            )

    def delete(self, key: str) -> bool:
        """Remove a specific key. Returns True if it existed."""
        with self._lock:
            return self._store.pop(key, None) is not None

    def flush(self) -> int:
        """Clear all cache entries. Returns the number of entries removed."""
        with self._lock:
            count = len(self._store)
            self._store.clear()
            return count

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> CacheStats:
        """Return current cache performance metrics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100.0) if total_requests > 0 else 0.0

            oldest_age: float | None = None
            if self._store:
                oldest_age = max(e.age_seconds for e in self._store.values())

            return CacheStats(
                total_keys=len(self._store),
                hits=self._hits,
                misses=self._misses,
                hit_rate_percent=round(hit_rate, 2),
                oldest_entry_age_seconds=round(oldest_age, 2) if oldest_age else None,
            )

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._store)

    # ------------------------------------------------------------------
    # Background sweep
    # ------------------------------------------------------------------

    def _background_sweep(self) -> None:
        """Periodically remove all expired entries."""
        while True:
            time.sleep(SWEEP_INTERVAL_SECONDS)
            with self._lock:
                expired_keys = [k for k, v in self._store.items() if v.is_expired]
                for key in expired_keys:
                    del self._store[key]


# ---------------------------------------------------------------------------
# Module-level singleton and top-level functions
# ---------------------------------------------------------------------------

_cache_instance: DrugSafetyCache | None = None


def get_cache_client() -> DrugSafetyCache:
    """Return the module-level cache singleton (lazy initialization). Used for FastAPI Depends."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = DrugSafetyCache()
    return _cache_instance


def generate_cache_key(medicines: list[str], current_medications: list[str]) -> str:
    """Helper to generate the SHA-256 key from sorted drug lists."""
    return DrugSafetyCache.make_key(medicines, current_medications)


def get_cache(key: str) -> Any | None:
    """Retrieve a value from the cache by key."""
    return get_cache_client().get(key)


def set_cache(key: str, value: Any) -> None:
    """Store a value in the cache with the default 1-hour TTL."""
    get_cache_client().set(key, value)
