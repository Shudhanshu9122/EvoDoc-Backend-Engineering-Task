"""
Tests for cache.py — TTL expiry, hit/miss, stats, flush.
"""

import time

import pytest

# Adjust path for test imports
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.cache import DrugSafetyCache


# ---------------------------------------------------------------------------
# Key generation
# ---------------------------------------------------------------------------


def test_make_key_is_deterministic():
    """Same inputs always produce the same key."""
    k1 = DrugSafetyCache.make_key(["Warfarin", "Aspirin"], ["Metformin"])
    k2 = DrugSafetyCache.make_key(["Warfarin", "Aspirin"], ["Metformin"])
    assert k1 == k2


def test_make_key_is_order_independent():
    """Drug list order should not affect the key."""
    k1 = DrugSafetyCache.make_key(["Warfarin", "Aspirin"], [])
    k2 = DrugSafetyCache.make_key(["Aspirin", "Warfarin"], [])
    assert k1 == k2


def test_make_key_is_case_insensitive():
    k1 = DrugSafetyCache.make_key(["warfarin"], [])
    k2 = DrugSafetyCache.make_key(["WARFARIN"], [])
    assert k1 == k2


def test_make_key_different_drugs_produce_different_keys():
    k1 = DrugSafetyCache.make_key(["Warfarin"], [])
    k2 = DrugSafetyCache.make_key(["Aspirin"], [])
    assert k1 != k2


# ---------------------------------------------------------------------------
# Get / Set
# ---------------------------------------------------------------------------


def test_cache_set_and_get():
    cache = DrugSafetyCache()
    cache.set("test-key", {"value": 42})
    result = cache.get("test-key")
    assert result == {"value": 42}


def test_cache_miss_returns_none():
    cache = DrugSafetyCache()
    assert cache.get("nonexistent-key") is None


def test_cache_expired_entry_returns_none():
    cache = DrugSafetyCache(ttl=1)  # 1 second TTL
    cache.set("expiring-key", "data")
    time.sleep(1.5)  # Wait for expiry
    assert cache.get("expiring-key") is None


def test_cache_valid_entry_not_expired():
    cache = DrugSafetyCache(ttl=60)
    cache.set("valid-key", "data")
    assert cache.get("valid-key") == "data"


# ---------------------------------------------------------------------------
# Hits and misses
# ---------------------------------------------------------------------------


def test_hit_increments_hit_count():
    cache = DrugSafetyCache()
    cache.set("k", "v")
    cache.get("k")
    stats = cache.stats()
    assert stats.hits == 1
    assert stats.misses == 0


def test_miss_increments_miss_count():
    cache = DrugSafetyCache()
    cache.get("does-not-exist")
    stats = cache.stats()
    assert stats.misses == 1
    assert stats.hits == 0


def test_hit_rate_calculated_correctly():
    cache = DrugSafetyCache()
    cache.set("k1", "v1")
    cache.get("k1")  # hit
    cache.get("k1")  # hit
    cache.get("no")  # miss
    stats = cache.stats()
    assert stats.hit_rate_percent == pytest.approx(66.67, abs=0.1)


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


def test_delete_existing_key():
    cache = DrugSafetyCache()
    cache.set("k", "v")
    result = cache.delete("k")
    assert result is True
    assert cache.get("k") is None


def test_delete_nonexistent_key_returns_false():
    cache = DrugSafetyCache()
    assert cache.delete("ghost-key") is False


# ---------------------------------------------------------------------------
# Flush
# ---------------------------------------------------------------------------


def test_flush_clears_all_entries():
    cache = DrugSafetyCache()
    cache.set("k1", "v1")
    cache.set("k2", "v2")
    removed = cache.flush()
    assert removed == 2
    assert cache.size == 0


def test_flush_empty_cache_returns_zero():
    cache = DrugSafetyCache()
    assert cache.flush() == 0


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def test_stats_returns_correct_size():
    cache = DrugSafetyCache()
    cache.set("a", 1)
    cache.set("b", 2)
    stats = cache.stats()
    assert stats.total_keys == 2


def test_stats_oldest_entry_age_none_when_empty():
    cache = DrugSafetyCache()
    stats = cache.stats()
    assert stats.oldest_entry_age_seconds is None
