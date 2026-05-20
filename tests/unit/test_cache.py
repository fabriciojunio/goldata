"""Testes do sistema de cache."""

import time
import pytest
from goldata.cache import CacheManager


@pytest.fixture
def cache():
    return CacheManager()


def test_set_and_get(cache):
    cache.set("key1", "value1")
    assert cache.get("key1") == "value1"


def test_get_nonexistent_returns_none(cache):
    assert cache.get("nonexistent") is None


def test_delete_existing_key(cache):
    cache.set("key", "val")
    result = cache.delete("key")
    assert result is True
    assert cache.get("key") is None


def test_delete_nonexistent_returns_false(cache):
    assert cache.delete("nonexistent") is False


def test_clear_returns_count(cache):
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)
    count = cache.clear()
    assert count == 3
    assert cache.size() == 0


def test_ttl_expiration(cache):
    cache.set("temp", "value", ttl=1)
    assert cache.get("temp") == "value"
    time.sleep(1.1)
    assert cache.get("temp") is None


def test_no_ttl_does_not_expire(cache):
    cache.set("permanent", "value", ttl=0)
    time.sleep(0.1)
    assert cache.get("permanent") == "value"


def test_overwrite_key(cache):
    cache.set("key", "old")
    cache.set("key", "new")
    assert cache.get("key") == "new"


def test_cache_size(cache):
    assert cache.size() == 0
    cache.set("a", 1)
    cache.set("b", 2)
    assert cache.size() == 2


def test_cache_keys(cache):
    cache.set("x", 1)
    cache.set("y", 2)
    keys = cache.keys()
    assert "x" in keys
    assert "y" in keys


def test_cache_stores_different_types(cache):
    cache.set("int", 42)
    cache.set("list", [1, 2, 3])
    cache.set("dict", {"a": 1})
    assert cache.get("int") == 42
    assert cache.get("list") == [1, 2, 3]
    assert cache.get("dict") == {"a": 1}


def test_clear_empty_cache(cache):
    count = cache.clear()
    assert count == 0


def test_expired_key_not_counted_in_size(cache):
    cache.set("fast", "val", ttl=1)
    assert cache.size() == 1
    time.sleep(1.1)
    cache.get("fast")  # trigger expiration check
    assert cache.size() == 0
