"""Unit tests for the viewer's byte-capped LRU row cache."""

import numpy as np

from microVis.main_window import _RowCache


def test_setitem_get_and_update():
    cache = _RowCache(max_bytes=1 << 30)
    img = np.zeros((4, 4, 2), dtype=np.uint16)
    cache[7] = (img, {"mask": np.ones((4, 4), dtype=bool)})
    assert 7 in cache
    payload = cache.get(7)
    assert payload[0] is img
    assert cache.get(99) is None


def test_lru_eviction_respects_byte_cap():
    # Each payload is an 8-byte image + a 4-byte mask → 12 bytes per row.
    def _img(seed):
        return np.full((2, 2, 2), seed, dtype=np.uint8)  # 8 bytes + mask 4 bytes

    cache = _RowCache(max_bytes=8 * 3 + 4 * 3)  # room for exactly 3 rows
    for row in range(5):
        cache[row] = (_img(row), {"m": np.ones((2, 2), dtype=bool)})

    # Oldest rows evicted first; the newest survive.
    assert 0 not in cache and 1 not in cache
    assert 2 in cache and 3 in cache and 4 in cache


def test_get_touches_lru_order():
    cache = _RowCache(max_bytes=8 * 2)  # room for two 8-byte payloads
    cache[0] = (np.full((2, 2, 2), 1, dtype=np.uint8), {})
    cache[1] = (np.full((2, 2, 2), 2, dtype=np.uint8), {})
    # Touch row 0 so row 1 becomes the LRU entry.
    cache.get(0)
    cache[2] = (np.full((2, 2, 2), 3, dtype=np.uint8), {})
    assert 0 in cache          # touched → survived
    assert 1 not in cache      # least recently used → evicted


def test_clear_and_bool():
    cache = _RowCache(max_bytes=1 << 30)
    assert not cache
    cache[1] = (np.zeros((2, 2, 2), dtype=np.uint8), {})
    assert bool(cache)
    cache.clear()
    assert not cache
    assert cache.get(1) is None


def test_oversized_entry_is_kept():
    """A single payload larger than the cap must still be cached (the
    cache never drops the entry it just stored)."""
    cache = _RowCache(max_bytes=8)
    big = np.zeros((100, 100), dtype=np.uint8)
    cache[0] = (big, {})
    assert cache.get(0)[0] is big
