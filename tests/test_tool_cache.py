"""测试工具结果缓存机制"""

import asyncio
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tools.registry import ToolCache, ToolRegistry, Tool, ToolSchema, ToolParameter


class MockTool(Tool):
    def __init__(self, name: str, cacheable: bool = False):
        self._name = name
        self._cacheable = cacheable
        self._call_count = 0

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self._name,
            description="Mock tool",
            parameters=[ToolParameter(name="query", type="string", description="Query")],
            cacheable=self._cacheable,
        )

    async def execute(self, **kwargs) -> str:
        self._call_count += 1
        return f"result_{kwargs.get('query', '')}_{self._call_count}"


async def test_cache_basic():
    cache = ToolCache(max_size=10, ttl=300)

    result = await cache.get("tool1", {"query": "test"})
    assert result is None, "Expected cache miss"

    await cache.set("tool1", {"query": "test"}, "result1")
    result = await cache.get("tool1", {"query": "test"})
    assert result == "result1", f"Expected 'result1', got '{result}'"

    result = await cache.get("tool1", {"query": "different"})
    assert result is None, "Expected cache miss for different args"

    result = await cache.get("tool2", {"query": "test"})
    assert result is None, "Expected cache miss for different tool"

    print("[PASS] test_cache_basic")


async def test_cache_expiration():
    cache = ToolCache(max_size=10, ttl=1)

    await cache.set("tool1", {"query": "test"}, "result1")
    result = await cache.get("tool1", {"query": "test"})
    assert result == "result1", "Expected hit before expiration"

    await asyncio.sleep(1.1)
    result = await cache.get("tool1", {"query": "test"})
    assert result is None, "Expected miss after expiration"

    print("[PASS] test_cache_expiration")


async def test_cache_size_limit():
    cache = ToolCache(max_size=3, ttl=300)

    await cache.set("tool1", {"query": "1"}, "r1")
    await cache.set("tool2", {"query": "2"}, "r2")
    await cache.set("tool3", {"query": "3"}, "r3")
    assert cache.get_stats()["size"] == 3

    await cache.set("tool4", {"query": "4"}, "r4")
    stats = cache.get_stats()
    assert stats["size"] == 3, f"Expected 3 after eviction, got {stats['size']}"

    assert await cache.get("tool1", {"query": "1"}) is None, "Oldest evicted"
    assert await cache.get("tool4", {"query": "4"}) == "r4", "New entry present"

    print("[PASS] test_cache_size_limit")


async def test_cache_stats():
    cache = ToolCache(max_size=10, ttl=300)

    stats = cache.get_stats()
    assert stats["hits"] == 0 and stats["misses"] == 0 and stats["hit_rate"] == 0.0

    await cache.set("tool1", {"query": "test"}, "result1")
    await cache.get("tool1", {"query": "test"})
    await cache.get("tool1", {"query": "test"})
    await cache.get("tool2", {"query": "test"})

    stats = cache.get_stats()
    assert stats["hits"] == 2
    assert stats["misses"] == 1
    assert abs(stats["hit_rate"] - 2 / 3) < 0.01

    print("[PASS] test_cache_stats")


async def test_registry_cache_integration():
    cached_tool = MockTool("cached_tool", cacheable=True)
    uncached_tool = MockTool("uncached_tool", cacheable=False)

    registry = ToolRegistry()
    registry.register(cached_tool)
    registry.register(uncached_tool)

    result1 = await registry.call("cached_tool", {"query": "test"})
    assert cached_tool._call_count == 1, "First call executes"

    result2 = await registry.call("cached_tool", {"query": "test"})
    assert cached_tool._call_count == 1, "Second call from cache"
    assert result1 == result2, "Same cached result"

    await registry.call("cached_tool", {"query": "different"})
    assert cached_tool._call_count == 2, "Different args triggers execution"

    await registry.call("uncached_tool", {"query": "test"})
    await registry.call("uncached_tool", {"query": "test"})
    assert uncached_tool._call_count == 2, "Uncached tool always executes"

    stats = registry.get_cache_stats()
    assert stats["hits"] == 1, f"Expected 1 hit, got {stats['hits']}"
    assert stats["misses"] == 2, f"Expected 2 misses, got {stats['misses']}"

    registry.clear_cache()
    await registry.call("cached_tool", {"query": "test"})
    assert cached_tool._call_count == 3, "After clear, re-executes"

    print("[PASS] test_registry_cache_integration")


async def test_cache_thread_safety():
    cache = ToolCache(max_size=100, ttl=300)

    async def writer(start: int):
        for i in range(10):
            await cache.set(f"tool_{start}", {"i": i}, f"result_{start}_{i}")

    async def reader(start: int) -> int:
        hits = 0
        for i in range(10):
            result = await cache.get(f"tool_{start}", {"i": i})
            if result:
                hits += 1
        return hits

    await asyncio.gather(*[writer(i) for i in range(10)])
    assert cache.get_stats()["size"] == 100

    results = await asyncio.gather(*[reader(i) for i in range(10)])
    assert sum(results) == 100

    print("[PASS] test_cache_thread_safety")


if __name__ == "__main__":
    asyncio.run(test_cache_basic())
    asyncio.run(test_cache_expiration())
    asyncio.run(test_cache_size_limit())
    asyncio.run(test_cache_stats())
    asyncio.run(test_registry_cache_integration())
    asyncio.run(test_cache_thread_safety())
    print("\nAll cache tests passed!")
