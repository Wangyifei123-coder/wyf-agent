"""测试工具调用优化逻辑"""

import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.graph import MATH_PATTERN, WEATHER_KEYWORDS, RAGGraph


def test_math_pattern():
    """测试数学表达式匹配"""
    test_cases = [
        ("计算 123 * 456", True),
        ("123 * 456 等于多少", True),
        ("100 + 200", True),
        ("10 加 20", True),
        ("你好", False),
        ("什么是Python", False),
        ("北京天气", False),
    ]

    passed = 0
    for query, expected in test_cases:
        result = bool(MATH_PATTERN.search(query))
        if result == expected:
            passed += 1
            print(f"[PASS] '{query}' -> match: {result}")
        else:
            print(f"[FAIL] '{query}' -> match: {result} (expected: {expected})")
    print(f"Results: {passed}/{len(test_cases)} passed")


def test_weather_keywords():
    """测试天气关键词匹配"""
    test_cases = [
        ("北京天气", True),
        ("上海今天天气怎么样", True),
        ("明天会下雨吗", True),
        ("计算 1+1", False),
        ("你好", False),
    ]

    passed = 0
    for query, expected in test_cases:
        normalized = query.strip().lower()
        result = any(kw in normalized for kw in WEATHER_KEYWORDS)
        if result == expected:
            passed += 1
            print(f"[PASS] '{query}' -> match: {result}")
        else:
            print(f"[FAIL] '{query}' -> match: {result} (expected: {expected})")
    print(f"Results: {passed}/{len(test_cases)} passed")


def test_extract_math_expression():
    """测试数学表达式提取"""
    graph = RAGGraph.__new__(RAGGraph)

    test_cases = [
        ("计算 123 * 456", "123 * 456"),
        ("100 + 200 等于多少", "100 + 200"),
        ("10 加 20", "10+20"),
        ("你好", None),
    ]

    passed = 0
    for query, expected in test_cases:
        result = graph._extract_math_expression(query)
        if result == expected:
            passed += 1
            print(f"[PASS] '{query}' -> '{result}'")
        else:
            print(f"[FAIL] '{query}' -> '{result}' (expected: '{expected}')")
    print(f"Results: {passed}/{len(test_cases)} passed")


def test_extract_city():
    """测试城市提取"""
    graph = RAGGraph.__new__(RAGGraph)

    test_cases = [
        ("北京天气", "北京"),
        ("上海今天天气怎么样", "上海"),
        ("查询深圳天气", "深圳"),
        ("明天北京会下雨吗", "北京"),
        ("你好", None),
    ]

    passed = 0
    for query, expected in test_cases:
        result = graph._extract_city(query)
        if result == expected:
            passed += 1
            print(f"[PASS] '{query}' -> '{result}'")
        else:
            print(f"[FAIL] '{query}' -> '{result}' (expected: '{expected}')")
    print(f"Results: {passed}/{len(test_cases)} passed")


if __name__ == "__main__":
    print("=== Math Expression Match Test ===")
    test_math_pattern()

    print("\n=== Weather Keywords Match Test ===")
    test_weather_keywords()

    print("\n=== Math Expression Extract Test ===")
    test_extract_math_expression()

    print("\n=== City Extract Test ===")
    test_extract_city()

    print("\nAll tests completed!")
