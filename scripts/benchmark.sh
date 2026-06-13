#!/bin/bash
# benchmark.sh — 性能基准测试
set -e

echo "=== WYF Agent — Benchmark ==="
echo "时间: $(date)"

echo ""
echo "--- 单元测试 ---"
pytest tests/ -v --tb=short --benchmark-disable 2>&1 || true

echo ""
echo "--- Lint 检查 ---"
ruff check src/ 2>&1 || true

echo ""
echo "--- 类型检查 ---"
mypy src/ 2>&1 || true

echo ""
echo "--- 测试覆盖率 ---"
pytest tests/ --cov=src --cov-report=term-missing 2>&1 || true

echo ""
echo "=== Benchmark 完成 ==="
