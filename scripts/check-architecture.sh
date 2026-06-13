#!/bin/bash
# check-architecture.sh — 检查架构边界违规
set -e

echo "=== WYF Agent — Architecture Check ==="
VIOLATIONS=0

echo ""
echo "--- 检查跨层依赖 ---"

# Gateway 不应依赖 Tools/Memory/Reasoning/Orchestration
if grep -rn "from.*tools\|from.*memory\|from.*reasoning\|from.*orchestration" src/gateway/ 2>/dev/null; then
    echo "❌ Gateway 层不应依赖 Tools/Memory/Reasoning/Orchestration"
    VIOLATIONS=$((VIOLATIONS + 1))
else
    echo "✅ Gateway 层依赖正确"
fi

# Tools 不应依赖 Reasoning/Orchestration
if grep -rn "from.*reasoning\|from.*orchestration" src/tools/ 2>/dev/null; then
    echo "❌ Tools 层不应依赖 Reasoning/Orchestration"
    VIOLATIONS=$((VIOLATIONS + 1))
else
    echo "✅ Tools 层依赖正确"
fi

# Memory 不应依赖 Reasoning/Orchestration
if grep -rn "from.*reasoning\|from.*orchestration" src/memory/ 2>/dev/null; then
    echo "❌ Memory 层不应依赖 Reasoning/Orchestration"
    VIOLATIONS=$((VIOLATIONS + 1))
else
    echo "✅ Memory 层依赖正确"
fi

# Reasoning 不应依赖 Orchestration
if grep -rn "from.*orchestration" src/reasoning/ 2>/dev/null; then
    echo "❌ Reasoning 层不应依赖 Orchestration"
    VIOLATIONS=$((VIOLATIONS + 1))
else
    echo "✅ Reasoning 层依赖正确"
fi

# UI/API 层不应有 Node.js 或硬编码路径
if grep -rn "localhost\|127\.0\.0\.1" src/ --include="*.py" 2>/dev/null | grep -v "test\|config\|api.py"; then
    echo "⚠️ 发现硬编码地址（应使用配置）"
fi

echo ""
if [ $VIOLATIONS -eq 0 ]; then
    echo "✅ 架构检查通过"
else
    echo "❌ 发现 $VIOLATIONS 个架构违规"
fi
