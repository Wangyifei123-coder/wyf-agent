#!/bin/bash
# cleanup-scanner.sh — 检测过期工件和代码卫生问题
set -e

echo "=== WYF Agent — Cleanup Scanner ==="
ISSUES=0

echo ""
echo "--- 检查 .env 是否被提交 ---"
if git ls-files | grep -q "\.env$"; then
    echo "❌ .env 文件被提交到 git！"
    ISSUES=$((ISSUES + 1))
else
    echo "✅ .env 未被提交"
fi

echo ""
echo "--- 检查 __pycache__ 是否被提交 ---"
if git ls-files | grep -q "__pycache__"; then
    echo "❌ __pycache__ 被提交到 git！"
    ISSUES=$((ISSUES + 1))
else
    echo "✅ __pycache__ 未被提交"
fi

echo ""
echo "--- 检查 TODO/FIXME/HACK ---"
grep -rn "TODO\|FIXME\|HACK" src/ 2>/dev/null | head -20 || echo "✅ 无 TODO/FIXME/HACK"

echo ""
echo "--- 检查未使用的导入 ---"
ruff check src/ --select F401 2>&1 || true

echo ""
echo "--- 检查 feature_list.json 状态 ---"
if [ -f feature_list.json ]; then
    PASSING=$(grep -c '"passing"' feature_list.json || true)
    TOTAL=$(grep -c '"id"' feature_list.json || true)
    echo "功能状态: $PASSING/$TOTAL passing"
fi

echo ""
if [ $ISSUES -eq 0 ]; then
    echo "✅ Cleanup scan 通过，无问题发现"
else
    echo "⚠️ 发现 $ISSUES 个问题，请修复"
fi
