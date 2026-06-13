#!/bin/bash
# init.sh — 统一启动与验证入口
set -e

INSTALL_CMD="pip install -e '.[dev]'"
VERIFY_CMD="pytest tests/ -v --tb=short"
START_CMD="python -m uvicorn src.api:app --reload --port 8080"

echo "=== WYF Agent — 初始化 ==="
echo "工作目录: $(pwd)"

echo ""
echo "--- 安装依赖 ---"
eval "$INSTALL_CMD"

echo ""
echo "--- 运行验证 ---"
eval "$VERIFY_CMD" || echo "⚠️ 验证失败，请先修复基础问题"

echo ""
echo "--- 启动命令 ---"
echo "运行以下命令启动服务："
echo "  $START_CMD"

if [ "${RUN_START_COMMAND:-0}" = "1" ]; then
    echo ""
    echo "--- 启动服务 ---"
    eval "$START_CMD"
fi
