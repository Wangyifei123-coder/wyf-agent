# Session Handoff

## 当前已验证

- 项目骨架已创建，7 层架构完整
- 所有 Harness 文件就位（AGENTS.md、ARCHITECTURE.md、feature_list.json、claude-progress.md）
- pyproject.toml 配置完成，依赖声明完整

## 本轮改动

- 从零初始化整个项目
- 实现所有 7 层的骨架代码
- 创建完整的文档体系

## 仍损坏或未验证

- LLM 调用未验证（需要 API Key）
- 向量存储未集成
- 端到端对话流程未测试
- 评估数据集为空

## 下一步最佳动作

1. 配置 `.env` 文件（填入 API Key）
2. 运行 `pip install -e ".[dev]"` 安装依赖
3. 运行 `pytest tests/ -v` 验证基础测试
4. 运行 `python -m uvicorn src.api:app --reload` 启动服务
5. 用 curl 测试 `/chat` 端点

## 命令

```bash
# 安装
pip install -e ".[dev]"

# 验证
pytest tests/ -v
ruff check src/
mypy src/

# 启动
python -m uvicorn src.api:app --reload --port 8080

# 测试
curl -X POST http://localhost:8080/chat -H "Content-Type: application/json" -d '{"message": "hello"}'
```
