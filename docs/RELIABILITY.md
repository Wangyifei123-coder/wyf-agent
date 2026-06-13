# RELIABILITY.md — 运行可靠性

## 标准启动路径

```bash
pip install -e ".[dev]"
python -m uvicorn src.api:app --reload --port 8080
```

## 标准验证路径

```bash
pytest tests/ -v
ruff check src/
mypy src/
```

## 健康检查

```bash
curl http://localhost:8080/health
```

## 运行信号

| 信号 | 正常值 | 告警阈值 |
|------|--------|---------|
| API 响应时间 | < 3s | > 10s |
| LLM 调用成功率 | > 95% | < 90% |
| 工具调用成功率 | > 98% | < 95% |
| 内存使用 | < 512MB | > 1GB |
| Token 用量/会话 | < 10k | > 50k |

## 重启要求

- 修改 Gateway 配置后需重启
- 修改 Safety 规则后需重启
- 添加新工具后需重启
- 修改 Prompt 后不需要重启（运行时加载）

## Benchmark

```bash
# 待实现
# pytest tests/benchmark/ -v --benchmark
```
