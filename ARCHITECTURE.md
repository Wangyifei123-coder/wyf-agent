# ARCHITECTURE.md — WYF Agent

这份文件是系统的顶层地图。保持简短，只提供最关键的结构信息。

## 系统形态

- 产品：WYF Agent — 企业级 AI 智能助手
- 主用户流程：用户提问 → 意图理解 → 工具调用 → 推理生成 → 带引用的回答
- 运行面：Web API (FastAPI) + 可选前端
- 产品行为真相来源：`docs/product-specs/`

## 七层架构

```
┌─────────────────────────────────────┐
│  Layer 7: 可观测性 & 治理            │  src/observability/
├─────────────────────────────────────┤
│  Layer 6: 安全 & 合规               │  src/safety/
├─────────────────────────────────────┤
│  Layer 5: 编排层 (Orchestration)     │  src/orchestration/
├─────────────────────────────────────┤
│  Layer 4: 记忆层 (Memory)            │  src/memory/
├─────────────────────────────────────┤
│  Layer 3: 推理层 (Reasoning)         │  src/reasoning/
├─────────────────────────────────────┤
│  Layer 2: 工具层 (Tools)             │  src/tools/
├─────────────────────────────────────┤
│  Layer 1: 模型层 (LLM Gateway)      │  src/gateway/
└─────────────────────────────────────┘
```

## 领域地图

| 领域 | 负责什么 | 主要入口 | 对应规格 |
|------|---------|---------|---------|
| Gateway | LLM 统一接口、多模型路由、token 管理 | `src/gateway/client.py` | `docs/product-specs/gateway.md` |
| Tools | 工具注册、调用、校验、审计 | `src/tools/registry.py` | `docs/product-specs/tools.md` |
| Memory | 短期/长期/工作记忆管理 | `src/memory/manager.py` | `docs/product-specs/memory.md` |
| Reasoning | ReAct / Plan-and-Execute 推理循环 | `src/reasoning/react.py` | `docs/product-specs/reasoning.md` |
| Orchestration | 多 Agent 协作、任务路由 | `src/orchestration/graph.py` | `docs/product-specs/orchestration.md` |
| Safety | 输入过滤、输出审查、沙箱 | `src/safety/guard.py` | `docs/product-specs/safety.md` |
| Observability | 日志、追踪、指标、告警 | `src/observability/tracer.py` | `docs/product-specs/observability.md` |

## 分层模型

严格单向依赖：`Types → Config → Gateway → Tools → Memory → Reasoning → Orchestration → API`

跨层规则：
- 低层不能依赖高层
- 每层通过接口（Protocol / ABC）通信，不直接引用实现
- 外部 API 调用必须通过 Gateway 层

## 横切接口

| 关注点 | 允许的边界 | 备注 |
|-------|-----------|------|
| 日志与 tracing | `src/observability/` | 结构化 JSON 日志，含 trace_id，不允许临时 console |
| Auth | `src/safety/auth.py` | JWT token，会话级鉴权 |
| 外部 API | `src/gateway/client.py` | 限流 + 指数退避 + fallback |
| 配置 | `config/` | 环境变量 + YAML，敏感值走 secrets manager |

## 当前热点

- Gateway 层：多模型 fallback 逻辑需要在实际 API 调用后验证
- Memory 层：向量存储选型待定（Chroma vs Qdrant）
- Safety 层：Prompt Injection 检测方案待实现

## 变更检查

当你修改了会影响架构的代码：

1. 如果领域地图或允许边界变了，更新这份文件。
2. 如果背后的设计理由变了，更新 `docs/design-docs/` 里的相关文档。
3. 如果规则应该机械执行，补一个可执行检查到 `scripts/`。
