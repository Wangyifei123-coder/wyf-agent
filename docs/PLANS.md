# PLANS.md — 计划生命周期

## 计划状态

- `draft` — 草案，待评审
- `active` — 当前正在执行（放在 `docs/exec-plans/active/`）
- `blocked` — 有依赖未解决
- `completed` — 已完成并验证（移到 `docs/exec-plans/completed/`）
- `abandoned` — 决定不做

## 规则

- 同一时间最多一个 `active` 计划
- 计划必须包含：目标、验收标准、预估步骤、风险
- 完成后必须有可运行证据
- 计划文件用日期前缀命名：`2026-06-13-initial-setup.md`

## 当前计划

| 计划 | 状态 | 描述 |
|------|------|------|
| 初始搭建 | active | 完成 7 层架构骨架，验证基础对话流程 |

## 下一步

1. 配置 API Key
2. 实现第一个工具（search_knowledge_base）
3. 端到端对话测试
4. 第一轮评估
