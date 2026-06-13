# AGENTS.md — WYF Agent

这个仓库是一个企业级 AI Agent 项目。保持这个文件简短，把它当成路由层，深层规则指向 `docs/`。

## 开工流程

改代码前先做这些事：

1. 用 `pwd` 确认仓库根目录。
2. 读取 `ARCHITECTURE.md`，理解 7 层架构和依赖规则。
3. 读取 `docs/QUALITY_SCORE.md`，先知道最弱的产品领域和架构层。
4. 读取 `docs/PLANS.md`，再打开当前要执行的 active plan。
5. 读取相关的 `docs/product-specs/` 规格文档。
6. 运行 `./scripts/init.sh` 完成依赖安装和基础验证。
7. 如果基础验证先失败，先修 baseline，再加新范围。

## 路由地图

| 文件 | 用途 |
|------|------|
| `ARCHITECTURE.md` | 7 层架构、领域地图、依赖规则 |
| `docs/PLANS.md` | 计划生命周期与执行计划规则 |
| `docs/QUALITY_SCORE.md` | 产品领域与架构层健康度 |
| `docs/RELIABILITY.md` | 运行信号、benchmark、重启要求 |
| `docs/SECURITY.md` | 密钥、沙箱、鉴权、速率限制 |
| `docs/FRONTEND.md` | UI 约束、设计系统规则 |
| `docs/DESIGN.md` | 设计决策与核心信念 |
| `docs/PRODUCT_SENSE.md` | 产品直觉与用户场景 |
| `docs/design-docs/` | 设计决策记录 (ADR) |
| `docs/product-specs/` | 产品行为规格 |
| `docs/exec-plans/active/` | 当前执行计划 |
| `docs/exec-plans/completed/` | 已完成计划归档 |
| `prompts/` | Prompt 版本管理（system / tools / few_shots） |
| `evals/` | 评估数据集与指标 |
| `src/` | 源代码（7 层架构） |

## 工作约定

- 一次只围绕一个有边界的计划或功能切片工作。
- 不能只靠读代码就宣布完成，必须有可运行证据。
- 只要改了行为，就同步更新对应的产品、计划或可靠性文档。
- Prompt 是代码，纳入版本管理、代码审查、回归测试。
- 工具 schema 是接口契约，变更需向后兼容。
- 如果某类 review feedback 反复出现，升级成机械规则或 linter。
- 生成物放进 `docs/generated/`，外部 reference 放进 `docs/references/`。

## 完成定义

一个改动只有在以下条件都满足时才算完成：

- 目标行为已实现
- 要求的验证真的跑过（`npm run check` + `npm run test`）
- 证据已经挂到相关 plan 或质量文档里
- 受影响的文档仍然是最新的
- 仓库能按标准启动路径干净重启
- Prompt 变更通过回归评估集

## 收尾

结束会话前：

1. 更新当前 active execution plan。
2. 如果产品领域或架构层有明显变化，更新 `docs/QUALITY_SCORE.md`。
3. 如果有延期处理的债务，记到 `docs/exec-plans/tech-debt-tracker.md`。
4. 已完成的计划及时移到 `docs/exec-plans/completed/`。
5. 保证仓库可重启，并留下清晰的下一步动作。
