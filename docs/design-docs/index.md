# Design Docs Index

存放设计决策记录 (ADR - Architecture Decision Records)。

## 格式

每个决策用一个 Markdown 文件记录：

```markdown
# ADR-NNN: 决策标题

## 状态
accepted / proposed / deprecated

## 背景
为什么需要做这个决策

## 决策
我们选择了什么

## 后果
这个决策带来的影响
```

## 现有决策

| ADR | 标题 | 状态 |
|-----|------|------|
| ADR-001 | 使用 LiteLLM 作为统一 LLM 网关 | accepted |
| ADR-002 | 自建 ReAct 引擎而非使用 LangGraph | accepted |
| ADR-003 | 使用 ChromaDB 作为向量存储 | proposed |
