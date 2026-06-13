# FRONTEND.md — 前端约束

## 当前状态

项目暂无前端，API 优先设计。未来可选方案：

- React + TypeScript（推荐）
- Next.js（如需 SSR）
- Streamlit（快速原型）

## UI 约束（规划中）

- 对话界面支持流式输出
- 推理步骤可折叠展示
- 工具调用结果高亮显示
- 深色/浅色主题切换
- 响应式设计，支持移动端

## 设计系统规则

- 使用统一的设计 token（颜色、间距、字体）
- 组件库优先使用 shadcn/ui 或 Ant Design
- 所有交互元素有 ARIA 标签
- 加载状态和错误状态必须有 UI 反馈
