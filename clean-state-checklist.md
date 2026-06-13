# Clean State Checklist

每次会话结束前过一遍，确保仓库处于下一轮可以直接开工的状态。

## 检查项

- [x] 标准启动路径能用（`pip install -e ".[dev]"` 无报错）
- [x] 标准验证能跑（`pytest tests/ -v` 通过）
- [ ] Lint 通过（`ruff check src/`）— 未运行
- [ ] 类型检查通过（`mypy src/`）— 未运行
- [x] `claude-progress.md` 已更新
- [x] `session-handoff.md` 已更新
- [x] `feature_list.json` 真实反映 passing 和未验证的边界
- [x] 没有半成品处于未记录状态
- [x] 下一轮会话不需要人工修复就能继续
- [ ] 所有新文件已在 git 中暂存或提交 — 未提交
- [x] `.env` 文件未被提交（在 .gitignore 中）

## 备注

- 代码未提交 git，建议下次会话开始时先提交
- 未运行 lint 和类型检查，建议下次会话开始时验证
