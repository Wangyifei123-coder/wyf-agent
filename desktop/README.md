# WYF Agent Desktop

桌面端 AI 智能助手

## 开发环境

### 前置要求
- Node.js 18+
- Python 3.11+
- 后端服务运行在 http://localhost:8080

### 安装依赖
```bash
cd desktop
npm install
cd src/renderer
npm install
```

### 启动开发
```bash
# 终端1：启动后端
cd D:\PythonProject\my-agent\wyf-agent
python -m uvicorn src.api:app --port 8080

# 终端2：启动前端开发服务器
cd desktop/src/renderer
npm start

# 终端3：启动 Electron
cd desktop
npm run dev:main
```

### 或者一键启动
```bash
cd desktop
npm run dev
```

## 构建打包
```bash
cd desktop
npm run build:win
```

## 快捷键

- `Ctrl+Shift+A` - 唤起窗口
- `Enter` - 发送消息
- `Shift+Enter` - 换行

## 功能特性

- 独立窗口聊天界面
- 系统托盘常驻
- 全局快捷键唤起
- 支持图片上传
- 流式输出
- Markdown 渲染
