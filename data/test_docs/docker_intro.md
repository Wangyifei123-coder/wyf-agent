# Docker 容器技术

Docker 是一个开源的应用容器引擎，让开发者可以打包应用及其依赖包到一个可移植的容器中。

## 核心概念

### 镜像（Image）
- 只读模板，包含运行应用所需的一切
- 分层存储，共享基础层
- 通过 Dockerfile 构建

### 容器（Container）
- 镜像的运行实例
- 相互隔离，拥有独立的文件系统
- 可以启动、停止、删除

### 仓库（Registry）
- 存储和分发镜像
- Docker Hub 是最大的公共仓库
- 支持私有仓库

## Dockerfile 最佳实践

```dockerfile
# 使用多阶段构建减小镜像体积
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .
CMD ["python", "app.py"]
```

## Docker Compose

用于定义和运行多容器应用：

```yaml
services:
  web:
    build: .
    ports:
      - "8080:8080"
  db:
    image: postgres:15
    environment:
      POSTGRES_PASSWORD: secret
```

## 常用命令

| 命令 | 说明 |
|------|------|
| `docker build -t name .` | 构建镜像 |
| `docker run -d -p 8080:8080 name` | 运行容器 |
| `docker ps` | 查看运行中的容器 |
| `docker logs container_id` | 查看日志 |
| `docker exec -it container_id bash` | 进入容器 |
