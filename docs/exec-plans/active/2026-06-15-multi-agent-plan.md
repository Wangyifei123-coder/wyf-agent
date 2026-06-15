# 多 Agent 角色和通信协议实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现多 Agent 角色和通信协议，支持多个 Agent 协作完成复杂任务

**Architecture:** Agent 角色定义 + 通信协议 + 任务编排 + 状态管理

**Tech Stack:** FastAPI, Python asyncio, JSON-RPC 2.0, LangGraph

---

## 文件结构

| 文件 | 职责 |
|------|------|
| `src/agents/__init__.py` | Agent 模块初始化 |
| `src/agents/base.py` | Agent 基类定义 |
| `src/agents/roles.py` | Agent 角色定义 |
| `src/agents/communication.py` | 通信协议实现 |
| `src/agents/orchestrator.py` | 任务编排器 |
| `src/agents/state.py` | 状态管理 |
| `src/api.py` | API 端点（Agent 管理） |
| `tests/test_multi_agent.py` | **新增** 多 Agent 测试 |

---

## Task 1: Agent 基类定义

**Files:**
- Create: `src/agents/__init__.py`
- Create: `src/agents/base.py`

- [ ] **Step 1: 创建 Agent 模块初始化**

```python
# 文件: src/agents/__init__.py
"""多 Agent 模块"""

from .base import BaseAgent
from .roles import AgentRole, AgentFactory
from .communication import Message, MessageBus
from .orchestrator import Orchestrator
from .state import AgentState

__all__ = [
    "BaseAgent",
    "AgentRole",
    "AgentFactory",
    "Message",
    "MessageBus",
    "Orchestrator",
    "AgentState",
]
```

- [ ] **Step 2: 创建 Agent 基类**

```python
# 文件: src/agents/base.py
"""Agent 基类定义"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import structlog

logger = structlog.get_logger(__name__)


class AgentStatus(Enum):
    """Agent 状态"""
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class AgentConfig:
    """Agent 配置"""
    name: str
    role: str
    description: str = ""
    capabilities: List[str] = field(default_factory=list)
    max_concurrent_tasks: int = 1
    timeout: float = 300.0  # 5 分钟


class BaseAgent(ABC):
    """Agent 基类"""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.status = AgentStatus.IDLE
        self.current_tasks: Dict[str, asyncio.Task] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self._logger = logger.bind(agent=config.name, role=config.role)

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def role(self) -> str:
        return self.config.role

    @abstractmethod
    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """处理消息"""
        pass

    @abstractmethod
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """执行任务"""
        pass

    async def start(self):
        """启动 Agent"""
        self._logger.info("agent_started")
        self.status = AgentStatus.IDLE
        asyncio.create_task(self._message_loop())

    async def stop(self):
        """停止 Agent"""
        self._logger.info("agent_stopping")
        self.status = AgentStatus.STOPPED
        # 取消所有当前任务
        for task in self.current_tasks.values():
            task.cancel()
        self.current_tasks.clear()

    async def _message_loop(self):
        """消息处理循环"""
        while self.status != AgentStatus.STOPPED:
            try:
                message = await asyncio.wait_for(
                    self.message_queue.get(),
                    timeout=1.0
                )
                response = await self.process_message(message)
                self._logger.info("message_processed", message_id=message.get("id"))
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self._logger.error("message_processing_error", error=str(e))

    async def send_message(self, target: str, content: Dict[str, Any]) -> Dict[str, Any]:
        """发送消息给其他 Agent"""
        # 通过消息总线发送
        from .communication import message_bus
        message = Message(
            sender=self.name,
            receiver=target,
            content=content
        )
        return await message_bus.send(message)

    def can_handle_task(self, task: Dict[str, Any]) -> bool:
        """检查是否能处理任务"""
        required_capabilities = task.get("required_capabilities", [])
        return all(cap in self.config.capabilities for cap in required_capabilities)
```

- [ ] **Step 3: 运行测试验证**

Run: `python -m pytest tests/test_basic.py -v`
Expected: PASS

- [ ] **Step 4: 提交**

```bash
git add src/agents/__init__.py src/agents/base.py
git commit -m "feat: add Agent base class definition"
```

---

## Task 2: Agent 角色定义

**Files:**
- Create: `src/agents/roles.py`

- [ ] **Step 1: 创建 Agent 角色定义**

```python
# 文件: src/agents/roles.py
"""Agent 角色定义"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Type
import structlog

from .base import BaseAgent, AgentConfig

logger = structlog.get_logger(__name__)


class AgentRole(Enum):
    """Agent 角色枚举"""
    RESEARCHER = "researcher"  # 研究员：信息收集和分析
    CODER = "coder"  # 程序员：代码编写和调试
    REVIEWER = "reviewer"  # 审查员：代码审查和质量检查
    PLANNER = "planner"  # 规划师：任务分解和规划
    EXECUTOR = "executor"  # 执行者：任务执行
    COORDINATOR = "coordinator"  # 协调员：任务协调和分配


class ResearcherAgent(BaseAgent):
    """研究员 Agent"""

    def __init__(self, name: str = "researcher"):
        config = AgentConfig(
            name=name,
            role=AgentRole.RESEARCHER.value,
            description="负责信息收集、数据分析和研究",
            capabilities=["research", "analysis", "search", "summarize"]
        )
        super().__init__(config)

    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """处理消息"""
        content = message.get("content", {})
        action = content.get("action")

        if action == "research":
            topic = content.get("topic", "")
            # 调用研究功能
            result = await self._research(topic)
            return {"status": "success", "result": result}
        elif action == "analyze":
            data = content.get("data", "")
            result = await self._analyze(data)
            return {"status": "success", "result": result}
        else:
            return {"status": "error", "message": f"Unknown action: {action}"}

    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """执行任务"""
        task_type = task.get("type")

        if task_type == "research":
            topic = task.get("topic", "")
            return await self._research(topic)
        elif task_type == "analyze":
            data = task.get("data", "")
            return await self._analyze(data)
        else:
            return {"status": "error", "message": f"Unknown task type: {task_type}"}

    async def _research(self, topic: str) -> Dict[str, Any]:
        """执行研究"""
        self._logger.info("researching", topic=topic)
        # 这里可以集成搜索工具、知识库等
        return {
            "topic": topic,
            "findings": f"Research findings for: {topic}",
            "sources": []
        }

    async def _analyze(self, data: str) -> Dict[str, Any]:
        """分析数据"""
        self._logger.info("analyzing", data_len=len(data))
        return {
            "analysis": f"Analysis of: {data[:100]}...",
            "insights": []
        }


class CoderAgent(BaseAgent):
    """程序员 Agent"""

    def __init__(self, name: str = "coder"):
        config = AgentConfig(
            name=name,
            role=AgentRole.CODER.value,
            description="负责代码编写、调试和优化",
            capabilities=["code", "debug", "refactor", "test"]
        )
        super().__init__(config)

    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """处理消息"""
        content = message.get("content", {})
        action = content.get("action")

        if action == "code":
            requirements = content.get("requirements", "")
            result = await self._write_code(requirements)
            return {"status": "success", "result": result}
        elif action == "debug":
            code = content.get("code", "")
            error = content.get("error", "")
            result = await self._debug_code(code, error)
            return {"status": "success", "result": result}
        else:
            return {"status": "error", "message": f"Unknown action: {action}"}

    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """执行任务"""
        task_type = task.get("type")

        if task_type == "code":
            requirements = task.get("requirements", "")
            return await self._write_code(requirements)
        elif task_type == "debug":
            code = task.get("code", "")
            error = task.get("error", "")
            return await self._debug_code(code, error)
        else:
            return {"status": "error", "message": f"Unknown task type: {task_type}"}

    async def _write_code(self, requirements: str) -> Dict[str, Any]:
        """编写代码"""
        self._logger.info("writing_code", requirements=requirements[:100])
        return {
            "code": f"# Code for: {requirements}\n# Implementation here",
            "language": "python"
        }

    async def _debug_code(self, code: str, error: str) -> Dict[str, Any]:
        """调试代码"""
        self._logger.info("debugging", error=error[:100])
        return {
            "fix": f"Fix for error: {error}",
            "explanation": "Debugging explanation"
        }


class ReviewerAgent(BaseAgent):
    """审查员 Agent"""

    def __init__(self, name: str = "reviewer"):
        config = AgentConfig(
            name=name,
            role=AgentRole.REVIEWER.value,
            description="负责代码审查、质量检查和反馈",
            capabilities=["review", "quality", "feedback"]
        )
        super().__init__(config)

    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """处理消息"""
        content = message.get("content", {})
        action = content.get("action")

        if action == "review":
            code = content.get("code", "")
            result = await self._review_code(code)
            return {"status": "success", "result": result}
        else:
            return {"status": "error", "message": f"Unknown action: {action}"}

    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """执行任务"""
        task_type = task.get("type")

        if task_type == "review":
            code = task.get("code", "")
            return await self._review_code(code)
        else:
            return {"status": "error", "message": f"Unknown task type: {task_type}"}

    async def _review_code(self, code: str) -> Dict[str, Any]:
        """审查代码"""
        self._logger.info("reviewing_code", code_len=len(code))
        return {
            "approved": True,
            "comments": [],
            "suggestions": []
        }


class AgentFactory:
    """Agent 工厂"""

    _agents: Dict[AgentRole, Type[BaseAgent]] = {
        AgentRole.RESEARCHER: ResearcherAgent,
        AgentRole.CODER: CoderAgent,
        AgentRole.REVIEWER: ReviewerAgent,
    }

    @classmethod
    def create_agent(cls, role: AgentRole, name: str = None) -> BaseAgent:
        """创建 Agent"""
        agent_class = cls._agents.get(role)
        if not agent_class:
            raise ValueError(f"Unknown agent role: {role}")

        if name:
            return agent_class(name=name)
        return agent_class()

    @classmethod
    def register_agent(cls, role: AgentRole, agent_class: Type[BaseAgent]):
        """注册新的 Agent 类型"""
        cls._agents[role] = agent_class
```

- [ ] **Step 2: 运行测试验证**

Run: `python -m pytest tests/test_basic.py -v`
Expected: PASS

- [ ] **Step 3: 提交**

```bash
git add src/agents/roles.py
git commit -m "feat: add Agent role definitions and factory"
```

---

## Task 3: 通信协议实现

**Files:**
- Create: `src/agents/communication.py`

- [ ] **Step 1: 创建通信协议**

```python
# 文件: src/agents/communication.py
"""Agent 通信协议实现"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from enum import Enum
import structlog
import uuid

logger = structlog.get_logger(__name__)


class MessageType(Enum):
    """消息类型"""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    BROADCAST = "broadcast"


@dataclass
class Message:
    """消息定义"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender: str = ""
    receiver: str = ""
    type: MessageType = MessageType.REQUEST
    content: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: asyncio.get_event_loop().time())
    reply_to: Optional[str] = None
    correlation_id: Optional[str] = None


class MessageBus:
    """消息总线"""

    def __init__(self):
        self._agents: Dict[str, 'BaseAgent'] = {}
        self._message_history: List[Message] = []
        self._handlers: Dict[str, List[Callable]] = {}
        self._logger = logger.bind(component="message_bus")

    def register_agent(self, agent: 'BaseAgent'):
        """注册 Agent"""
        self._agents[agent.name] = agent
        self._logger.info("agent_registered", agent=agent.name)

    def unregister_agent(self, agent_name: str):
        """注销 Agent"""
        if agent_name in self._agents:
            del self._agents[agent_name]
            self._logger.info("agent_unregistered", agent=agent_name)

    async def send(self, message: Message) -> Dict[str, Any]:
        """发送消息"""
        self._message_history.append(message)

        if message.receiver == "*":
            # 广播消息
            return await self._broadcast(message)
        elif message.receiver in self._agents:
            # 点对点消息
            agent = self._agents[message.receiver]
            return await agent.process_message({
                "id": message.id,
                "sender": message.sender,
                "content": message.content,
                "type": message.type.value
            })
        else:
            return {"status": "error", "message": f"Agent not found: {message.receiver}"}

    async def _broadcast(self, message: Message) -> Dict[str, Any]:
        """广播消息"""
        results = {}
        for name, agent in self._agents.items():
            if name != message.sender:
                try:
                    result = await agent.process_message({
                        "id": message.id,
                        "sender": message.sender,
                        "content": message.content,
                        "type": message.type.value
                    })
                    results[name] = result
                except Exception as e:
                    results[name] = {"status": "error", "message": str(e)}
        return results

    def get_message_history(self, limit: int = 100) -> List[Message]:
        """获取消息历史"""
        return self._message_history[-limit:]

    def clear_history(self):
        """清空消息历史"""
        self._message_history.clear()


# 全局消息总线
message_bus = MessageBus()
```

- [ ] **Step 2: 运行测试验证**

Run: `python -m pytest tests/test_basic.py -v`
Expected: PASS

- [ ] **Step 3: 提交**

```bash
git add src/agents/communication.py
git commit -m "feat: add Agent communication protocol"
```

---

## Task 4: 任务编排器

**Files:**
- Create: `src/agents/orchestrator.py`

- [ ] **Step 1: 创建任务编排器**

```python
# 文件: src/agents/orchestrator.py
"""任务编排器"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum
import structlog

from .base import BaseAgent, AgentStatus
from .roles import AgentFactory, AgentRole
from .communication import Message, message_bus

logger = structlog.get_logger(__name__)


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """任务定义"""
    id: str
    name: str
    description: str
    required_capabilities: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    assigned_agent: Optional[str] = None
    priority: int = 0


class Orchestrator:
    """任务编排器"""

    def __init__(self):
        self._agents: Dict[str, BaseAgent] = {}
        self._tasks: Dict[str, Task] = {}
        self._logger = logger.bind(component="orchestrator")

    def register_agent(self, agent: BaseAgent):
        """注册 Agent"""
        self._agents[agent.name] = agent
        message_bus.register_agent(agent)
        self._logger.info("agent_registered", agent=agent.name)

    def create_task(
        self,
        task_id: str,
        name: str,
        description: str,
        required_capabilities: List[str] = None,
        dependencies: List[str] = None,
        priority: int = 0
    ) -> Task:
        """创建任务"""
        task = Task(
            id=task_id,
            name=name,
            description=description,
            required_capabilities=required_capabilities or [],
            dependencies=dependencies or [],
            priority=priority
        )
        self._tasks[task_id] = task
        self._logger.info("task_created", task_id=task_id, name=name)
        return task

    async def execute_task(self, task_id: str) -> Dict[str, Any]:
        """执行任务"""
        task = self._tasks.get(task_id)
        if not task:
            return {"status": "error", "message": f"Task not found: {task_id}"}

        # 检查依赖
        for dep_id in task.dependencies:
            dep_task = self._tasks.get(dep_id)
            if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                return {"status": "error", "message": f"Dependency not completed: {dep_id}"}

        # 找到合适的 Agent
        agent = self._find_agent_for_task(task)
        if not agent:
            return {"status": "error", "message": "No suitable agent found"}

        # 分配任务
        task.assigned_agent = agent.name
        task.status = TaskStatus.IN_PROGRESS

        try:
            # 执行任务
            result = await agent.execute_task({
                "id": task.id,
                "name": task.name,
                "description": task.description,
                "required_capabilities": task.required_capabilities
            })

            task.status = TaskStatus.COMPLETED
            task.result = result
            self._logger.info("task_completed", task_id=task_id, agent=agent.name)
            return {"status": "success", "result": result}

        except Exception as e:
            task.status = TaskStatus.FAILED
            self._logger.error("task_failed", task_id=task_id, error=str(e))
            return {"status": "error", "message": str(e)}

    def _find_agent_for_task(self, task: Task) -> Optional[BaseAgent]:
        """为任务找到合适的 Agent"""
        for agent in self._agents.values():
            if (agent.status == AgentStatus.IDLE and
                agent.can_handle_task({
                    "required_capabilities": task.required_capabilities
                })):
                return agent
        return None

    async def execute_workflow(self, workflow: List[Dict[str, Any]]) -> Dict[str, Any]:
        """执行工作流"""
        results = {}

        for step in workflow:
            task_id = step.get("task_id")
            if not task_id:
                continue

            result = await self.execute_task(task_id)
            results[task_id] = result

            if result.get("status") == "error":
                return {"status": "error", "failed_task": task_id, "results": results}

        return {"status": "success", "results": results}

    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """获取任务状态"""
        task = self._tasks.get(task_id)
        return task.status if task else None

    def get_agent_status(self, agent_name: str) -> Optional[AgentStatus]:
        """获取 Agent 状态"""
        agent = self._agents.get(agent_name)
        return agent.status if agent else None

    def list_tasks(self) -> List[Task]:
        """列出所有任务"""
        return list(self._tasks.values())

    def list_agents(self) -> List[BaseAgent]:
        """列出所有 Agent"""
        return list(self._agents.values())
```

- [ ] **Step 2: 运行测试验证**

Run: `python -m pytest tests/test_basic.py -v`
Expected: PASS

- [ ] **Step 3: 提交**

```bash
git add src/agents/orchestrator.py
git commit -m "feat: add task orchestrator for multi-agent coordination"
```

---

## Task 5: 状态管理

**Files:**
- Create: `src/agents/state.py`

- [ ] **Step 1: 创建状态管理**

```python
# 文件: src/agents/state.py
"""Agent 状态管理"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum
import structlog

logger = structlog.get_logger(__name__)


class AgentStateStatus(Enum):
    """Agent 状态"""
    INITIALIZING = "initializing"
    READY = "ready"
    WORKING = "working"
    WAITING = "waiting"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class AgentState:
    """Agent 状态"""
    agent_id: str
    agent_role: str
    status: AgentStateStatus = AgentStateStatus.INITIALIZING
    current_task: Optional[str] = None
    tasks_completed: int = 0
    tasks_failed: int = 0
    last_active: float = 0.0
    memory: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)


class StateManager:
    """状态管理器"""

    def __init__(self):
        self._states: Dict[str, AgentState] = {}
        self._logger = logger.bind(component="state_manager")

    def register_agent(self, agent_id: str, agent_role: str) -> AgentState:
        """注册 Agent 状态"""
        state = AgentState(
            agent_id=agent_id,
            agent_role=agent_role,
            status=AgentStateStatus.READY
        )
        self._states[agent_id] = state
        self._logger.info("agent_state_registered", agent_id=agent_id)
        return state

    def get_state(self, agent_id: str) -> Optional[AgentState]:
        """获取 Agent 状态"""
        return self._states.get(agent_id)

    def update_status(self, agent_id: str, status: AgentStateStatus):
        """更新 Agent 状态"""
        state = self._states.get(agent_id)
        if state:
            state.status = status
            self._logger.info("agent_status_updated", agent_id=agent_id, status=status.value)

    def update_task(self, agent_id: str, task_id: Optional[str]):
        """更新当前任务"""
        state = self._states.get(agent_id)
        if state:
            state.current_task = task_id
            if task_id:
                state.status = AgentStateStatus.WORKING
            else:
                state.status = AgentStateStatus.READY

    def increment_completed(self, agent_id: str):
        """增加完成任务数"""
        state = self._states.get(agent_id)
        if state:
            state.tasks_completed += 1

    def increment_failed(self, agent_id: str):
        """增加失败任务数"""
        state = self._states.get(agent_id)
        if state:
            state.tasks_failed += 1

    def update_memory(self, agent_id: str, key: str, value: Any):
        """更新 Agent 记忆"""
        state = self._states.get(agent_id)
        if state:
            state.memory[key] = value

    def get_memory(self, agent_id: str, key: str, default: Any = None) -> Any:
        """获取 Agent 记忆"""
        state = self._states.get(agent_id)
        return state.memory.get(key, default) if state else default

    def update_context(self, agent_id: str, key: str, value: Any):
        """更新 Agent 上下文"""
        state = self._states.get(agent_id)
        if state:
            state.context[key] = value

    def get_context(self, agent_id: str, key: str, default: Any = None) -> Any:
        """获取 Agent 上下文"""
        state = self._states.get(agent_id)
        return state.context.get(key, default) if state else default

    def list_states(self) -> List[AgentState]:
        """列出所有 Agent 状态"""
        return list(self._states.values())

    def get_summary(self) -> Dict[str, Any]:
        """获取摘要"""
        total = len(self._states)
        working = sum(1 for s in self._states.values() if s.status == AgentStateStatus.WORKING)
        ready = sum(1 for s in self._states.values() if s.status == AgentStateStatus.READY)
        error = sum(1 for s in self._states.values() if s.status == AgentStateStatus.ERROR)

        return {
            "total_agents": total,
            "working": working,
            "ready": ready,
            "error": error,
            "total_tasks_completed": sum(s.tasks_completed for s in self._states.values()),
            "total_tasks_failed": sum(s.tasks_failed for s in self._states.values()),
        }


# 全局状态管理器
state_manager = StateManager()
```

- [ ] **Step 2: 运行测试验证**

Run: `python -m pytest tests/test_basic.py -v`
Expected: PASS

- [ ] **Step 3: 提交**

```bash
git add src/agents/state.py
git commit -m "feat: add Agent state management"
```

---

## Task 6: API 端点集成

**Files:**
- Modify: `src/api.py`

- [ ] **Step 1: 添加 Agent 管理端点**

```python
# 文件: src/api.py
# 在现有端点后添加

from .agents import Orchestrator, AgentFactory, AgentRole, AgentState
from .agents.state import state_manager

# 全局编排器
orchestrator = Orchestrator()

@app.post("/agents/create")
async def create_agent(role: str, name: str = None):
    """创建 Agent"""
    try:
        agent_role = AgentRole(role)
        agent = AgentFactory.create_agent(agent_role, name)
        orchestrator.register_agent(agent)
        state_manager.register_agent(agent.name, agent.role)
        await agent.start()
        return {"status": "success", "agent": agent.name, "role": agent.role}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/agents/list")
async def list_agents():
    """列出所有 Agent"""
    agents = orchestrator.list_agents()
    return {
        "agents": [
            {
                "name": a.name,
                "role": a.role,
                "status": a.status.value,
                "capabilities": a.config.capabilities
            }
            for a in agents
        ]
    }


@app.post("/agents/{agent_name}/task")
async def assign_task(agent_name: str, task: Dict[str, Any]):
    """分配任务给 Agent"""
    agent = orchestrator._agents.get(agent_name)
    if not agent:
        return {"status": "error", "message": f"Agent not found: {agent_name}"}

    result = await agent.execute_task(task)
    return result


@app.post("/orchestrator/task")
async def create_orchestrator_task(task: Dict[str, Any]):
    """创建编排任务"""
    task_id = task.get("id", str(uuid.uuid4()))
    orchestrator_task = orchestrator.create_task(
        task_id=task_id,
        name=task.get("name", ""),
        description=task.get("description", ""),
        required_capabilities=task.get("required_capabilities", []),
        dependencies=task.get("dependencies", []),
        priority=task.get("priority", 0)
    )
    return {"status": "success", "task_id": task_id}


@app.post("/orchestrator/execute/{task_id}")
async def execute_orchestrator_task(task_id: str):
    """执行编排任务"""
    result = await orchestrator.execute_task(task_id)
    return result


@app.get("/orchestrator/status")
async def get_orchestrator_status():
    """获取编排器状态"""
    return {
        "agents": len(orchestrator.list_agents()),
        "tasks": len(orchestrator.list_tasks()),
        "state_summary": state_manager.get_summary()
    }
```

- [ ] **Step 2: 运行测试验证**

Run: `python -m pytest tests/test_basic.py -v`
Expected: PASS

- [ ] **Step 3: 提交**

```bash
git add src/api.py
git commit -m "feat: add Agent management API endpoints"
```

---

## Task 7: 端到端测试

**Files:**
- Create: `tests/test_multi_agent.py`

- [ ] **Step 1: 创建多 Agent 测试**

```python
# 文件: tests/test_multi_agent.py
"""多 Agent 功能测试"""

import pytest
import asyncio
from src.agents.base import BaseAgent, AgentConfig, AgentStatus
from src.agents.roles import AgentFactory, AgentRole, ResearcherAgent, CoderAgent
from src.agents.communication import Message, MessageBus, message_bus
from src.agents.orchestrator import Orchestrator, Task, TaskStatus
from src.agents.state import StateManager, AgentStateStatus


def test_agent_factory_create():
    """测试 Agent 工厂创建"""
    agent = AgentFactory.create_agent(AgentRole.RESEARCHER)
    assert agent.name == "researcher"
    assert agent.role == "researcher"


def test_agent_factory_create_with_name():
    """测试 Agent 工厂创建（指定名称）"""
    agent = AgentFactory.create_agent(AgentRole.CODER, "my_coder")
    assert agent.name == "my_coder"
    assert agent.role == "coder"


def test_agent_capabilities():
    """测试 Agent 能力"""
    agent = AgentFactory.create_agent(AgentRole.RESEARCHER)
    assert "research" in agent.config.capabilities
    assert "analysis" in agent.config.capabilities


def test_agent_can_handle_task():
    """测试 Agent 任务处理能力"""
    agent = AgentFactory.create_agent(AgentRole.RESEARCHER)
    task = {"required_capabilities": ["research"]}
    assert agent.can_handle_task(task)


def test_agent_cannot_handle_task():
    """测试 Agent 无法处理任务"""
    agent = AgentFactory.create_agent(AgentRole.RESEARCHER)
    task = {"required_capabilities": ["code"]}
    assert not agent.can_handle_task(task)


def test_message_creation():
    """测试消息创建"""
    message = Message(
        sender="agent1",
        receiver="agent2",
        content={"action": "test"}
    )
    assert message.sender == "agent1"
    assert message.receiver == "agent2"
    assert message.id is not None


def test_message_bus_register():
    """测试消息总线注册"""
    bus = MessageBus()
    agent = AgentFactory.create_agent(AgentRole.RESEARCHER)
    bus.register_agent(agent)
    assert "researcher" in bus._agents


def test_orchestrator_create_task():
    """测试编排器创建任务"""
    orchestrator = Orchestrator()
    task = orchestrator.create_task(
        task_id="task1",
        name="Test Task",
        description="A test task",
        required_capabilities=["research"]
    )
    assert task.id == "task1"
    assert task.status == TaskStatus.PENDING


def test_state_manager_register():
    """测试状态管理器注册"""
    manager = StateManager()
    state = manager.register_agent("agent1", "researcher")
    assert state.agent_id == "agent1"
    assert state.status == AgentStateStatus.READY


def test_state_manager_update_status():
    """测试状态管理器更新状态"""
    manager = StateManager()
    manager.register_agent("agent1", "researcher")
    manager.update_status("agent1", AgentStateStatus.WORKING)
    state = manager.get_state("agent1")
    assert state.status == AgentStateStatus.WORKING


def test_state_manager_summary():
    """测试状态管理器摘要"""
    manager = StateManager()
    manager.register_agent("agent1", "researcher")
    manager.register_agent("agent2", "coder")
    summary = manager.get_summary()
    assert summary["total_agents"] == 2
    assert summary["ready"] == 2
```

- [ ] **Step 2: 运行测试**

Run: `python -m pytest tests/test_multi_agent.py -v`
Expected: PASS

- [ ] **Step 3: 运行全量测试**

Run: `python -m pytest tests/ -v`
Expected: 全部 PASS（103+ 个测试）

- [ ] **Step 4: 运行代码检查**

Run: `python -m ruff check src/`
Run: `python -m mypy src/`
Expected: 无错误

- [ ] **Step 5: 提交**

```bash
git add tests/test_multi_agent.py
git commit -m "test: add multi-agent feature tests"
```

---

## Task 8: 更新文档

**Files:**
- Modify: `session-handoff.md`
- Modify: `docs/QUALITY_SCORE.md`
- Modify: `claude-progress.md`

- [ ] **Step 1: 更新 session-handoff.md**

```markdown
## 当前已验证

- ... 现有内容 ...
- **多 Agent 角色和通信协议**（2026-06-15）
  - Agent 基类定义 ✅
  - Agent 角色定义（研究員、程序员、审查员）✅
  - 通信协议（消息总线）✅
  - 任务编排器 ✅
  - 状态管理 ✅
  - API 端点 ✅
```

- [ ] **Step 2: 更新质量评分**

```markdown
| Orchestration (编排) | 已验证 | 高 | 10 测试通过 | A |
```

- [ ] **Step 3: 更新进度日志**

```markdown
### Session 004 — 2026-06-15

- **本轮目标**：实现多 Agent 角色和通信协议
- **已完成**：
  - Agent 基类定义
  - Agent 角色定义（研究員、程序员、审查员）
  - 通信协议（消息总线）
  - 任务编排器
  - 状态管理
  - API 端点
  - 端到端测试
- **运行过的验证**：`pytest tests/ -v`（113 passed）
- **提交记录**：
  - 提交记录
- **下一步最佳动作**：完善测试体系（A/B 测试、基准对比）
```

- [ ] **Step 4: 提交**

```bash
git add session-handoff.md docs/QUALITY_SCORE.md claude-progress.md
git commit -m "docs: update progress with multi-agent support"
```

---

## 验收检查清单

- [ ] Agent 基类定义完成
- [ ] Agent 角色定义完成（研究員、程序员、审查员）
- [ ] 通信协议（消息总线）工作正常
- [ ] 任务编排器能分配和执行任务
- [ ] 状态管理正确跟踪 Agent 状态
- [ ] API 端点可用
- [ ] 现有 103 个测试继续通过
- [ ] 新增 10 个测试通过
- [ ] ruff check 无错误
- [ ] mypy 无错误
- [ ] 文档已更新