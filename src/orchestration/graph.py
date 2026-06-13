"""编排图 — 状态机驱动的多 Agent 协作"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine

import structlog

logger = structlog.get_logger(__name__)


class NodeStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Node:
    name: str
    handler: Callable[[dict[str, Any]], Coroutine[Any, Any, dict[str, Any]]]
    depends_on: list[str] = field(default_factory=list)
    status: NodeStatus = NodeStatus.PENDING
    result: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphResult:
    outputs: dict[str, dict[str, Any]]
    node_statuses: dict[str, NodeStatus]
    success: bool


class OrchestrationGraph:
    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}

    def add_node(
        self,
        name: str,
        handler: Callable[[dict[str, Any]], Coroutine[Any, Any, dict[str, Any]]],
        depends_on: list[str] | None = None,
    ) -> None:
        self._nodes[name] = Node(name=name, handler=handler, depends_on=depends_on or [])

    async def execute(self, initial_input: dict[str, Any]) -> GraphResult:
        outputs: dict[str, dict[str, Any]] = {}
        completed: set[str] = set()

        for _ in range(len(self._nodes)):
            runnable = [
                n for n in self._nodes.values()
                if n.status == NodeStatus.PENDING and all(d in completed for d in n.depends_on)
            ]
            if not runnable:
                break

            for node in runnable:
                node.status = NodeStatus.RUNNING
                merged_input = {**initial_input}
                for dep in node.depends_on:
                    merged_input.update(outputs.get(dep, {}))

                try:
                    node.result = await node.handler(merged_input)
                    node.status = NodeStatus.SUCCESS
                    outputs[node.name] = node.result
                    completed.add(node.name)
                    logger.info("node_success", node=node.name)
                except Exception as e:
                    node.status = NodeStatus.FAILED
                    logger.error("node_failed", node=node.name, error=str(e))

        return GraphResult(
            outputs=outputs,
            node_statuses={n.name: n.status for n in self._nodes.values()},
            success=all(n.status == NodeStatus.SUCCESS for n in self._nodes.values()),
        )
