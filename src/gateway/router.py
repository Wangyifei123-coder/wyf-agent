"""多模型路由 — 按任务类型选择最优模型"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TaskType(str, Enum):
    CHAT = "chat"
    CODE = "code"
    REASONING = "reasoning"
    SUMMARIZATION = "summarization"
    EXTRACTION = "extraction"


@dataclass
class ModelProfile:
    model: str
    strengths: list[TaskType]
    cost_per_1k_input: float
    cost_per_1k_output: float
    max_context: int


DEFAULT_PROFILES: list[ModelProfile] = [
    ModelProfile(
        model="gpt-4o",
        strengths=[TaskType.CHAT, TaskType.CODE, TaskType.EXTRACTION],
        cost_per_1k_input=0.0025,
        cost_per_1k_output=0.01,
        max_context=128000,
    ),
    ModelProfile(
        model="claude-sonnet-4-20250514",
        strengths=[TaskType.REASONING, TaskType.SUMMARIZATION],
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.015,
        max_context=200000,
    ),
]


class ModelRouter:
    def __init__(self, profiles: list[ModelProfile] | None = None) -> None:
        self.profiles = profiles or DEFAULT_PROFILES

    def route(self, task_type: TaskType, prefer_cost: bool = False) -> str:
        candidates = [p for p in self.profiles if task_type in p.strengths]
        if not candidates:
            candidates = self.profiles

        if prefer_cost:
            candidates.sort(key=lambda p: p.cost_per_1k_input)
        else:
            candidates.sort(key=lambda p: p.max_context, reverse=True)

        return candidates[0].model
