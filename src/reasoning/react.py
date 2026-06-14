"""ReAct 推理引擎 — 思考 → 行动 → 观察 循环"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import structlog

from ..gateway.client import LLMClient
from ..memory.manager import MemoryManager
from ..tools.registry import ToolRegistry

logger = structlog.get_logger(__name__)


class StepType(StrEnum):
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    ANSWER = "answer"


@dataclass
class Step:
    type: StepType
    content: str
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReActResult:
    answer: str
    steps: list[Step]
    total_iterations: int
    total_tokens: int


REACT_SYSTEM_PROMPT = """You are a reasoning agent that uses the ReAct framework.

For each user question, follow this cycle:
1. THOUGHT: Think about what you need to do
2. ACTION: Call a tool if needed (or skip to ANSWER)
3. OBSERVATION: Review the tool result
4. Repeat until you have enough information

Available tools:
{tools}

Respond in this exact format:
THOUGHT: <your reasoning>
ACTION: <tool_name>
ACTION_INPUT: {{"arg1": "value1"}}

When you have enough information:
THOUGHT: <final reasoning>
ANSWER: <your final answer to the user>
"""


class ReActEngine:
    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry,
        memory: MemoryManager,
        max_iterations: int = 10,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.memory = memory
        self.max_iterations = max_iterations

    async def run(self, query: str) -> ReActResult:
        steps: list[Step] = []
        total_tokens = 0

        tool_descriptions = "\n".join(
            f"- {s.name}: {s.description}" for s in self.tools.list_tools()
        )
        system_prompt = REACT_SYSTEM_PROMPT.format(tools=tool_descriptions or "No tools available.")

        context = self.memory.get_context()
        messages = [
            {"role": "system", "content": system_prompt},
            *context,
            {"role": "user", "content": query},
        ]

        logger.info("react_start", query=query[:200])

        for iteration in range(self.max_iterations):
            response = await self.llm.chat(messages)
            total_tokens += response.usage.get("total_tokens", 0)

            parsed = self._parse_response(response.content)

            for step in parsed:
                steps.append(step)
                messages.append({"role": "assistant", "content": step.content})

                if step.type == StepType.ACTION and step.tool_name:
                    result = await self.tools.call(step.tool_name, step.tool_args or {})
                    obs = Step(type=StepType.OBSERVATION, content=result)
                    steps.append(obs)
                    messages.append({"role": "user", "content": f"OBSERVATION: {result}"})
                    logger.info(
                        "react_step",
                        iteration=iteration,
                        tool=step.tool_name,
                        result_length=len(result),
                    )

                if step.type == StepType.ANSWER:
                    self.memory.add_message("user", query)
                    self.memory.add_message("assistant", step.content)
                    logger.info("react_finish", iterations=iteration + 1, total_tokens=total_tokens)
                    return ReActResult(
                        answer=step.content,
                        steps=steps,
                        total_iterations=iteration + 1,
                        total_tokens=total_tokens,
                    )

        fallback = "I was unable to reach a conclusion within the iteration limit."
        return ReActResult(
            answer=fallback,
            steps=steps,
            total_iterations=self.max_iterations,
            total_tokens=total_tokens,
        )

    def _parse_response(self, text: str) -> list[Step]:
        steps: list[Step] = []
        lines = text.strip().split("\n")
        current_type: StepType | None = None
        current_content: list[str] = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("THOUGHT:"):
                if current_type:
                    content = "\n".join(current_content).strip()
                    steps.append(Step(type=current_type, content=content))
                current_type = StepType.THOUGHT
                current_content = [stripped[len("THOUGHT:") :].strip()]
            elif stripped.startswith("ACTION:"):
                if current_type and current_type != StepType.THOUGHT:
                    content = "\n".join(current_content).strip()
                    steps.append(Step(type=current_type, content=content))
                current_type = StepType.ACTION
                tool_name = stripped[len("ACTION:") :].strip()
                current_content = [tool_name]
            elif stripped.startswith("ACTION_INPUT:"):
                import json

                try:
                    args = json.loads(stripped[len("ACTION_INPUT:") :].strip())
                except json.JSONDecodeError:
                    args = {}
                steps.append(Step(
                    type=StepType.ACTION,
                    content=str(current_content),
                    tool_name=current_content[0] if current_content else None,
                    tool_args=args,
                ))
                current_type = None
                current_content = []
            elif stripped.startswith("ANSWER:"):
                if current_type:
                    content = "\n".join(current_content).strip()
                    steps.append(Step(type=current_type, content=content))
                current_type = StepType.ANSWER
                current_content = [stripped[len("ANSWER:") :].strip()]
            else:
                current_content.append(stripped)

        if current_type:
            steps.append(Step(type=current_type, content="\n".join(current_content).strip()))

        if not steps:
            steps.append(Step(type=StepType.ANSWER, content=text.strip()))

        return steps
