"""推理引擎 — 支持 ReAct、Plan-and-Execute、Reflexion 三种模式"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import structlog

from ..gateway.client import LLMClient
from ..memory.manager import MemoryManager
from ..tools.registry import ToolRegistry

logger = structlog.get_logger(__name__)


class ReasoningMode(StrEnum):
    REACT = "react"
    PLAN_AND_EXECUTE = "plan_and_execute"
    REFLEXION = "reflexion"


class StepType(StrEnum):
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    ANSWER = "answer"
    PLAN = "plan"
    PLAN_STEP = "plan_step"
    REFLECTION = "reflection"
    REVISED_PLAN = "revised_plan"


@dataclass
class Step:
    type: StepType
    content: str
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    step_index: int | None = None
    success: bool = True
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningResult:
    answer: str
    steps: list[Step]
    mode: ReasoningMode
    total_iterations: int
    total_tokens: int
    success: bool
    reflections: list[str] = field(default_factory=list)


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

PLAN_AND_EXECUTE_SYSTEM_PROMPT = """You are a planning agent that breaks complex tasks into steps.

Given a user question, first create a plan, then execute each step.

Available tools:
{tools}

Phase 1 - PLANNING:
Create a numbered list of steps to solve the problem.

PLAN:
1. <step 1 description>
2. <step 2 description>
...

Phase 2 - EXECUTION:
For each step, use tools as needed:

EXECUTE_STEP: <step number>
THOUGHT: <reasoning about this step>
ACTION: <tool_name>
ACTION_INPUT: {{"arg1": "value1"}}

After all steps:
THOUGHT: <final synthesis>
ANSWER: <your final answer to the user>
"""

REFLEXION_SYSTEM_PROMPT = """You are a self-reflecting agent that learns from failures.

When a task fails, analyze what went wrong and try again with a better approach.

Available tools:
{tools}

Process:
1. Attempt the task using ReAct reasoning
2. If you fail or get unexpected results, REFLECT on what went wrong
3. REVISE your approach and try again
4. Repeat until success or max attempts reached

Format:
THOUGHT: <your reasoning>
ACTION: <tool_name>
ACTION_INPUT: {{"arg1": "value1"}}

If you need to reflect:
REFLECTION: <what went wrong and how to improve>
REVISED_APPROACH: <new strategy>

When complete:
THOUGHT: <final reasoning>
ANSWER: <your final answer to the user>
"""


class ReasoningEngine:
    """统一推理引擎，支持三种模式"""

    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry,
        memory: MemoryManager,
        max_iterations: int = 15,
        max_reflections: int = 3,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.memory = memory
        self.max_iterations = max_iterations
        self.max_reflections = max_reflections

    async def run(
        self,
        query: str,
        mode: ReasoningMode | None = None,
    ) -> ReasoningResult:
        if mode is None:
            mode = self._auto_select_mode(query)

        logger.info("reasoning_start", query=query[:200], mode=mode)

        if mode == ReasoningMode.REACT:
            return await self._run_react(query)
        elif mode == ReasoningMode.PLAN_AND_EXECUTE:
            return await self._run_plan_and_execute(query)
        elif mode == ReasoningMode.REFLEXION:
            return await self._run_reflexion(query)
        else:
            raise ValueError(f"Unknown reasoning mode: {mode}")

    def _auto_select_mode(self, query: str) -> ReasoningMode:
        query_lower = query.lower()

        complex_keywords = ["计划", "步骤", "流程", "如何", "怎么", "实现", "设计", "搭建"]
        if any(kw in query_lower for kw in complex_keywords) and len(query) > 50:
            return ReasoningMode.PLAN_AND_EXECUTE

        retry_keywords = ["重试", "再试", "修复", "解决", "debug", "fix"]
        if any(kw in query_lower for kw in retry_keywords):
            return ReasoningMode.REFLEXION

        return ReasoningMode.REACT

    async def _run_react(self, query: str) -> ReasoningResult:
        steps: list[Step] = []
        total_tokens = 0

        tool_descriptions = self._get_tool_descriptions()
        system_prompt = REACT_SYSTEM_PROMPT.format(tools=tool_descriptions)
        messages = self._build_messages(system_prompt, query)

        for iteration in range(self.max_iterations):
            response = await self.llm.chat(messages)
            total_tokens += response.usage.get("total_tokens", 0)

            parsed = self._parse_react_response(response.content)

            for step in parsed:
                steps.append(step)
                messages.append({"role": "assistant", "content": step.content})

                if step.type == StepType.ACTION and step.tool_name:
                    result = await self._execute_tool(step.tool_name, step.tool_args)
                    obs = Step(type=StepType.OBSERVATION, content=result["output"])
                    steps.append(obs)
                    messages.append({"role": "user", "content": f"OBSERVATION: {result['output']}"})

                if step.type == StepType.ANSWER:
                    self._save_to_memory(query, step.content)
                    return ReasoningResult(
                        answer=step.content,
                        steps=steps,
                        mode=ReasoningMode.REACT,
                        total_iterations=iteration + 1,
                        total_tokens=total_tokens,
                        success=True,
                    )

        return ReasoningResult(
            answer="未能在迭代限制内得出结论",
            steps=steps,
            mode=ReasoningMode.REACT,
            total_iterations=self.max_iterations,
            total_tokens=total_tokens,
            success=False,
        )

    async def _run_plan_and_execute(self, query: str) -> ReasoningResult:
        steps: list[Step] = []
        total_tokens = 0
        plan_steps: list[str] = []

        tool_descriptions = self._get_tool_descriptions()
        system_prompt = PLAN_AND_EXECUTE_SYSTEM_PROMPT.format(tools=tool_descriptions)
        messages = self._build_messages(system_prompt, query)

        planning_prompt = f"""请为以下任务制定执行计划：

任务：{query}

请使用以下格式：
PLAN:
1. <步骤1描述>
2. <步骤2描述>
..."""

        messages.append({"role": "user", "content": planning_prompt})
        response = await self.llm.chat(messages)
        total_tokens += response.usage.get("total_tokens", 0)

        plan_steps = self._parse_plan(response.content)
        steps.append(Step(type=StepType.PLAN, content=response.content))

        if not plan_steps:
            return await self._run_react(query)

        context = ""
        for i, plan_step in enumerate(plan_steps):
            steps.append(Step(
                type=StepType.PLAN_STEP,
                content=plan_step,
                step_index=i + 1,
            ))

            exec_prompt = f"""执行计划步骤 {i + 1}/{len(plan_steps)}:
{plan_step}

{f'之前的执行结果：{context}' if context else ''}

请使用工具完成这一步，或直接给出结果。"""

            messages.append({"role": "user", "content": exec_prompt})

            for _ in range(3):
                response = await self.llm.chat(messages)
                total_tokens += response.usage.get("total_tokens", 0)

                parsed = self._parse_react_response(response.content)
                step_success = True

                for step in parsed:
                    steps.append(step)
                    messages.append({"role": "assistant", "content": step.content})

                    if step.type == StepType.ACTION and step.tool_name:
                        result = await self._execute_tool(step.tool_name, step.tool_args)
                        obs = Step(
                            type=StepType.OBSERVATION,
                            content=result["output"],
                            success=result["success"],
                        )
                        steps.append(obs)
                        messages.append({"role": "user", "content": f"OBSERVATION: {result['output']}"})
                        if not result["success"]:
                            step_success = False

                    if step.type == StepType.ANSWER:
                        context += f"\n步骤{i + 1}结果: {step.content}"
                        break

                if step_success:
                    break

        synthesis_prompt = f"""基于以上执行结果，请给出最终答案：

任务：{query}

执行结果：
{context}

ANSWER: """

        messages.append({"role": "user", "content": synthesis_prompt})
        response = await self.llm.chat(messages)
        total_tokens += response.usage.get("total_tokens", 0)

        final_answer = self._extract_answer(response.content)
        steps.append(Step(type=StepType.ANSWER, content=final_answer))

        self._save_to_memory(query, final_answer)

        return ReasoningResult(
            answer=final_answer,
            steps=steps,
            mode=ReasoningMode.PLAN_AND_EXECUTE,
            total_iterations=len(plan_steps) + 1,
            total_tokens=total_tokens,
            success=True,
        )

    async def _run_reflexion(self, query: str) -> ReasoningResult:
        steps: list[Step] = []
        total_tokens = 0
        reflections: list[str] = []
        previous_attempts: list[str] = []

        tool_descriptions = self._get_tool_descriptions()
        system_prompt = REFLEXION_SYSTEM_PROMPT.format(tools=tool_descriptions)

        for attempt in range(self.max_reflections + 1):
            messages = self._build_messages(system_prompt, query)

            if previous_attempts:
                attempts_text = "\n".join([
                    f"尝试 {i + 1}: {att}" for i, att in enumerate(previous_attempts)
                ])
                retry_prompt = f"""之前的尝试失败了：

{attempts_text}

{f'反思：{reflections[-1]}' if reflections else ''}

请用不同的方法重试。"""
                messages.append({"role": "user", "content": retry_prompt})

            for iteration in range(self.max_iterations):
                response = await self.llm.chat(messages)
                total_tokens += response.usage.get("total_tokens", 0)

                if "REFLECTION:" in response.content:
                    reflection = self._extract_section(response.content, "REFLECTION")
                    reflections.append(reflection)
                    steps.append(Step(type=StepType.REFLECTION, content=reflection))
                    logger.info("reflection", attempt=attempt, content=reflection[:200])
                    break

                parsed = self._parse_react_response(response.content)

                for step in parsed:
                    steps.append(step)
                    messages.append({"role": "assistant", "content": step.content})

                    if step.type == StepType.ACTION and step.tool_name:
                        result = await self._execute_tool(step.tool_name, step.tool_args)
                        obs = Step(
                            type=StepType.OBSERVATION,
                            content=result["output"],
                            success=result["success"],
                        )
                        steps.append(obs)
                        messages.append({"role": "user", "content": f"OBSERVATION: {result['output']}"})

                        if not result["success"]:
                            previous_attempts.append(
                                f"工具 {step.tool_name} 失败: {result['output']}"
                            )
                            break

                    if step.type == StepType.ANSWER:
                        self._save_to_memory(query, step.content)
                        return ReasoningResult(
                            answer=step.content,
                            steps=steps,
                            mode=ReasoningMode.REFLEXION,
                            total_iterations=attempt * self.max_iterations + iteration + 1,
                            total_tokens=total_tokens,
                            success=True,
                            reflections=reflections,
                        )

            previous_attempts.append("未能得出答案")

        return ReasoningResult(
            answer=f"经过 {self.max_reflections} 次反思重试后仍未能解决问题",
            steps=steps,
            mode=ReasoningMode.REFLEXION,
            total_iterations=self.max_iterations * (self.max_reflections + 1),
            total_tokens=total_tokens,
            success=False,
            reflections=reflections,
        )

    def _get_tool_descriptions(self) -> str:
        tools = self.tools.list_tools()
        if not tools:
            return "No tools available."
        return "\n".join(f"- {t.name}: {t.description}" for t in tools)

    def _build_messages(self, system_prompt: str, query: str) -> list[dict[str, str]]:
        context = self.memory.get_context()
        return [
            {"role": "system", "content": system_prompt},
            *context,
            {"role": "user", "content": query},
        ]

    async def _execute_tool(
        self, tool_name: str, tool_args: dict[str, Any] | None
    ) -> dict[str, Any]:
        try:
            result = await self.tools.call(tool_name, tool_args or {})
            return {"output": result, "success": True}
        except Exception as e:
            logger.error("tool_execution_failed", tool=tool_name, error=str(e))
            return {"output": f"Error: {str(e)}", "success": False}

    def _save_to_memory(self, query: str, answer: str) -> None:
        self.memory.add_message("user", query)
        self.memory.add_message("assistant", answer)

    def _parse_plan(self, text: str) -> list[str]:
        steps = []
        in_plan = False

        for line in text.split("\n"):
            stripped = line.strip()
            if stripped.upper().startswith("PLAN:"):
                in_plan = True
                continue
            if in_plan:
                if stripped and stripped[0].isdigit():
                    step = stripped.split(".", 1)[-1].strip()
                    if step:
                        steps.append(step)
                elif stripped.startswith("-"):
                    step = stripped[1:].strip()
                    if step:
                        steps.append(step)

        return steps

    def _parse_react_response(self, text: str) -> list[Step]:
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
                current_content = [stripped[len("THOUGHT:"):].strip()]
            elif stripped.startswith("ACTION:"):
                if current_type and current_type != StepType.THOUGHT:
                    content = "\n".join(current_content).strip()
                    steps.append(Step(type=current_type, content=content))
                current_type = StepType.ACTION
                tool_name = stripped[len("ACTION:"):].strip()
                current_content = [tool_name]
            elif stripped.startswith("ACTION_INPUT:"):
                import json
                try:
                    args = json.loads(stripped[len("ACTION_INPUT:"):].strip())
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
                current_content = [stripped[len("ANSWER:"):].strip()]
            else:
                current_content.append(stripped)

        if current_type:
            steps.append(Step(type=current_type, content="\n".join(current_content).strip()))

        if not steps:
            steps.append(Step(type=StepType.ANSWER, content=text.strip()))

        return steps

    def _extract_answer(self, text: str) -> str:
        if "ANSWER:" in text:
            return text.split("ANSWER:")[-1].strip()
        return text.strip()

    def _extract_section(self, text: str, section: str) -> str:
        if f"{section}:" in text:
            parts = text.split(f"{section}:")
            if len(parts) > 1:
                end_markers = ["THOUGHT:", "ACTION:", "ANSWER:", "REVISED_APPROACH:"]
                content = parts[1]
                for marker in end_markers:
                    if marker in content:
                        content = content.split(marker)[0]
                        break
                return content.strip()
        return ""
