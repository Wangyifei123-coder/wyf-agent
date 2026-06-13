"""Token 计数器 — 追踪每次调用的 token 用量和成本"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class UsageRecord:
    model: str
    input_tokens: int
    output_tokens: int
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


COST_TABLE: dict[str, tuple[float, float]] = {
    "gpt-4o": (0.0025, 0.01),
    "gpt-4o-mini": (0.00015, 0.0006),
    "claude-sonnet-4-20250514": (0.003, 0.015),
    "claude-haiku": (0.00025, 0.00125),
}


class TokenCounter:
    def __init__(self) -> None:
        self.records: list[UsageRecord] = []

    def record(self, model: str, input_tokens: int, output_tokens: int) -> None:
        self.records.append(
            UsageRecord(model=model, input_tokens=input_tokens, output_tokens=output_tokens)
        )

    @property
    def total_input_tokens(self) -> int:
        return sum(r.input_tokens for r in self.records)

    @property
    def total_output_tokens(self) -> int:
        return sum(r.output_tokens for r in self.records)

    @property
    def total_cost(self) -> float:
        cost = 0.0
        for r in self.records:
            rates = COST_TABLE.get(r.model, (0.0, 0.0))
            cost += (r.input_tokens / 1000) * rates[0]
            cost += (r.output_tokens / 1000) * rates[1]
        return round(cost, 6)

    def summary(self) -> dict[str, object]:
        return {
            "total_calls": len(self.records),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": self.total_cost,
        }
