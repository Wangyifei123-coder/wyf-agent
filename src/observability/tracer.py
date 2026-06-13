"""分布式追踪 — OpenTelemetry 集成"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class Span:
    name: str
    trace_id: str
    span_id: str
    parent_id: str | None = None
    start_time: float = field(default_factory=time.monotonic)
    end_time: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def duration_ms(self) -> float | None:
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000


class Tracer:
    def __init__(self, service_name: str = "wyf-agent") -> None:
        self.service_name = service_name
        self._spans: list[Span] = []
        self._trace_counter = 0
        self._span_counter = 0

    def _new_trace_id(self) -> str:
        self._trace_counter += 1
        return f"trace-{self._trace_counter}"

    def _new_span_id(self) -> str:
        self._span_counter += 1
        return f"span-{self._span_counter}"

    @contextmanager
    def start_span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
        parent: Span | None = None,
    ) -> Generator[Span, None, None]:
        span = Span(
            name=name,
            trace_id=parent.trace_id if parent else self._new_trace_id(),
            span_id=self._new_span_id(),
            parent_id=parent.span_id if parent else None,
            attributes=attributes or {},
        )

        logger.info("span_start", span=name, trace_id=span.trace_id, span_id=span.span_id)
        try:
            yield span
        except Exception as e:
            span.attributes["error"] = str(e)
            span.events.append({"name": "exception", "message": str(e)})
            raise
        finally:
            span.end_time = time.monotonic()
            self._spans.append(span)
            logger.info(
                "span_end",
                span=name,
                duration_ms=round(span.duration_ms or 0, 2),
                trace_id=span.trace_id,
            )

    def get_spans(self, trace_id: str | None = None) -> list[Span]:
        if trace_id:
            return [s for s in self._spans if s.trace_id == trace_id]
        return list(self._spans)

    def summary(self) -> dict[str, Any]:
        return {
            "total_spans": len(self._spans),
            "total_traces": len(set(s.trace_id for s in self._spans)),
            "service": self.service_name,
        }
