"""Prometheus 指标模块"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

REQUEST_COUNT = Counter(
    "wyf_agent_requests_total",
    "Total number of requests",
    ["endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "wyf_agent_request_duration_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0],
)

CHAT_COUNT = Counter(
    "wyf_agent_chat_total",
    "Total number of chat requests",
    ["intent"],
)

KNOWLEDGE_DOCS = Gauge(
    "wyf_agent_knowledge_docs",
    "Number of documents in knowledge base",
)

ACTIVE_SESSIONS = Gauge(
    "wyf_agent_active_sessions",
    "Number of active sessions",
)
