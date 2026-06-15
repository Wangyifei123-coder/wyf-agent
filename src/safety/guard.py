"""安全守卫 — 输入/输出过滤、Prompt Injection 检测"""

from __future__ import annotations

import re
from dataclasses import dataclass

import structlog

logger = structlog.get_logger(__name__)

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"you\s+are\s+now\s+(a|an)\s+",
    r"system\s*:\s*you\s+are",
    r"<\|im_start\|>",
    r"\[INST\]",
    r"forget\s+(all\s+)?(your|previous)\s+(instructions|rules)",
    r"override\s+(safety|security|rules)",
    r"act\s+as\s+(if|though)\s+you\s+(have|are|were)",
    r"pretend\s+you\s+(are|have|were)",
    r"disregard\s+(all\s+)?(your|the)\s+(rules|instructions|guidelines)",
    r"bypass\s+(all\s+)?(safety|security|content|filter)",
    r"reveal\s+(your|the)\s+(system|initial|secret)\s+prompt",
    r"what\s+(are|is)\s+your\s+(system|initial)\s+(prompt|instruction)",
    r"repeat\s+(the|your)\s+(system|initial)\s+(prompt|message)",
    r"输出(你的|系统|初始)(提示词|指令|prompt)",
    r"忽略(之前|以前|所有)(的)?(指令|规则|限制)",
    r"你现在是",
    r"假装(你|你是|自己是)",
    r"绕过(安全|过滤|限制)",
    r"告诉我(你的|系统)(提示词|指令)",
    r"base64\s*(decode|编码|解码)",
    r"rot13",
    r"\\x[0-9a-fA-F]{2}",
    r"\\u[0-9a-fA-F]{4}",
    r"<script[^>]*>",
    r"javascript:",
    r"data:text/html",
    r"<!--.*?-->",
    r"\{\{.*?\}\}",
    r"\{\%.*?\%\}",
    r"\$\{.*?\}",
    r"__import__",
    r"eval\s*\(",
    r"exec\s*\(",
    r"os\.system",
    r"subprocess",
    r"rm\s+-rf",
    r"DROP\s+TABLE",
    r"DELETE\s+FROM",
    r"UNION\s+SELECT",
]

INJECTION_PATTERNS_COMPILED = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]

PII_PATTERNS = {
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "phone": r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "ssn": r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b",
    "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    "id_card": r"\b\d{17}[\dXx]\b",
    "ip_address": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
}

PII_PATTERNS_COMPILED = {k: re.compile(v) for k, v in PII_PATTERNS.items()}


@dataclass
class SafetyCheckResult:
    safe: bool
    reason: str = ""
    detected_issues: list[str] | None = None
    risk_level: str = "low"


class SafetyGuard:
    def __init__(
        self,
        enable_input_filter: bool = True,
        enable_output_filter: bool = True,
        max_input_length: int = 10000,
    ) -> None:
        self.enable_input_filter = enable_input_filter
        self.enable_output_filter = enable_output_filter
        self.max_input_length = max_input_length

    def check_input(self, text: str) -> SafetyCheckResult:
        if not self.enable_input_filter:
            return SafetyCheckResult(safe=True)

        if len(text) > self.max_input_length:
            return SafetyCheckResult(
                safe=False,
                reason="Input exceeds maximum length",
                detected_issues=["length_exceeded"],
                risk_level="medium",
            )

        issues: list[str] = []
        for i, pattern in enumerate(INJECTION_PATTERNS_COMPILED):
            if pattern.search(text):
                issues.append(f"potential_injection: {INJECTION_PATTERNS[i][:50]}")

        if issues:
            risk_level = "high" if len(issues) >= 3 else "medium"
            logger.warning("input_safety_issue", issues=issues, text_preview=text[:100])
            return SafetyCheckResult(
                safe=False,
                reason="Potential prompt injection detected",
                detected_issues=issues,
                risk_level=risk_level,
            )

        return SafetyCheckResult(safe=True)

    def check_output(self, text: str) -> SafetyCheckResult:
        if not self.enable_output_filter:
            return SafetyCheckResult(safe=True)

        issues: list[str] = []
        for pii_type, pattern in PII_PATTERNS_COMPILED.items():
            matches = pattern.findall(text)
            if matches:
                issues.append(f"pii_detected: {pii_type} ({len(matches)} instances)")

        if issues:
            logger.warning("output_safety_issue", issues=issues)
            return SafetyCheckResult(
                safe=False,
                reason="PII detected in output",
                detected_issues=issues,
                risk_level="medium",
            )

        return SafetyCheckResult(safe=True)

    def redact_pii(self, text: str) -> str:
        redacted = text
        for pii_type, pattern in PII_PATTERNS_COMPILED.items():
            redacted = pattern.sub(f"[REDACTED_{pii_type.upper()}]", redacted)
        return redacted
