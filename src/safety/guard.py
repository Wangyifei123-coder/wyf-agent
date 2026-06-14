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
]

PII_PATTERNS = {
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "phone": r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "ssn": r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b",
    "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
}


@dataclass
class SafetyCheckResult:
    safe: bool
    reason: str = ""
    detected_issues: list[str] | None = None


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
            )

        issues: list[str] = []
        text_lower = text.lower()
        for pattern in INJECTION_PATTERNS:
            if re.search(pattern, text_lower):
                issues.append(f"potential_injection: {pattern}")

        if issues:
            logger.warning("input_safety_issue", issues=issues, text_preview=text[:100])
            return SafetyCheckResult(
                safe=False,
                reason="Potential prompt injection detected",
                detected_issues=issues,
            )

        return SafetyCheckResult(safe=True)

    def check_output(self, text: str) -> SafetyCheckResult:
        if not self.enable_output_filter:
            return SafetyCheckResult(safe=True)

        issues: list[str] = []
        for pii_type, pattern in PII_PATTERNS.items():
            matches = re.findall(pattern, text)
            if matches:
                issues.append(f"pii_detected: {pii_type} ({len(matches)} instances)")

        if issues:
            logger.warning("output_safety_issue", issues=issues)
            return SafetyCheckResult(
                safe=False,
                reason="PII detected in output",
                detected_issues=issues,
            )

        return SafetyCheckResult(safe=True)

    def redact_pii(self, text: str) -> str:
        redacted = text
        for pii_type, pattern in PII_PATTERNS.items():
            redacted = re.sub(pattern, f"[REDACTED_{pii_type.upper()}]", redacted)
        return redacted
