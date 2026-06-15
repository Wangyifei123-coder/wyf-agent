"""回归测试框架 — 每次 prompt 修改后自动运行评估集"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

BASE_URL = "http://localhost:8081"
EVAL_DATASET_PATH = Path(__file__).parent / "test_dataset.json"
REGRESSION_HISTORY_PATH = Path(__file__).parent / "regression_history.json"


@dataclass
class TestCase:
    id: str
    scenario: str
    input: str
    expected_output: str
    expected_keywords: list[str] = field(default_factory=list)
    expected_intent: str = ""


@dataclass
class TestResult:
    test_id: str
    scenario: str
    status: str  # PASS, FAIL, ERROR
    score: float
    keyword_hits: int = 0
    keyword_total: int = 0
    actual_output: str = ""
    error: str = ""
    latency_ms: float = 0.0


@dataclass
class RegressionResult:
    timestamp: str
    version: str
    total_tests: int
    passed: int
    failed: int
    errors: int
    avg_score: float
    avg_latency_ms: float
    by_scenario: dict[str, dict[str, Any]]
    results: list[TestResult]
    baseline_comparison: dict[str, Any] | None = None


class RegressionTester:
    def __init__(
        self,
        base_url: str = BASE_URL,
        dataset_path: Path = EVAL_DATASET_PATH,
        pass_threshold: float = 0.5,
        username: str = "admin",
        password: str = "admin123",
    ):
        self.base_url = base_url
        self.dataset_path = dataset_path
        self.pass_threshold = pass_threshold
        self.username = username
        self.password = password
        self.token: str = ""
        self.dataset: list[TestCase] = []
        self._load_dataset()

    def _load_dataset(self) -> None:
        with open(self.dataset_path, encoding="utf-8") as f:
            data = json.load(f)

        self.dataset = []
        for item in data:
            self.dataset.append(TestCase(
                id=item["id"],
                scenario=item.get("scenario", "unknown"),
                input=item["input"],
                expected_output=item.get("expected_output", ""),
                expected_keywords=item.get("retrieval_context_keywords", []),
                expected_intent=item.get("expected_intent", ""),
            ))

    async def _get_token(self) -> str:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{self.base_url}/auth/login",
                json={"username": self.username, "password": self.password},
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("token", "")

    async def _call_api(self, message: str) -> dict[str, Any]:
        if not self.token:
            self.token = await self._get_token()

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{self.base_url}/chat",
                json={"message": message},
                headers={"Authorization": f"Bearer {self.token}"},
            )
            resp.raise_for_status()
            return resp.json()

    def _calculate_score(self, actual: str, expected: str, keywords: list[str]) -> float:
        if keywords:
            hits = sum(1 for kw in keywords if kw in actual)
            return hits / len(keywords) if keywords else 1.0

        actual_lower = actual.lower()
        expected_words = expected.lower().split()
        matches = sum(1 for w in expected_words if w in actual_lower)
        return matches / len(expected_words) if expected_words else 0.0

    async def run_single(self, test_case: TestCase) -> TestResult:
        start_time = time.monotonic()

        try:
            response = await self._call_api(test_case.input)
            latency = (time.monotonic() - start_time) * 1000

            actual_output = response.get("answer", "")
            score = self._calculate_score(
                actual_output,
                test_case.expected_output,
                test_case.expected_keywords,
            )

            keyword_hits = 0
            if test_case.expected_keywords:
                keyword_hits = sum(1 for kw in test_case.expected_keywords if kw in actual_output)

            status = "PASS" if score >= self.pass_threshold else "FAIL"

            return TestResult(
                test_id=test_case.id,
                scenario=test_case.scenario,
                status=status,
                score=round(score, 3),
                keyword_hits=keyword_hits,
                keyword_total=len(test_case.expected_keywords),
                actual_output=actual_output[:500],
                latency_ms=round(latency, 2),
            )

        except Exception as e:
            latency = (time.monotonic() - start_time) * 1000
            return TestResult(
                test_id=test_case.id,
                scenario=test_case.scenario,
                status="ERROR",
                score=0.0,
                error=str(e),
                latency_ms=round(latency, 2),
            )

    async def run_all(self, version: str = "unknown") -> RegressionResult:
        print(f"\n{'='*60}")
        print(f"Regression Test - Version: {version}")
        print(f"Dataset: {len(self.dataset)} test cases")
        print(f"{'='*60}\n")

        results: list[TestResult] = []
        for i, test_case in enumerate(self.dataset, 1):
            print(f"[{i}/{len(self.dataset)}] {test_case.id}: {test_case.input[:40]}...", end=" ")
            result = await self.run_single(test_case)
            results.append(result)

            icon = "[PASS]" if result.status == "PASS" else "[FAIL]" if result.status == "FAIL" else "[ERR]"
            print(f"{icon} score={result.score:.2f} latency={result.latency_ms:.0f}ms")

        by_scenario: dict[str, list[TestResult]] = {}
        for r in results:
            by_scenario.setdefault(r.scenario, []).append(r)

        scenario_stats = {}
        for scenario, cases in by_scenario.items():
            passed = sum(1 for c in cases if c.status == "PASS")
            avg_score = sum(c.score for c in cases) / len(cases)
            avg_latency = sum(c.latency_ms for c in cases) / len(cases)
            scenario_stats[scenario] = {
                "total": len(cases),
                "passed": passed,
                "failed": len(cases) - passed,
                "avg_score": round(avg_score, 3),
                "avg_latency_ms": round(avg_latency, 2),
            }

        total_passed = sum(1 for r in results if r.status == "PASS")
        total_failed = sum(1 for r in results if r.status == "FAIL")
        total_errors = sum(1 for r in results if r.status == "ERROR")
        avg_score = sum(r.score for r in results) / len(results) if results else 0
        avg_latency = sum(r.latency_ms for r in results) / len(results) if results else 0

        regression_result = RegressionResult(
            timestamp=datetime.now().isoformat(),
            version=version,
            total_tests=len(results),
            passed=total_passed,
            failed=total_failed,
            errors=total_errors,
            avg_score=round(avg_score, 3),
            avg_latency_ms=round(avg_latency, 2),
            by_scenario=scenario_stats,
            results=results,
        )

        self._save_result(regression_result)
        self._print_summary(regression_result)

        return regression_result

    def _save_result(self, result: RegressionResult) -> None:
        history = []
        if REGRESSION_HISTORY_PATH.exists():
            with open(REGRESSION_HISTORY_PATH, encoding="utf-8") as f:
                history = json.load(f)

        history.append({
            "timestamp": result.timestamp,
            "version": result.version,
            "total_tests": result.total_tests,
            "passed": result.passed,
            "failed": result.failed,
            "errors": result.errors,
            "avg_score": result.avg_score,
            "avg_latency_ms": result.avg_latency_ms,
            "by_scenario": result.by_scenario,
        })

        with open(REGRESSION_HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    def _print_summary(self, result: RegressionResult) -> None:
        print(f"\n{'='*60}")
        print("Regression Test Summary")
        print(f"{'='*60}\n")

        print(f"  Version: {result.version}")
        print(f"  Timestamp: {result.timestamp}")
        print(f"  Total Tests: {result.total_tests}")
        print(f"  Passed: {result.passed} ({result.passed/result.total_tests*100:.1f}%)")
        print(f"  Failed: {result.failed}")
        print(f"  Errors: {result.errors}")
        print(f"  Average Score: {result.avg_score:.3f}")
        print(f"  Average Latency: {result.avg_latency_ms:.0f}ms")

        print(f"\n  By Scenario:")
        for scenario, stats in result.by_scenario.items():
            print(f"    {scenario}: {stats['passed']}/{stats['total']} passed, "
                  f"avg_score={stats['avg_score']:.3f}")

        if result.baseline_comparison:
            print(f"\n  Baseline Comparison:")
            print(f"    Score Change: {result.baseline_comparison.get('score_change', 0):+.3f}")
            print(f"    Pass Rate Change: {result.baseline_comparison.get('pass_rate_change', 0):+.1f}%")

    def compare_with_baseline(self, current: RegressionResult, baseline_version: str) -> dict[str, Any]:
        if not REGRESSION_HISTORY_PATH.exists():
            return {}

        with open(REGRESSION_HISTORY_PATH, encoding="utf-8") as f:
            history = json.load(f)

        baseline = None
        for h in history:
            if h.get("version") == baseline_version:
                baseline = h
                break

        if not baseline:
            return {}

        score_change = current.avg_score - baseline.get("avg_score", 0)
        current_pass_rate = current.passed / current.total_tests * 100 if current.total_tests else 0
        baseline_pass_rate = baseline.get("passed", 0) / baseline.get("total_tests", 1) * 100

        return {
            "baseline_version": baseline_version,
            "score_change": round(score_change, 3),
            "pass_rate_change": round(current_pass_rate - baseline_pass_rate, 1),
            "latency_change": round(current.avg_latency_ms - baseline.get("avg_latency_ms", 0), 2),
        }


async def main():
    import sys

    version = sys.argv[1] if len(sys.argv) > 1 else "test"

    tester = RegressionTester()
    result = await tester.run_all(version=version)

    if result.failed > 0 or result.errors > 0:
        print(f"\n⚠️  Regression test found issues!")
        sys.exit(1)
    else:
        print(f"\n✓ All regression tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
