"""A/B 测试框架 — 新旧 prompt / 模型对比"""

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
AB_RESULTS_PATH = Path(__file__).parent / "ab_test_results.json"


@dataclass
class ABConfig:
    name: str
    description: str
    base_url: str = BASE_URL
    headers: dict[str, str] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class TestMetrics:
    total: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    avg_score: float = 0.0
    avg_latency_ms: float = 0.0
    score_distribution: dict[str, int] = field(default_factory=dict)
    by_scenario: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class ABResult:
    test_name: str
    timestamp: str
    variant_a: ABConfig
    variant_b: ABConfig
    metrics_a: TestMetrics
    metrics_b: TestMetrics
    comparison: dict[str, Any]
    winner: str
    confidence: float
    details: list[dict[str, Any]]


class ABTester:
    def __init__(
        self,
        dataset_path: Path = EVAL_DATASET_PATH,
        pass_threshold: float = 0.5,
    ):
        self.dataset_path = dataset_path
        self.pass_threshold = pass_threshold
        self.dataset: list[dict[str, Any]] = []
        self._load_dataset()

    def _load_dataset(self) -> None:
        with open(self.dataset_path, encoding="utf-8") as f:
            self.dataset = json.load(f)

    async def _call_api(
        self,
        config: ABConfig,
        message: str,
    ) -> tuple[dict[str, Any], float]:
        start = time.monotonic()

        async with httpx.AsyncClient(timeout=120) as client:
            headers = {"Content-Type": "application/json", **config.headers}
            resp = await client.post(
                f"{config.base_url}/chat",
                json={"message": message, **config.params},
                headers=headers,
            )
            resp.raise_for_status()
            latency = (time.monotonic() - start) * 1000
            return resp.json(), latency

    def _calculate_score(self, actual: str, expected_keywords: list[str]) -> float:
        if not expected_keywords:
            return 0.5
        hits = sum(1 for kw in expected_keywords if kw in actual)
        return hits / len(expected_keywords)

    async def _run_variant(
        self,
        config: ABConfig,
        variant_name: str,
    ) -> tuple[TestMetrics, list[dict[str, Any]]]:
        print(f"\n  Running variant: {variant_name} ({config.name})")

        metrics = TestMetrics()
        details = []
        scores = []

        for i, case in enumerate(self.dataset, 1):
            case_id = case["id"]
            scenario = case.get("scenario", "unknown")
            user_input = case["input"]
            keywords = case.get("retrieval_context_keywords", [])

            print(f"    [{i}/{len(self.dataset)}] {case_id}: {user_input[:30]}...", end=" ")

            try:
                response, latency = await self._call_api(config, user_input)
                actual_output = response.get("answer", "")
                score = self._calculate_score(actual_output, keywords)

                status = "PASS" if score >= self.pass_threshold else "FAIL"
                scores.append(score)

                metrics.total += 1
                if status == "PASS":
                    metrics.passed += 1
                else:
                    metrics.failed += 1

                if scenario not in metrics.by_scenario:
                    metrics.by_scenario[scenario] = {"total": 0, "passed": 0, "scores": []}
                metrics.by_scenario[scenario]["total"] += 1
                if status == "PASS":
                    metrics.by_scenario[scenario]["passed"] += 1
                metrics.by_scenario[scenario]["scores"].append(score)

                details.append({
                    "id": case_id,
                    "scenario": scenario,
                    "variant": variant_name,
                    "status": status,
                    "score": round(score, 3),
                    "latency_ms": round(latency, 2),
                    "answer_preview": actual_output[:200],
                })

                icon = "✓" if status == "PASS" else "✗"
                print(f"{icon} {score:.2f} {latency:.0f}ms")

            except Exception as e:
                metrics.total += 1
                metrics.errors += 1
                details.append({
                    "id": case_id,
                    "scenario": scenario,
                    "variant": variant_name,
                    "status": "ERROR",
                    "error": str(e),
                })
                print(f"! ERROR: {e}")

        metrics.avg_score = sum(scores) / len(scores) if scores else 0
        metrics.avg_latency_ms = sum(d.get("latency_ms", 0) for d in details) / len(details) if details else 0

        for scenario, data in metrics.by_scenario.items():
            scenario_scores = data.get("scores", [])
            data["avg_score"] = sum(scenario_scores) / len(scenario_scores) if scenario_scores else 0
            del data["scores"]

        return metrics, details

    async def run(
        self,
        test_name: str,
        variant_a: ABConfig,
        variant_b: ABConfig,
    ) -> ABResult:
        print(f"\n{'='*60}")
        print(f"A/B Test: {test_name}")
        print(f"{'='*60}")
        print(f"  Variant A: {variant_a.name} - {variant_a.description}")
        print(f"  Variant B: {variant_b.name} - {variant_b.description}")
        print(f"  Dataset: {len(self.dataset)} test cases")

        print(f"\n--- Variant A ---")
        metrics_a, details_a = await self._run_variant(variant_a, "A")

        print(f"\n--- Variant B ---")
        metrics_b, details_b = await self._run_variant(variant_b, "B")

        comparison = self._compare(metrics_a, metrics_b)
        winner, confidence = self._determine_winner(metrics_a, metrics_b)

        result = ABResult(
            test_name=test_name,
            timestamp=datetime.now().isoformat(),
            variant_a=variant_a,
            variant_b=variant_b,
            metrics_a=metrics_a,
            metrics_b=metrics_b,
            comparison=comparison,
            winner=winner,
            confidence=confidence,
            details=details_a + details_b,
        )

        self._save_result(result)
        self._print_summary(result)

        return result

    def _compare(self, a: TestMetrics, b: TestMetrics) -> dict[str, Any]:
        score_diff = b.avg_score - a.avg_score
        latency_diff = b.avg_latency_ms - a.avg_latency_ms
        pass_rate_a = a.passed / a.total * 100 if a.total else 0
        pass_rate_b = b.passed / b.total * 100 if b.total else 0

        scenario_comparison = {}
        all_scenarios = set(list(a.by_scenario.keys()) + list(b.by_scenario.keys()))
        for scenario in all_scenarios:
            a_data = a.by_scenario.get(scenario, {})
            b_data = b.by_scenario.get(scenario, {})
            a_score = a_data.get("avg_score", 0)
            b_score = b_data.get("avg_score", 0)
            scenario_comparison[scenario] = {
                "a_score": round(a_score, 3),
                "b_score": round(b_score, 3),
                "diff": round(b_score - a_score, 3),
                "winner": "B" if b_score > a_score else "A" if a_score > b_score else "Tie",
            }

        return {
            "score_diff": round(score_diff, 3),
            "score_improvement": f"{score_diff*100:+.1f}%",
            "latency_diff_ms": round(latency_diff, 2),
            "pass_rate_a": round(pass_rate_a, 1),
            "pass_rate_b": round(pass_rate_b, 1),
            "scenario_comparison": scenario_comparison,
        }

    def _determine_winner(self, a: TestMetrics, b: TestMetrics) -> tuple[str, float]:
        if a.total == 0 or b.total == 0:
            return "N/A", 0.0

        score_diff = abs(b.avg_score - a.avg_score)
        pass_rate_a = a.passed / a.total
        pass_rate_b = b.passed / b.total
        pass_diff = abs(pass_rate_b - pass_rate_a)

        confidence = min(1.0, (score_diff * 5 + pass_diff * 3))

        if b.avg_score > a.avg_score and pass_rate_b >= pass_rate_a:
            return "B", round(confidence, 2)
        elif a.avg_score > b.avg_score and pass_rate_a >= pass_rate_b:
            return "A", round(confidence, 2)
        else:
            return "Tie", round(confidence, 2)

    def _save_result(self, result: ABResult) -> None:
        history = []
        if AB_RESULTS_PATH.exists():
            with open(AB_RESULTS_PATH, encoding="utf-8") as f:
                history = json.load(f)

        history.append({
            "test_name": result.test_name,
            "timestamp": result.timestamp,
            "variant_a": {"name": result.variant_a.name, "description": result.variant_a.description},
            "variant_b": {"name": result.variant_b.name, "description": result.variant_b.description},
            "metrics_a": {
                "total": result.metrics_a.total,
                "passed": result.metrics_a.passed,
                "avg_score": result.metrics_a.avg_score,
                "avg_latency_ms": result.metrics_a.avg_latency_ms,
            },
            "metrics_b": {
                "total": result.metrics_b.total,
                "passed": result.metrics_b.passed,
                "avg_score": result.metrics_b.avg_score,
                "avg_latency_ms": result.metrics_b.avg_latency_ms,
            },
            "comparison": result.comparison,
            "winner": result.winner,
            "confidence": result.confidence,
        })

        with open(AB_RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    def _print_summary(self, result: ABResult) -> None:
        print(f"\n{'='*60}")
        print(f"A/B Test Results: {result.test_name}")
        print(f"{'='*60}")

        print(f"\n  Variant A: {result.variant_a.name}")
        print(f"    Passed: {result.metrics_a.passed}/{result.metrics_a.total}")
        print(f"    Avg Score: {result.metrics_a.avg_score:.3f}")
        print(f"    Avg Latency: {result.metrics_a.avg_latency_ms:.0f}ms")

        print(f"\n  Variant B: {result.variant_b.name}")
        print(f"    Passed: {result.metrics_b.passed}/{result.metrics_b.total}")
        print(f"    Avg Score: {result.metrics_b.avg_score:.3f}")
        print(f"    Avg Latency: {result.metrics_b.avg_latency_ms:.0f}ms")

        print(f"\n  Comparison:")
        print(f"    Score: {result.comparison['score_improvement']}")
        print(f"    Pass Rate: A={result.comparison['pass_rate_a']}% vs B={result.comparison['pass_rate_b']}%")

        print(f"\n  Winner: {result.winner} (confidence: {result.confidence:.0%})")


async def main():
    import sys

    test_name = sys.argv[1] if len(sys.argv) > 1 else "default_test"

    variant_a = ABConfig(
        name="baseline",
        description="Current production version",
        base_url=BASE_URL,
    )

    variant_b = ABConfig(
        name="candidate",
        description="New version to test",
        base_url=BASE_URL,
        params={"mode": "react"},
    )

    tester = ABTester()
    result = await tester.run(test_name, variant_a, variant_b)

    print(f"\nResults saved to: {AB_RESULTS_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
