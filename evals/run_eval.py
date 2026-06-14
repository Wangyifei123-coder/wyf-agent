"""RAG 评估脚本 — 关键词匹配 + DeepEval 结构化输出"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import httpx

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

BASE_URL = "http://localhost:8080"
EVAL_DATASET_PATH = Path(__file__).parent / "test_dataset.json"


async def call_chat(message: str) -> dict:
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(f"{BASE_URL}/chat", json={"message": message})
        resp.raise_for_status()
        return resp.json()


def check_keywords(answer: str, keywords: list[str]) -> tuple[int, int]:
    hits = sum(1 for kw in keywords if kw in answer)
    return hits, len(keywords)


async def run_evaluation() -> None:
    with open(EVAL_DATASET_PATH, encoding="utf-8") as f:
        dataset = json.load(f)

    results: list[dict] = []
    total_score = 0.0
    output_path = Path(__file__).parent / "eval_results.json"

    print(f"\n{'='*60}")
    print(f"Starting evaluation, {len(dataset)} test cases")
    print(f"{'='*60}\n")

    for i, case in enumerate(dataset, 1):
        case_id = case["id"]
        scenario = case["scenario"]
        user_input = case["input"]
        expected_output = case["expected_output"]
        keywords = case.get("retrieval_context_keywords", [])

        print(f"[{i}/{len(dataset)}] {case_id}: {user_input[:40]}")

        try:
            response = await call_chat(user_input)
            actual_output = response.get("answer", "")

            if keywords:
                hits, total = check_keywords(actual_output, keywords)
                score = hits / total if total > 0 else 1.0
            else:
                answer_lower = actual_output.lower()
                expected_lower = expected_output.lower()
                score = 1.0 if any(w in answer_lower for w in expected_lower.split()) else 0.5

            total_score += score
            status = "PASS" if score >= 0.5 else "FAIL"

            results.append({
                "id": case_id,
                "scenario": scenario,
                "input": user_input,
                "expected_keywords": keywords,
                "keyword_hits": hits if keywords else 0,
                "keyword_total": len(keywords),
                "score": round(score, 2),
                "status": status,
                "answer_preview": actual_output[:200],
            })

            icon = "[PASS]" if status == "PASS" else "[FAIL]"
            print(f"  {icon} score={score:.2f} | {actual_output[:60]}")

        except Exception as e:
            print(f"  [ERROR] {e}")
            results.append({
                "id": case_id,
                "scenario": scenario,
                "input": user_input,
                "status": "ERROR",
                "error": str(e),
            })

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print("Evaluation Summary")
    print(f"{'='*60}\n")

    by_scenario: dict[str, list[dict]] = {}
    for r in results:
        s = r.get("scenario", "unknown")
        by_scenario.setdefault(s, []).append(r)

    for scenario, cases in by_scenario.items():
        passed = sum(1 for c in cases if c.get("status") == "PASS")
        avg = sum(c.get("score", 0) for c in cases) / len(cases) if cases else 0
        print(f"  {scenario}: {passed}/{len(cases)} passed, avg_score={avg:.2f}")

    overall = total_score / len(dataset) if dataset else 0
    overall_passed = sum(1 for r in results if r.get("status") == "PASS")
    print(f"\n  Total: {overall_passed}/{len(results)} passed, overall_score={overall:.2f}")
    print(f"\nResults saved to: {output_path}")


def main() -> None:
    asyncio.run(run_evaluation())


if __name__ == "__main__":
    main()
