"""评估运行器 — 统一入口运行回归测试和 A/B 测试"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from regression_test import RegressionTester
from ab_test import ABConfig, ABTester


async def run_regression(version: str, base_url: str) -> int:
    print(f"\n{'='*60}")
    print("Running Regression Test")
    print(f"{'='*60}")

    tester = RegressionTester(base_url=base_url)
    result = await tester.run_all(version=version)

    if result.failed > 0 or result.errors > 0:
        print(f"\n❌ Regression test found {result.failed} failures and {result.errors} errors")
        return 1

    print(f"\n✅ All {result.total_tests} regression tests passed!")
    return 0


async def run_ab_test(
    test_name: str,
    base_url_a: str,
    base_url_b: str,
    variant_a_name: str,
    variant_b_name: str,
) -> int:
    print(f"\n{'='*60}")
    print("Running A/B Test")
    print(f"{'='*60}")

    variant_a = ABConfig(
        name=variant_a_name,
        description=f"Variant A from {base_url_a}",
        base_url=base_url_a,
    )

    variant_b = ABConfig(
        name=variant_b_name,
        description=f"Variant B from {base_url_b}",
        base_url=base_url_b,
    )

    tester = ABTester()
    result = await tester.run(test_name, variant_a, variant_b)

    print(f"\n🏆 Winner: {result.winner} (confidence: {result.confidence:.0%})")
    return 0


async def run_full_evaluation(version: str, base_url: str) -> int:
    print(f"\n{'='*60}")
    print("Running Full Evaluation Suite")
    print(f"{'='*60}")

    print("\n[1/2] Regression Test")
    regression_result = await run_regression(version, base_url)

    print("\n[2/2] Summary")
    print(f"  Regression: {'PASS' if regression_result == 0 else 'FAIL'}")

    return regression_result


def main():
    parser = argparse.ArgumentParser(description="WYF Agent Evaluation Runner")
    parser.add_argument(
        "command",
        choices=["regression", "ab-test", "full"],
        help="Test type to run",
    )
    parser.add_argument(
        "--version",
        default="unknown",
        help="Version identifier for regression test",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8081",
        help="Base URL for API",
    )
    parser.add_argument(
        "--test-name",
        default="ab_test",
        help="Name for A/B test",
    )
    parser.add_argument(
        "--url-a",
        help="URL for variant A (A/B test)",
    )
    parser.add_argument(
        "--url-b",
        help="URL for variant B (A/B test)",
    )
    parser.add_argument(
        "--name-a",
        default="baseline",
        help="Name for variant A",
    )
    parser.add_argument(
        "--name-b",
        default="candidate",
        help="Name for variant B",
    )

    args = parser.parse_args()

    if args.command == "regression":
        exit_code = asyncio.run(run_regression(args.version, args.base_url))
    elif args.command == "ab-test":
        url_a = args.url_a or args.base_url
        url_b = args.url_b or args.base_url
        exit_code = asyncio.run(
            run_ab_test(args.test_name, url_a, url_b, args.name_a, args.name_b)
        )
    elif args.command == "full":
        exit_code = asyncio.run(run_full_evaluation(args.version, args.base_url))
    else:
        parser.print_help()
        exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
