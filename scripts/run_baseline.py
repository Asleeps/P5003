#!/usr/bin/env python3
"""
Convenience wrapper to run baseline benchmarks.

Usage:
    python scripts/run_baseline.py [--config config/benchmark.json]
"""

import argparse
from pathlib import Path

from src.benchmarks.baseline import BaselineBenchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline performance benchmarks")
    parser.add_argument(
        "--config",
        default=str(Path("config") / "benchmark.json"),
        help="Path to benchmark configuration JSON",
    )
    args = parser.parse_args()

    benchmark = BaselineBenchmark(config_path=args.config)
    results = benchmark.run_all()

    print("\n" + "=" * 80)
    print("BASELINE SUMMARY")
    print("=" * 80)
    print(f"Algorithms benchmarked: {len(results['algorithms'])}")
    print(f"Results saved to: data/processed/baseline_summary.json")


if __name__ == "__main__":
    main()
