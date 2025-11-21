#!/usr/bin/env python3
"""
Convenience wrapper to run parallelism benchmarks (threading or multiprocessing).

Usage examples:
    python scripts/run_parallelism.py --mode threading
    python scripts/run_parallelism.py --mode multiprocess
"""

import argparse
from pathlib import Path

from src.benchmarks.parallelism_threading import run_benchmark as run_threading
from src.benchmarks.parallelism_multiprocess import ParallelismBenchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Run parallelism throughput benchmarks")
    parser.add_argument(
        "--mode",
        choices=["threading", "multiprocess"],
        default="threading",
        help="Execution model",
    )
    parser.add_argument(
        "--config",
        default=str(Path("config") / "benchmark.json"),
        help="Path to benchmark configuration JSON",
    )
    args = parser.parse_args()

    if args.mode == "threading":
        results = run_threading("default", config_path=args.config)
        label = results["metadata"].get("mode_label", "Threading Mode")
        max_threads = max(results["metadata"]["thread_counts"])
    else:
        benchmark = ParallelismBenchmark(config_path=args.config)
        results = benchmark.run_all()
        label = "Multiprocessing Mode - GIL-free"
        max_threads = max(results["metadata"]["thread_counts"])

    print("\n" + "=" * 80)
    print(f"THROUGHPUT SUMMARY ({label})")
    print("=" * 80)
    for algo, data in results["algorithms"].items():
        if "error" in data:
            print(f"{algo}: ERROR - {data['error']}")
            continue
        sign_eff = data["sign_scaling_efficiency"].get(max_threads, 0) * 100
        verify_eff = data["verify_scaling_efficiency"].get(max_threads, 0) * 100
        sign_ops = data["sign_ops_per_sec"].get(max_threads)
        verify_ops = data["verify_ops_per_sec"].get(max_threads)
        print(
            f"{algo}: Sign {sign_ops:.1f} ops/sec ({sign_eff:.1f}% eff) | "
            f"Verify {verify_ops:.1f} ops/sec ({verify_eff:.1f}% eff)"
        )


if __name__ == "__main__":
    main()
