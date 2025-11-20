#!/usr/bin/env python3
"""
Chapter 3: System-Level Parallelism Benchmarks (Threading Edition)

This runner drives cryptographic sign/verify loops using Python threads with
GIL-releasing implementations:

- Classical algorithms (RSA/ECDSA/EdDSA): Direct OpenSSL bindings via CFFI
- PQC algorithms (Dilithium/SPHINCS+): liboqs library

Both implementations release the Python GIL during cryptographic operations,
enabling true multi-core parallelism. This provides realistic throughput
numbers for production Python deployments with optimal threading performance.

Outputs:
- data/raw/parallelism/*.csv          → timestamped worker telemetry
- data/raw/parallelism/*.json         → power/system summary per config
- data/processed/parallelism_threading.json     → throughput/scaling matrices
- data/processed/parallelism_threading_summary.csv
"""

from __future__ import annotations

import base64
import csv
import json
import os
import queue
import sys
import argparse
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

MODE_CONFIG: Dict[str, Dict[str, Any]] = {
    "default": {
        "label": "Python Threading (All Algorithms)",
        "allowed_categories": None,
        "output_stem": "parallelism_threading",
        "thread_note": "",
        "use_openssl_direct": True,  # Use direct OpenSSL binding (GIL-releasing for classical algos)
        "description": "Multi-threaded benchmark with GIL-releasing implementations. Classical algorithms (RSA/ECDSA/EdDSA) use direct OpenSSL bindings via CFFI, while PQC algorithms (Dilithium/SPHINCS+) use liboqs. ALL algorithms release the GIL during cryptographic operations, enabling true multi-core parallelism and demonstrating optimal threading performance.",
    },
}

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore

# Add project root to path for relative imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.algorithms.factory import AlgorithmFactory
from src.utils.power import PowerMonitor
from src.utils.stats import StatisticsCollector


def _current_rss_bytes() -> Optional[int]:
    """Return current RSS bytes if psutil is available."""
    if psutil is None:
        return None
    try:
        return psutil.Process(os.getpid()).memory_info().rss  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - psutil can raise on exit
        return None


def worker_thread(
    worker_id: int,
    algo_name: str,
    algo_type: str,
    operation: str,
    public_key: bytes,
    private_key: bytes,
    signature: bytes,
    message: bytes,
    duration: float,
    analysis_window: float,
    warmup_duration: float,
    sample_interval: float,
    max_latency_samples: int,
    measurement_flags: Dict[str, Any],
    telemetry_queue: queue.Queue,
    ready_event: threading.Event,
    start_event: threading.Event,
    use_openssl_direct: bool = True,
) -> Dict[str, Any]:
    """
    Worker thread function. Executes cryptographic operations in a tight loop.
    
    Classical algorithms (RSA/ECDSA/EdDSA): Release GIL via OpenSSL direct binding (CFFI).
    PQC algorithms (Dilithium/SPHINCS+): Release GIL via liboqs.
    
    All algorithms enable true multi-core parallelism.
    """
    try:
        factory = AlgorithmFactory(use_openssl_direct=use_openssl_direct)
        algorithm = factory.create_algorithm(algo_type, algo_name)
        algorithm._public_key = public_key
        algorithm._private_key = private_key

        # Warmup loop (discarded from stats)
        warmup_end = time.perf_counter() + warmup_duration
        while time.perf_counter() < warmup_end:
            if operation == "sign":
                algorithm.sign(message)
            else:
                if not algorithm.verify(message, signature):
                    raise ValueError("Warmup verification failed")

        ready_event.set()
        start_event.wait()

        measurement_start = time.perf_counter()
        measurement_end = measurement_start + duration

        # Place the analysis window in the middle of the measurement interval
        analysis_window_actual = min(analysis_window, duration)
        margin = (duration - analysis_window_actual) / 2
        analysis_start = measurement_start + max(margin, 0)
        analysis_end = analysis_start + analysis_window_actual

        next_sample_time = measurement_start + sample_interval
        telemetry_enabled = measurement_flags.get("collect_memory", False) or measurement_flags.get(
            "collect_latencies", False
        )

        total_ops = 0
        analysis_ops = 0
        latency_samples: List[float] = []
        latency_sum = 0.0
        latency_sum_sq = 0.0
        rss_bytes = None
        cpu_time_start = time.process_time_ns()

        while True:
            now = time.perf_counter()
            if now >= measurement_end:
                break

            op_start = now
            # NOTE: All algorithms release GIL during cryptographic operations:
            # - Classical: via direct OpenSSL bindings (CFFI)
            # - PQC: via liboqs library
            if operation == "sign":
                algorithm.sign(message)
            else:
                if not algorithm.verify(message, signature):
                    raise ValueError("Verification failed during measurement")
            op_end = time.perf_counter()

            total_ops += 1
            if analysis_start <= op_end <= analysis_end:
                analysis_ops += 1
                latency_ms = (op_end - op_start) * 1000
                latency_sum += latency_ms
                latency_sum_sq += latency_ms * latency_ms
                if measurement_flags.get("collect_latencies", True) and len(latency_samples) < max_latency_samples:
                    latency_samples.append(latency_ms)

            if telemetry_enabled and op_end >= next_sample_time:
                rss_bytes = _current_rss_bytes()
                telemetry_queue.put(
                    {
                        "kind": "telemetry",
                        "worker_id": worker_id,
                        "timestamp": time.time(),
                        "ops_total": total_ops,
                        "analysis_ops": analysis_ops,
                        "rss_bytes": rss_bytes,
                    }
                )
                next_sample_time += sample_interval

        cpu_time_ns = time.process_time_ns() - cpu_time_start
        if rss_bytes is None:
            rss_bytes = _current_rss_bytes()

        return {
            "kind": "result",
            "worker_id": worker_id,
            "ops_total": total_ops,
            "analysis_ops": analysis_ops,
            "latency_samples": latency_samples,
            "latency_sum": latency_sum,
            "latency_sum_sq": latency_sum_sq,
            "analysis_window": analysis_window_actual,
            "cpu_time_ns": cpu_time_ns,
            "rss_bytes": rss_bytes,
        }
    except Exception as exc:  # pylint: disable=broad-except
        return {"kind": "error", "worker_id": worker_id, "message": str(exc)}


class CPUSampler:
    """Background sampler for psutil-based CPU utilization."""

    def __init__(self, enabled: bool, interval: float) -> None:
        self.enabled = enabled and psutil is not None
        self.interval = max(interval, 0.1)
        self.samples: List[Dict[str, Any]] = []
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._process = psutil.Process() if psutil is not None else None
        self._cpu_count = psutil.cpu_count() if psutil is not None else 1

    def start(self) -> None:
        if not self.enabled or psutil is None or self._process is None:
            return
        self.samples = []
        self._stop_event.clear()
        # Prime the process cpu_percent measurement with two warm-up calls
        # to establish baseline and stabilize measurement
        self._process.cpu_percent(interval=None)
        time.sleep(0.05)  # Short delay for initial measurement
        self._process.cpu_percent(interval=None)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        return self.samples

    def _run(self) -> None:
        assert psutil is not None and self._process is not None
        cpu_count = self._cpu_count or 1
        while not self._stop_event.wait(self.interval):
            sample_start = time.time()
            # Get process-level CPU percent (can exceed 100% on multi-core)
            # Use interval=0.05 for more responsive measurement within sampling window
            process_cpu_raw = self._process.cpu_percent(interval=0.05)
            # Normalize to 0-100% range by dividing by CPU count
            process_cpu = process_cpu_raw / cpu_count
            # Also get per-core system-wide CPU for reference
            percpu = psutil.cpu_percent(interval=0.05, percpu=True)  # type: ignore[attr-defined]
            avg = sum(percpu) / len(percpu) if percpu else None
            self.samples.append({
                "timestamp": sample_start,
                "process_cpu": process_cpu,
                "process_cpu_raw": process_cpu_raw,
                "percpu": percpu,
                "avg": avg,
                "sample_duration": time.time() - sample_start
            })


class ThreadingBenchmark:
    """Coordinator for threading-based parallelism benchmarks with configurable modes."""

    def __init__(self, mode_config: Dict[str, Any], config_path: Optional[str] = None) -> None:
        if config_path is None:
            config_path = str(project_root / "config" / "benchmark.json")

        with open(config_path, "r", encoding="utf-8") as fh:
            config = json.load(fh)

        self.config = config
        self.parallel_config = config["parallelism"]
        
        # Initialize mode_config early (needed for cache directory selection)
        self.mode_config = mode_config
        self.label = mode_config["label"]
        self.allowed_categories = mode_config.get("allowed_categories")
        self.output_stem = mode_config.get("output_stem", "parallelism_threading")
        self.thread_note = mode_config.get("thread_note", "")
        self.description = mode_config.get("description", "")
        self.use_openssl_direct = mode_config.get("use_openssl_direct", True)
        
        self.thread_counts = self.parallel_config["thread_counts"]
        self.duration = self.parallel_config["duration_sec"]
        self.analysis_window = self.parallel_config.get("analysis_window_sec", self.duration)
        self.warmup_duration = self.parallel_config.get("warmup_duration_sec", 1)
        self.sample_interval = self.parallel_config.get("sample_interval_sec", 0.25)
        self.max_latency_samples = self.parallel_config.get("max_latency_samples", 1024)
        self.measurement_flags = self.parallel_config.get("measurement", {})
        self.powermetrics_config = self.parallel_config.get("powermetrics", {})

        # Use separate cache directory for OpenSSL-based threading (DER format)
        base_cache_dir = self.parallel_config.get("cache_dir", "data/cache")
        cache_dir = f"{base_cache_dir}/openssl" if self.use_openssl_direct else f"{base_cache_dir}/cryptography"
        self.cache_dir = project_root / cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.raw_dir = project_root / "data" / "raw" / "parallelism"
        self.processed_dir = project_root / "data" / "processed"
        self.power_dir = project_root / "results" / "power"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.power_dir.mkdir(parents=True, exist_ok=True)

        self.factory = AlgorithmFactory(use_openssl_direct=self.use_openssl_direct)
        self.message = b"x" * config["experiment_params"]["message_size_bytes"]

    def _log_debug(self, message: str) -> None:
        print(f"[debug] {message}")

    def _cache_path(self, algo_name: str) -> Path:
        safe_name = algo_name.replace(" ", "_").replace("/", "_")
        return self.cache_dir / f"{safe_name}.json"

    def _load_cached_material(self, algo_name: str) -> Optional[Tuple[bytes, bytes, bytes]]:
        path = self._cache_path(algo_name)
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        try:
            public_key = base64.b64decode(data["public_key"])
            private_key = base64.b64decode(data["private_key"])
            signature = base64.b64decode(data["signature"])
            return public_key, private_key, signature
        except (KeyError, ValueError):
            return None

    def _store_cached_material(
        self, algo_name: str, public_key: bytes, private_key: bytes, signature: bytes
    ) -> None:
        path = self._cache_path(algo_name)
        payload = {
            "public_key": base64.b64encode(public_key).decode("ascii"),
            "private_key": base64.b64encode(private_key).decode("ascii"),
            "signature": base64.b64encode(signature).decode("ascii"),
            "timestamp": datetime.now().isoformat(),
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    def _generate_key_material(self, algo_name: str, algo_type: str) -> Tuple[bytes, bytes, bytes]:
        cached = self._load_cached_material(algo_name)
        if cached:
            print(f"  {algo_name}: loaded from cache")
            return cached

        print(f"  Generating keypair for {algo_name}...", end="", flush=True)
        start = time.time()
        algo = self.factory.create_algorithm(algo_type, algo_name)
        public_key, private_key = algo.generate_keypair()
        signature = algo.sign(self.message)
        elapsed = time.time() - start
        print(f" done ({elapsed:.1f}s)")
        
        self._store_cached_material(algo_name, public_key, private_key, signature)
        return public_key, private_key, signature

    def run_all(self) -> Dict[str, Any]:
        print("=" * 80)
        print(f"PARALLELISM THROUGHPUT BENCHMARK - {self.label}")
        print("=" * 80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        if self.description:
            print(self.description + "\n")

        algorithms = self.factory.get_all_algorithms()
        if self.allowed_categories:
            algorithms = [a for a in algorithms if a["category"] in self.allowed_categories]
        if not algorithms:
            raise RuntimeError("No algorithms match the selected category filter for this mode.")
        configs_per_algo = len(self.thread_counts) * 2
        est_time = len(algorithms) * configs_per_algo * (self.duration + self.warmup_duration + 2)

        print(f"Testing {len(algorithms)} algorithms")
        print(f"Thread counts: {self.thread_counts}")
        print(
            f"Duration per config: {self.duration}s (analysis window: {self.analysis_window}s, "
            f"warmup: {self.warmup_duration}s)"
        )
        print(f"Estimated total time: ~{format_duration(est_time)}\n")

        # Pre-generate all keypairs
        print("Preparing key materials...")
        for algo in algorithms:
            self._generate_key_material(algo["name"], algo["type"])
        print()

        all_results: Dict[str, Any] = {}
        start_time = time.time()

        for idx, algo in enumerate(algorithms, start=1):
            algo_name = algo["name"]
            algo_type = algo["type"]
            print(f"[{idx}/{len(algorithms)}] {algo_name} (Level {algo['security_level']})")

            try:
                public_key, private_key, signature = self._generate_key_material(algo_name, algo_type)
                algo_results = self._benchmark_algorithm(
                    algo_name, algo_type, public_key, private_key, signature
                )
                all_results[algo_name] = algo_results
            except Exception as exc:  # pylint: disable=broad-except
                print(f"  ✗ Error: {exc}")
                all_results[algo_name] = {"error": str(exc)}

            elapsed = time.time() - start_time
            remaining = max(est_time - elapsed, 0)
            print(f"  Progress: {progress_bar(idx, len(algorithms), width=30)}")
            print(f"  Elapsed: {format_duration(elapsed)} | Remaining: ~{format_duration(remaining)}")
            print()

        processed = self._process_results(all_results)
        self._save_processed(processed)

        total_time = time.time() - start_time
        print("=" * 80)
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total wall time: {format_duration(total_time)}")
        print(f"Raw data directory: {self.raw_dir}")
        print(f"Processed data directory: {self.processed_dir}")
        print("=" * 80)
        return processed

    def _benchmark_algorithm(
        self,
        algo_name: str,
        algo_type: str,
        public_key: bytes,
        private_key: bytes,
        signature: bytes,
    ) -> Dict[str, Any]:
        # Pre-flight check: ensure algorithm works in this process before spawning threads
        try:
            check_algo = self.factory.create_algorithm(algo_type, algo_name)
            check_algo.load_keypair(public_key, private_key)
            # Verify we can sign and verify
            test_sig = check_algo.sign(self.message)
            if not check_algo.verify(self.message, test_sig):
                raise RuntimeError("Self-verification failed")
        except Exception as e:
            raise RuntimeError(f"Algorithm pre-flight check failed: {e}")

        results: Dict[str, Dict[int, Dict[str, Any]]] = {"sign": {}, "verify": {}}
        for operation in ["sign", "verify"]:
            print(f"  {operation.capitalize()}:")
            for num_threads in self.thread_counts:
                config_result = self._run_configuration(
                    algo_name=algo_name,
                    algo_type=algo_type,
                    operation=operation,
                    num_threads=num_threads,
                    public_key=public_key,
                    private_key=private_key,
                    signature=signature,
                )
                results[operation][num_threads] = config_result

            baseline = results[operation][1]["ops_per_sec"]
            for num_threads in self.thread_counts:
                efficiency = results[operation][num_threads]["ops_per_sec"] / (baseline * num_threads)
                results[operation][num_threads]["scaling_efficiency"] = efficiency if baseline > 0 else 0.0

        return results

    def _run_configuration(
        self,
        algo_name: str,
        algo_type: str,
        operation: str,
        num_threads: int,
        public_key: bytes,
        private_key: bytes,
        signature: bytes,
    ) -> Dict[str, Any]:
        telemetry_queue: queue.Queue = queue.Queue()
        ready_events = [threading.Event() for _ in range(num_threads)]
        start_event = threading.Event()

        power_monitor = PowerMonitor(
            enabled=self.powermetrics_config.get("enabled", False),
            binary=self.powermetrics_config.get("binary", "/usr/bin/powermetrics"),
            samplers=self.powermetrics_config.get("samplers", ["smc"]),
            sample_interval_sec=self.powermetrics_config.get("sample_interval_sec", 1.0),
            output_dir=self.power_dir,
            logger=lambda msg: print(f"[powermetrics] {msg}"),
        )
        power_monitor.start(f"{algo_name}_{operation}_{num_threads}t")

        cpu_sampler = CPUSampler(
            enabled=self.measurement_flags.get("collect_cpu_percent", True),
            interval=self.sample_interval,
        )
        cpu_sampler.start()

        # Launch worker threads
        with ThreadPoolExecutor(max_workers=num_threads, thread_name_prefix=f"{operation}_worker") as executor:
            futures = []
            for worker_id in range(num_threads):
                future = executor.submit(
                    worker_thread,
                    worker_id,
                    algo_name,
                    algo_type,
                    operation,
                    public_key,
                    private_key,
                    signature,
                    self.message,
                    self.duration,
                    self.analysis_window,
                    self.warmup_duration,
                    self.sample_interval,
                    self.max_latency_samples,
                    self.measurement_flags,
                    telemetry_queue,
                    ready_events[worker_id],
                    start_event,
                    self.use_openssl_direct,
                )
                futures.append(future)

            # Wait for all workers to be ready
            for event in ready_events:
                event.wait(timeout=30)

            # Release workers into the measurement window
            start_event.set()

            # Wait for all workers to complete
            worker_results = []
            errors = []
            for future in futures:
                try:
                    result = future.result(timeout=self.duration + 10)
                    if result.get("kind") == "result":
                        worker_results.append(result)
                    elif result.get("kind") == "error":
                        errors.append(f"Worker {result.get('worker_id')} error: {result.get('message')}")
                except Exception as exc:  # pylint: disable=broad-except
                    errors.append(f"Future exception: {exc}")

        telemetry_records: List[Dict[str, Any]] = []
        self._drain_telemetry_queue(telemetry_queue, telemetry_records)

        cpu_samples = cpu_sampler.stop()
        power_summary = power_monitor.stop()

        if errors:
            raise RuntimeError("; ".join(errors))

        analysis_ops = sum(r["analysis_ops"] for r in worker_results)
        ops_per_sec = analysis_ops / self.analysis_window if self.analysis_window > 0 else 0
        latency_samples = [lat for r in worker_results for lat in r["latency_samples"]]
        latency_stats = None
        if latency_samples:
            latency_stats = StatisticsCollector.compute_stats(latency_samples)
            latency_stats.pop("samples", None)

        cpu_utils = [
            r["cpu_time_ns"] / (1e9 * self.analysis_window) if self.analysis_window > 0 else 0
            for r in worker_results
        ]
        rss_samples = [r["rss_bytes"] for r in worker_results if r["rss_bytes"]]
        avg_rss = sum(rss_samples) / len(rss_samples) if rss_samples else None

        cpu_summary = self._summarize_cpu_samples(cpu_samples)
        energy_per_op = None
        energy_joules = power_summary.get("energy_joules")
        if energy_joules is not None and analysis_ops > 0:
            energy_per_op = energy_joules / analysis_ops

        csv_path = self.raw_dir / f"{algo_name}_{operation}_t{num_threads}.csv"
        system_json_path = self.raw_dir / f"{algo_name}_{operation}_t{num_threads}_system.json"
        self._write_worker_csv(
            csv_path,
            telemetry_records,
            system_cpu=cpu_summary.get("avg_system"),
        )
        self._write_system_json(system_json_path, cpu_samples, power_summary)

        cpu_pct = cpu_summary.get('avg_process')
        cpu_str = f"{cpu_pct:.1f}%" if cpu_pct is not None else "N/A"
        
        thread_note = self.thread_note if (self.thread_note and num_threads > 1) else ""
        
        if latency_stats:
            msg = (
                f"    {num_threads:2d}t → {ops_per_sec:.1f} ops/sec | "
                f"p99: {latency_stats['p99']:.2f} ms | "
                f"CPU: {cpu_str}{thread_note}"
            )
        else:
            msg = f"    {num_threads:2d}t → {ops_per_sec:.1f} ops/sec | CPU: {cpu_str}{thread_note}"
        print(msg)

        return {
            "thread_count": num_threads,
            "ops_per_sec": ops_per_sec,
            "analysis_ops": analysis_ops,
            "latency_stats": latency_stats,
            "cpu_utilization": cpu_utils,
            "avg_cpu_utilization": sum(cpu_utils) / len(cpu_utils) if cpu_utils else None,
            "system_cpu_percent": cpu_summary.get("avg_process"),
            "avg_rss_bytes": avg_rss,
            "energy_per_op": energy_per_op,
            "power_summary": {k: v for k, v in power_summary.items() if k != "samples"},
            "system_cpu_summary": cpu_summary,
            "telemetry_file": str(csv_path),
            "system_file": str(system_json_path),
        }

    def _drain_telemetry_queue(self, telemetry_queue: queue.Queue, records: List[Dict[str, Any]]) -> None:
        while True:
            try:
                msg = telemetry_queue.get_nowait()
            except queue.Empty:
                break
            if msg.get("kind") == "telemetry":
                records.append(msg)

    def _write_worker_csv(
        self,
        path: Path,
        telemetry_records: List[Dict[str, Any]],
        system_cpu: Optional[float],
    ) -> None:
        headers = [
            "timestamp",
            "worker_id",
            "ops_total",
            "analysis_ops",
            "rss_bytes",
            "system_cpu_percent",
        ]
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(headers)
            for record in telemetry_records:
                writer.writerow(
                    [
                        f"{record.get('timestamp', 0):.6f}",
                        record.get("worker_id"),
                        record.get("ops_total"),
                        record.get("analysis_ops"),
                        record.get("rss_bytes"),
                        f"{system_cpu:.2f}" if system_cpu is not None else "",
                    ]
                )

    def _write_system_json(
        self,
        path: Path,
        cpu_samples: List[Dict[str, Any]],
        power_summary: Dict[str, Any],
    ) -> None:
        payload = {
            "cpu_samples": cpu_samples,
            "power": power_summary,
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    def _summarize_cpu_samples(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not samples:
            return {"avg_process": None, "avg_system": None, "avg_percpu": [], "sample_count": 0}
        
        # Process-level CPU (can exceed 100% on multi-core systems)
        process_cpus = [s["process_cpu"] for s in samples if "process_cpu" in s]
        avg_process = sum(process_cpus) / len(process_cpus) if process_cpus else None
        
        # System-level per-CPU averages
        percpu_lists = [sample["percpu"] for sample in samples if sample.get("percpu")]
        if percpu_lists:
            transposed = list(zip(*percpu_lists))
            avg_percpu = [sum(core) / len(core) for core in transposed]
            avg_system = sum(avg_percpu) / len(avg_percpu) if avg_percpu else None
        else:
            avg_percpu = []
            avg_system = None
            
        return {
            "avg_process": avg_process,
            "avg_system": avg_system,
            "avg_percpu": avg_percpu,
            "sample_count": len(samples)
        }

    def _process_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "platform": self.config["platform"],
            "thread_counts": self.thread_counts,
            "duration_sec": self.duration,
            "analysis_window_sec": self.analysis_window,
            "warmup_duration_sec": self.warmup_duration,
            "message_size_bytes": self.config["experiment_params"]["message_size_bytes"],
            "execution_model": "threading",
            "mode_label": self.label,
            "category_filter": self.allowed_categories,
            "output_stem": self.output_stem,
        }
        processed = {"metadata": metadata, "algorithms": {}}

        for algo_name, algo_data in results.items():
            if "error" in algo_data:
                processed["algorithms"][algo_name] = {"error": algo_data["error"]}
                continue

            summary = {
                "sign_ops_per_sec": {},
                "verify_ops_per_sec": {},
                "sign_scaling_efficiency": {},
                "verify_scaling_efficiency": {},
                "sign_latency_ms": {},
                "verify_latency_ms": {},
                "sign_energy_per_op": {},
                "verify_energy_per_op": {},
                "sign_cpu_percent": {},
                "verify_cpu_percent": {},
            }

            for operation in ["sign", "verify"]:
                ops_key = f"{operation}_ops_per_sec"
                eff_key = f"{operation}_scaling_efficiency"
                lat_key = f"{operation}_latency_ms"
                energy_key = f"{operation}_energy_per_op"
                cpu_key = f"{operation}_cpu_percent"

                for num_threads, metrics in algo_data[operation].items():
                    summary[ops_key][num_threads] = metrics["ops_per_sec"]
                    summary[eff_key][num_threads] = metrics["scaling_efficiency"]
                    summary[lat_key][num_threads] = metrics["latency_stats"]["p99"] if metrics["latency_stats"] else None
                    summary[energy_key][num_threads] = metrics["energy_per_op"]
                    summary[cpu_key][num_threads] = metrics.get("system_cpu_percent")

            processed["algorithms"][algo_name] = summary

        return processed

    def _save_processed(self, processed: Dict[str, Any]) -> None:
        json_path = self.processed_dir / f"{self.output_stem}.json"
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(processed, fh, indent=2)

        # Also dump a CSV summary for spreadsheet users
        csv_path = self.processed_dir / f"{self.output_stem}_summary.csv"
        header = ["algorithm", "operation", "thread_count", "ops_per_sec", "scaling_efficiency", "p99_latency_ms", "cpu_percent"]
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            for algo, data in processed["algorithms"].items():
                if "error" in data:
                    continue
                for operation in ["sign", "verify"]:
                    ops_key = f"{operation}_ops_per_sec"
                    eff_key = f"{operation}_scaling_efficiency"
                    lat_key = f"{operation}_latency_ms"
                    cpu_key = f"{operation}_cpu_percent"
                    for num_threads in self.thread_counts:
                        writer.writerow(
                            [
                                algo,
                                operation,
                                num_threads,
                                data[ops_key].get(num_threads),
                                data[eff_key].get(num_threads),
                                data[lat_key].get(num_threads),
                                data[cpu_key].get(num_threads),
                            ]
                        )


def progress_bar(current: int, total: int, width: int = 30, prefix: str = "") -> str:
    """Generate a progress bar string."""
    if total <= 0:
        return f"{prefix}[" + "░" * width + "] 0.0% (0/{total})"
    filled = int(width * current / total)
    bar = "█" * filled + "░" * (width - filled)
    percent = current / total * 100
    return f"{prefix}[{bar}] {percent:5.1f}% ({current}/{total})"


def format_duration(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    remaining = seconds % 60
    if remaining == 0:
        return f"{minutes}m"
    return f"{minutes}m{remaining}s"


def run_benchmark(mode: str, config_path: Optional[str] = None) -> Dict[str, Any]:
    mode_key = mode.lower()
    if mode_key not in MODE_CONFIG:
        raise ValueError(f"Unknown threading mode '{mode}'. Valid options: {', '.join(MODE_CONFIG)}")
    benchmark = ThreadingBenchmark(MODE_CONFIG[mode_key], config_path=config_path)
    return benchmark.run_all()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run threading parallelism benchmarks")
    parser.add_argument(
        "--config",
        default=None,
        help="Optional path to benchmark configuration JSON",
    )
    args = parser.parse_args()

    results = run_benchmark("default", config_path=args.config)

    print("\n" + "=" * 80)
    label = results["metadata"].get("mode_label", "Threading Mode")
    print(f"THROUGHPUT SUMMARY ({label})")
    print("=" * 80)
    for algo, data in results["algorithms"].items():
        if "error" in data:
            print(f"{algo}: ERROR - {data['error']}")
            continue
        max_threads = max(results["metadata"]["thread_counts"])
        sign_eff = data["sign_scaling_efficiency"].get(max_threads, 0) * 100
        verify_eff = data["verify_scaling_efficiency"].get(max_threads, 0) * 100
        sign_ops = data["sign_ops_per_sec"].get(max_threads)
        verify_ops = data["verify_ops_per_sec"].get(max_threads)
        sign_cpu = data["sign_cpu_percent"].get(max_threads)
        verify_cpu = data["verify_cpu_percent"].get(max_threads)
        sign_cpu_str = f"{sign_cpu:.1f}%" if sign_cpu is not None else "N/A"
        verify_cpu_str = f"{verify_cpu:.1f}%" if verify_cpu is not None else "N/A"
        print(
            f"{algo}: Sign {sign_ops:.1f} ops/sec ({sign_eff:.1f}% eff, {sign_cpu_str} CPU) | "
            f"Verify {verify_ops:.1f} ops/sec ({verify_eff:.1f}% eff, {verify_cpu_str} CPU)"
        )


if __name__ == "__main__":
    main()
