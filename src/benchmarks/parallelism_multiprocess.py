#!/usr/bin/env python3
"""
Chapter 3: System-Level Parallelism Benchmarks (Multiprocessing Edition)

This runner drives cryptographic sign/verify loops using independent OS processes
to bypass Python's GIL. Each worker runs in a separate Python interpreter with
its own GIL, enabling true multi-core parallelism.

Used for Section 3.5 controlled experiment (Threading vs Multiprocessing comparison).

Outputs:
- data/raw/parallelism/*.csv          → timestamped worker telemetry
- data/raw/parallelism/*.json         → power/system summary per config
- data/processed/parallelism_multiprocess.json → throughput/scaling matrices
- data/processed/parallelism_multiprocess_summary.csv
"""

from __future__ import annotations

import base64
import binascii
import csv
import json
import os
import queue
import sys
import threading
import time
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore

# Add project root to path for relative imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.algorithms.factory import AlgorithmFactory
from src.utils.affinity import AffinityManager
from src.utils.power import PowerMonitor
from src.utils.stats import StatisticsCollector


def _current_rss_bytes() -> Optional[int]:
    """Return current RSS bytes if psutil is available."""
    if psutil is None:
        return None
    try:
        return psutil.Process().memory_info().rss  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover
        return None


def _apply_worker_affinity(core_id: Optional[int], enabled: bool) -> None:
    """Apply CPU affinity inside the worker process when requested."""
    if not enabled or core_id is None:
        return
    manager = AffinityManager(enabled=True, performance_cores=[core_id], efficiency_cores=[], logger=lambda _: None)
    manager.apply(core_id)


def worker_entry(
    payload: Dict[str, Any],
    telemetry_queue: Any,
    result_queue: Any,
    ready_queue: Any,
    start_event: Any,
) -> None:
    """Process entry point for multiprocessing workers."""
    worker_id = payload["worker_id"]
    algo_name = payload["algo_name"]
    algo_type = payload["algo_type"]
    operation = payload["operation"]
    duration = payload["duration"]
    analysis_window = payload["analysis_window"]
    warmup_duration = payload["warmup_duration"]
    sample_interval = payload["sample_interval"]
    max_latency_samples = payload["max_latency_samples"]
    measurement_flags = payload["measurement_flags"]
    message = payload["message"]
    signature = payload.get("signature", b"")

    try:
        _apply_worker_affinity(payload.get("core_id"), payload.get("affinity_enabled", False))

        factory = AlgorithmFactory()
        algorithm = factory.create_algorithm(algo_type, algo_name)
        algorithm._public_key = payload["public_key"]
        algorithm._private_key = payload["private_key"]

        warmup_end = time.perf_counter() + warmup_duration
        while time.perf_counter() < warmup_end:
            if operation == "sign":
                algorithm.sign(message)
            else:
                if not algorithm.verify(message, signature):
                    raise ValueError("Warmup verification failed")

        ready_queue.put({"worker_id": worker_id, "status": "ready"})
        start_event.wait()

        measurement_start = time.perf_counter()
        measurement_end = measurement_start + duration
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

        result_queue.put(
            {
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
        )
    except Exception as exc:  # pylint: disable=broad-except
        result_queue.put({"kind": "error", "worker_id": worker_id, "message": str(exc)})


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
        # Prime the process cpu_percent measurement for main process
        self._process.cpu_percent(interval=None)
        # Prime all child processes as well
        for child in self._process.children(recursive=True):
            try:
                child.cpu_percent(interval=None)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
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
            # Get process-level CPU percent including all child processes
            process_cpu_raw = self._process.cpu_percent(interval=None)
            # Add CPU from all child processes using blocking measurement
            try:
                for child in self._process.children(recursive=True):
                    try:
                        # Use blocking mode with 0.05s interval for immediate accurate reading
                        process_cpu_raw += child.cpu_percent(interval=0.05)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            # Normalize to 0-100% range by dividing by CPU count
            process_cpu = process_cpu_raw / cpu_count
            # Also get per-core system-wide CPU for reference
            percpu = psutil.cpu_percent(interval=None, percpu=True)  # type: ignore[attr-defined]
            avg = sum(percpu) / len(percpu) if percpu else None
            self.samples.append({
                "timestamp": time.time(),
                "process_cpu": process_cpu,
                "process_cpu_raw": process_cpu_raw,
                "percpu": percpu,
                "avg": avg
            })


class ParallelismBenchmark:
    """Coordinator for the multi-process parallelism benchmarks."""

    def __init__(self, config_path: Optional[str] = None) -> None:
        if config_path is None:
            config_path = str(project_root / "config" / "benchmark.json")

        with open(config_path, "r", encoding="utf-8") as fh:
            config = json.load(fh)

        self.config = config
        self.parallel_config = config["parallelism"]
        self.thread_counts = self.parallel_config["thread_counts"]
        self.duration = self.parallel_config["duration_sec"]
        self.analysis_window = self.parallel_config.get("analysis_window_sec", self.duration)
        self.warmup_duration = self.parallel_config.get("warmup_duration_sec", 1)
        self.sample_interval = self.parallel_config.get("sample_interval_sec", 0.25)
        self.max_latency_samples = self.parallel_config.get("max_latency_samples", 1024)
        self.measurement_flags = self.parallel_config.get("measurement", {})
        self.powermetrics_config = self.parallel_config.get("powermetrics", {})

        cache_dir = self.parallel_config.get("cache_dir", "data/cache")
        self.cache_dir = project_root / cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.raw_dir = project_root / "data" / "raw" / "parallelism"
        self.processed_dir = project_root / "data" / "processed"
        self.power_dir = project_root / "results" / "power"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.power_dir.mkdir(parents=True, exist_ok=True)

        affinity_cfg = self.parallel_config.get("cpu_affinity", {})
        self.affinity_manager = AffinityManager(
            enabled=affinity_cfg.get("enabled", False),
            performance_cores=affinity_cfg.get("performance_cores"),
            efficiency_cores=affinity_cfg.get("efficiency_cores"),
            strategy=affinity_cfg.get("strategy", "performance_first"),
            logger=self._log_debug,
        )
        self.affinity_enabled = affinity_cfg.get("enabled", False)

        self.factory = AlgorithmFactory()
        self.message = b"x" * config["experiment_params"]["message_size_bytes"]
        self.ctx = get_context("spawn")

    def _log_debug(self, message: str) -> None:
        print(f"[affinity] {message}")

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
        except (KeyError, ValueError, binascii.Error):
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
        print("PARALLELISM THROUGHPUT BENCHMARK - MULTIPROCESSING MODE")
        print("(Section 3.5 Controlled Experiment: GIL-Free Baseline)")
        print("=" * 80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        algorithms = self.factory.get_all_algorithms()
        configs_per_algo = len(self.thread_counts) * 2
        est_time = len(algorithms) * configs_per_algo * (self.duration + self.warmup_duration + 2)

        print(f"Testing {len(algorithms)} algorithms")
        print(f"Worker counts: {self.thread_counts}")
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
        results: Dict[str, Dict[int, Dict[str, Any]]] = {"sign": {}, "verify": {}}
        for operation in ["sign", "verify"]:
            print(f"  {operation.capitalize()}:")
            for workers in self.thread_counts:
                config_result = self._run_configuration(
                    algo_name=algo_name,
                    algo_type=algo_type,
                    operation=operation,
                    worker_count=workers,
                    public_key=public_key,
                    private_key=private_key,
                    signature=signature,
                )
                results[operation][workers] = config_result

            baseline = results[operation][1]["ops_per_sec"]
            for workers in self.thread_counts:
                efficiency = results[operation][workers]["ops_per_sec"] / (baseline * workers)
                results[operation][workers]["scaling_efficiency"] = efficiency if baseline > 0 else 0.0

        return results

    def _run_configuration(
        self,
        algo_name: str,
        algo_type: str,
        operation: str,
        worker_count: int,
        public_key: bytes,
        private_key: bytes,
        signature: bytes,
    ) -> Dict[str, Any]:
        ctx = self.ctx
        telemetry_queue = ctx.Queue()
        result_queue = ctx.Queue()
        ready_queue = ctx.Queue()
        start_event = ctx.Event()

        plan = self.affinity_manager.build_plan(worker_count)
        processes: List[Any] = []
        for worker_id in range(worker_count):
            payload = {
                "worker_id": worker_id,
                "algo_name": algo_name,
                "algo_type": algo_type,
                "operation": operation,
                "duration": self.duration,
                "analysis_window": self.analysis_window,
                "warmup_duration": self.warmup_duration,
                "sample_interval": self.sample_interval,
                "max_latency_samples": self.max_latency_samples,
                "measurement_flags": self.measurement_flags,
                "message": self.message,
                "signature": signature if operation == "verify" else b"",
                "public_key": public_key,
                "private_key": private_key,
                "core_id": plan.cores[worker_id] if plan.cores else None,
                "affinity_enabled": self.affinity_enabled,
            }
            proc = ctx.Process(
                target=worker_entry,
                args=(payload, telemetry_queue, result_queue, ready_queue, start_event),
            )
            proc.start()
            processes.append(proc)

        ready_count = 0
        while ready_count < worker_count:
            ready_msg = ready_queue.get()
            if ready_msg.get("status") == "ready":
                ready_count += 1

        power_monitor = PowerMonitor(
            enabled=self.powermetrics_config.get("enabled", False),
            binary=self.powermetrics_config.get("binary", "/usr/bin/powermetrics"),
            samplers=self.powermetrics_config.get("samplers", ["smc"]),
            sample_interval_sec=self.powermetrics_config.get("sample_interval_sec", 1.0),
            output_dir=self.power_dir,
            logger=lambda msg: print(f"[powermetrics] {msg}"),
        )
        power_monitor.start(f"{algo_name}_{operation}_{worker_count}t")

        cpu_sampler = CPUSampler(
            enabled=self.measurement_flags.get("collect_cpu_percent", True),
            interval=self.sample_interval,
        )
        cpu_sampler.start()

        # Release workers into the measurement window
        start_event.set()

        telemetry_records: List[Dict[str, Any]] = []
        worker_results: List[Dict[str, Any]] = []
        errors: List[str] = []

        while len(worker_results) < worker_count and not errors:
            try:
                msg = result_queue.get(timeout=self.duration + 5)
            except queue.Empty:
                msg = "Timeout waiting for worker results"
                print(f"  ⚠ {msg}; aborting configuration.")
                errors.append(msg)
                break

            if msg.get("kind") == "result":
                worker_results.append(msg)
            elif msg.get("kind") == "error":
                errors.append(f"Worker {msg.get('worker_id')} error: {msg.get('message')}")
            self._drain_telemetry_queue(telemetry_queue, telemetry_records)

        self._drain_telemetry_queue(telemetry_queue, telemetry_records)

        cpu_samples = cpu_sampler.stop()
        power_summary = power_monitor.stop()

        for process in processes:
            process.join(timeout=2)
            if process.exitcode not in (0, None):
                errors.append(f"Worker exited with code {process.exitcode}")

        if errors:
            raise RuntimeError("; ".join(errors))

        analysis_window_actual = worker_results[0]["analysis_window"] if worker_results else self.analysis_window
        analysis_ops = sum(r["analysis_ops"] for r in worker_results)
        ops_per_sec = (
            analysis_ops / analysis_window_actual
            if analysis_window_actual > 0
            else 0
        )
        latency_samples = [lat for r in worker_results for lat in r["latency_samples"]]
        latency_stats = None
        if latency_samples:
            latency_stats = StatisticsCollector.compute_stats(latency_samples)
            latency_stats.pop("samples", None)

        cpu_utils = [
            r["cpu_time_ns"] / (1e9 * analysis_window_actual) if analysis_window_actual > 0 else 0
            for r in worker_results
        ]
        rss_samples = [r["rss_bytes"] for r in worker_results if r["rss_bytes"]]
        avg_rss = sum(rss_samples) / len(rss_samples) if rss_samples else None

        cpu_summary = self._summarize_cpu_samples(cpu_samples)
        energy_per_op = None
        energy_joules = power_summary.get("energy_joules")
        if energy_joules is not None and analysis_ops > 0:
            energy_per_op = energy_joules / analysis_ops

        csv_path = self.raw_dir / f"{algo_name}_{operation}_t{worker_count}.csv"
        system_json_path = self.raw_dir / f"{algo_name}_{operation}_t{worker_count}_system.json"
        self._write_worker_csv(
            csv_path,
            telemetry_records,
            system_cpu=cpu_summary.get("avg_system"),
            avg_temperature=power_summary.get("avg_temperature_c"),
        )
        self._write_system_json(system_json_path, cpu_samples, power_summary)

        cpu_pct = cpu_summary.get('avg_process')
        cpu_str = f"{cpu_pct:.1f}%" if cpu_pct is not None else "N/A"

        if latency_stats:
            msg = (
                f"    {worker_count:2d}t → {ops_per_sec:10.1f} ops/sec | "
                f"p99: {latency_stats['p99']:.2f} ms | "
                f"CPU: {cpu_str}"
            )
        else:
            msg = f"    {worker_count:2d}t → {ops_per_sec:10.1f} ops/sec | CPU: {cpu_str}"
        print(msg)

        return {
            "worker_count": worker_count,
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

    def _drain_telemetry_queue(self, telemetry_queue: Any, records: List[Dict[str, Any]]) -> None:
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
        avg_temperature: Optional[float],
    ) -> None:
        headers = [
            "timestamp",
            "worker_id",
            "ops_total",
            "analysis_ops",
            "rss_bytes",
            "system_cpu_percent",
            "cpu_temperature_c",
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
                        f"{avg_temperature:.2f}" if avg_temperature is not None else "",
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
            "execution_model": "multiprocessing",
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

                for workers, metrics in algo_data[operation].items():
                    summary[ops_key][workers] = metrics["ops_per_sec"]
                    summary[eff_key][workers] = metrics["scaling_efficiency"]
                    summary[lat_key][workers] = metrics["latency_stats"]["p99"] if metrics["latency_stats"] else None
                    summary[energy_key][workers] = metrics["energy_per_op"]
                    summary[cpu_key][workers] = metrics.get("system_cpu_percent")

            processed["algorithms"][algo_name] = summary

        return processed

    def _save_processed(self, processed: Dict[str, Any]) -> None:
        json_path = self.processed_dir / "parallelism_multiprocess.json"
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(processed, fh, indent=2)

        # Also dump a CSV summary for spreadsheet users
        csv_path = self.processed_dir / "parallelism_multiprocess_summary.csv"
        header = ["algorithm", "operation", "worker_count", "ops_per_sec", "scaling_efficiency", "p99_latency_ms", "cpu_percent"]
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
                    for workers in self.thread_counts:
                        writer.writerow(
                            [
                                algo,
                                operation,
                                workers,
                                data[ops_key].get(workers),
                                data[eff_key].get(workers),
                                data[lat_key].get(workers),
                                data[cpu_key].get(workers),
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


def main() -> None:
    benchmark = ParallelismBenchmark()
    results = benchmark.run_all()

    print("\n" + "=" * 80)
    print("THROUGHPUT SUMMARY (MULTIPROCESSING MODE - GIL-FREE)")
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
        print(
            f"{algo}: Sign {sign_ops:.1f} ops/sec ({sign_eff:.1f}% eff) | "
            f"Verify {verify_ops:.1f} ops/sec ({verify_eff:.1f}% eff)"
        )


if __name__ == "__main__":
    main()
