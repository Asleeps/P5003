"""
Utilities for sampling power/thermal data via macOS powermetrics.

The helper intentionally degrades gracefully when powermetrics is not
available (e.g., missing binary or insufficient privileges).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

Logger = Callable[[str], None]

TEMP_PATTERN = re.compile(r"CPU die temperature:\s+([\d\.]+)\s*C", re.IGNORECASE)
POWER_PATTERN = re.compile(r"CPU [Pp]ower:\s+([\d\.]+)\s+(mW|W)", re.IGNORECASE)
ENERGY_PATTERN = re.compile(r"CPU energy:\s+([\d\.]+)\s+J", re.IGNORECASE)


class PowerMonitor:
    """
    Wraps powermetrics sampling in a background process and parses
    coarse power/temperature data for later analysis.
    """

    def __init__(
        self,
        enabled: bool,
        binary: str,
        samplers: Optional[List[str]],
        sample_interval_sec: float,
        output_dir: Path,
        logger: Optional[Logger] = None,
    ) -> None:
        self.enabled = enabled
        self.binary = binary
        self.samplers = samplers or ["smc"]
        self.sample_interval_sec = max(sample_interval_sec, 0.1)
        self.output_dir = output_dir
        self.logger = logger or (lambda msg: None)

        self._process: Optional[subprocess.Popen[str]] = None
        self._collector_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._samples: List[Dict[str, float]] = []
        self._energy_samples: List[float] = []
        self._log_path: Optional[Path] = None
        self._start_time: Optional[float] = None
        self._stop_time: Optional[float] = None

    def start(self, label: str) -> None:
        """Begin powermetrics sampling if enabled."""
        if not self.enabled:
            return

        if shutil.which(self.binary) is None:
            self.logger(f"powermetrics binary '{self.binary}' not found; disabling sampling.")
            self.enabled = False
            return

        # powermetrics requires elevated privileges; detect early when possible
        if os.geteuid() != 0:
            self.logger("powermetrics requires sudo access; disable powermetrics or re-run with sudo.")
            self.enabled = False
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self._log_path = self.output_dir / f"powermetrics_{label}_{timestamp}.log"
        interval_ms = max(int(self.sample_interval_sec * 1000), 10)
        cmd = [
            self.binary,
            "--samplers",
            ",".join(self.samplers),
            "-i",
            str(interval_ms),
        ]

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except Exception as exc:  # pylint: disable=broad-except
            self.logger(f"Failed to start powermetrics: {exc}")
            self.enabled = False
            self._process = None
            return

        self._stop_event.clear()
        self._samples = []
        self._energy_samples = []
        self._collector_thread = threading.Thread(target=self._collector_loop, daemon=True)
        self._collector_thread.start()
        self._start_time = time.time()

    def stop(self) -> Dict[str, Any]:
        """Stop powermetrics sampling and return aggregated stats."""
        if not self.enabled or self._process is None:
            return {
                "energy_joules": None,
                "avg_power_w": None,
                "avg_temperature_c": None,
                "log_path": None,
                "sample_count": 0,
                "samples": [],
            }

        self._stop_event.set()
        self._stop_time = time.time()

        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()

        if self._collector_thread is not None:
            self._collector_thread.join(timeout=2)

        avg_temp = self._average_sample_key("temperature_c")
        avg_power = self._average_sample_key("power_w")
        total_energy = self._compute_energy(avg_power)

        return {
            "energy_joules": total_energy,
            "avg_power_w": avg_power,
            "avg_temperature_c": avg_temp,
            "log_path": str(self._log_path) if self._log_path else None,
            "sample_count": len(self._samples),
            "samples": list(self._samples),
        }

    def _collector_loop(self) -> None:
        assert self._process is not None
        stdout = self._process.stdout
        if stdout is None:
            return

        log_file = self._log_path.open("w") if self._log_path else None

        try:
            for line in stdout:
                timestamp = time.time()
                if log_file:
                    log_file.write(line)

                temp_match = TEMP_PATTERN.search(line)
                power_match = POWER_PATTERN.search(line)
                energy_match = ENERGY_PATTERN.search(line)

                sample: Dict[str, float] = {}
                if temp_match:
                    sample["temperature_c"] = float(temp_match.group(1))
                if power_match:
                    power_value = float(power_match.group(1))
                    unit = power_match.group(2)
                    # Convert mW to W
                    sample["power_w"] = power_value / 1000.0 if unit.lower() == "mw" else power_value
                if sample:
                    sample["timestamp"] = timestamp
                    self._samples.append(sample)
                if energy_match:
                    self._energy_samples.append(float(energy_match.group(1)))

                if self._stop_event.is_set():
                    break
        finally:
            if log_file:
                log_file.flush()
                log_file.close()

    def _average_sample_key(self, key: str) -> Optional[float]:
        values = [sample[key] for sample in self._samples if key in sample]
        if not values:
            return None
        return sum(values) / len(values)

    def _compute_energy(self, avg_power: Optional[float]) -> Optional[float]:
        if len(self._energy_samples) >= 2:
            start = self._energy_samples[0]
            end = self._energy_samples[-1]
            return max(end - start, 0.0)

        if avg_power is not None and self._start_time and self._stop_time:
            duration = max(self._stop_time - self._start_time, 0.0)
            return avg_power * duration

        return None
