"""
Helpers for binding benchmark worker processes to specific CPU cores.

macOS does not expose a public API for strict CPU affinity like Linux.
This module therefore best-effort applies affinity using the mechanisms
available on the current platform (sched_setaffinity on Linux, psutil's
cpu_affinity where supported) and falls back gracefully when pinning is
not possible.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from typing import Callable, List, Optional

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - psutil optional
    psutil = None  # type: ignore


Logger = Callable[[str], None]


@dataclass
class AffinityPlan:
    """Describes how workers should be mapped to CPU cores."""

    cores: List[Optional[int]]


class AffinityManager:
    """
    Computes and applies CPU affinity for benchmark worker processes.

    The manager prefers performance cores and spills over to efficiency cores
    once all high-performance slots are in use.
    """

    def __init__(
        self,
        enabled: bool,
        performance_cores: Optional[List[int]] = None,
        efficiency_cores: Optional[List[int]] = None,
        strategy: str = "performance_first",
        logger: Optional[Logger] = None,
    ) -> None:
        self.enabled = enabled
        self.performance_cores = performance_cores or []
        self.efficiency_cores = efficiency_cores or []
        self.strategy = strategy
        self.logger = logger or (lambda msg: None)
        self._warned = False

    def build_plan(self, worker_count: int) -> AffinityPlan:
        """
        Generate a per-worker core plan. Workers beyond the known core list
        wrap around, ensuring every worker receives a deterministic core ID.
        """
        if not self.enabled or worker_count <= 0:
            return AffinityPlan([None] * worker_count)

        ordered: List[int] = []
        if self.strategy == "efficiency_first":
            ordered.extend(self.efficiency_cores)
            ordered.extend(self.performance_cores)
        else:
            ordered.extend(self.performance_cores)
            ordered.extend(self.efficiency_cores)

        if not ordered:
            return AffinityPlan([None] * worker_count)

        plan: List[Optional[int]] = []
        for i in range(worker_count):
            plan.append(ordered[i % len(ordered)])
        return AffinityPlan(plan)

    def apply(self, core_id: Optional[int], pid: Optional[int] = None) -> bool:
        """
        Attempt to bind the given pid (defaults to current process) to a core.

        Returns True if the operation was successful, False otherwise.
        """
        if not self.enabled or core_id is None:
            return False

        pid = pid or os.getpid()

        # Linux / BSD style sched_setaffinity
        if hasattr(os, "sched_setaffinity"):  # pragma: no cover - platform specific
            try:
                os.sched_setaffinity(pid, {core_id})
                return True
            except PermissionError:
                self._log_once("sched_setaffinity requires elevated permissions.")
            except OSError as exc:
                self._log_once(f"sched_setaffinity failed: {exc}")

        # psutil abstraction (Linux + Windows)
        if psutil is not None:  # pragma: no cover - optional dependency
            try:
                process = psutil.Process(pid)
                if hasattr(process, "cpu_affinity"):
                    process.cpu_affinity([core_id])
                    return True
            except Exception as exc:  # pylint: disable=broad-except
                self._log_once(f"psutil cpu_affinity failed: {exc}")

        # macOS fallback: expose an informative warning once.
        if platform.system() == "Darwin":  # pragma: no cover - macOS specific
            self._log_once(
                "macOS does not expose strict per-core pinning. "
                "Ensure background load is minimized for consistent results."
            )
            return False

        self._log_once("CPU affinity controls unavailable on this platform.")
        return False

    def _log_once(self, message: str) -> None:
        if not self._warned:
            self.logger(message)
            self._warned = True
