import numpy as np
from typing import List, Dict


class StatisticsCollector:
    """Collect and compute statistics for benchmark results."""
    
    @staticmethod
    def compute_stats(values: List[float]) -> Dict:
        """Compute basic statistics from a list of values."""
        arr = np.array(values)
        return {
            'mean': float(np.mean(arr)),
            'median': float(np.median(arr)),
            'std': float(np.std(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'p95': float(np.percentile(arr, 95)),
            'p99': float(np.percentile(arr, 99)),
            'samples': values  # Keep raw samples for detailed analysis
        }
    
    @staticmethod
    def format_time(seconds: float) -> str:
        """Format time in appropriate unit (s, ms, μs, ns)."""
        if seconds >= 1:
            return f"{seconds:.3f} s"
        elif seconds >= 1e-3:
            return f"{seconds * 1e3:.3f} ms"
        elif seconds >= 1e-6:
            return f"{seconds * 1e6:.3f} μs"
        else:
            return f"{seconds * 1e9:.3f} ns"
    
    @staticmethod
    def format_size(bytes_size: int) -> str:
        """Format size in appropriate unit (B, KB, MB)."""
        if bytes_size >= 1024 * 1024:
            return f"{bytes_size / (1024 * 1024):.2f} MB"
        elif bytes_size >= 1024:
            return f"{bytes_size / 1024:.2f} KB"
        else:
            return f"{bytes_size} B"
