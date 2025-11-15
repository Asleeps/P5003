import time
from typing import List


class HighPrecisionTimer:
    """High-precision timer for performance measurements."""
    
    def __init__(self):
        self.start_time = None
        self.elapsed = None
    
    def start(self):
        """Start timing."""
        self.start_time = time.perf_counter()
    
    def stop(self) -> float:
        """Stop timing and return elapsed time in seconds."""
        if self.start_time is None:
            raise RuntimeError("Timer not started")
        self.elapsed = time.perf_counter() - self.start_time
        return self.elapsed
    
    @staticmethod
    def measure_operation(operation, iterations: int = 1) -> float:
        """Measure average time for an operation over multiple iterations."""
        timer = HighPrecisionTimer()
        timer.start()
        for _ in range(iterations):
            operation()
        total_time = timer.stop()
        return total_time / iterations
    
    @staticmethod
    def measure_with_warmup(operation, warmup: int, iterations: int) -> float:
        """Measure operation with warmup phase."""
        for _ in range(warmup):
            operation()
        
        return HighPrecisionTimer.measure_operation(operation, iterations)
