"""
Code signing scenario modeling.
Implements Chapter 6 of the experimental design.
"""


class CodeSigningScenario:
    """
    Models application code signing verification (e.g., macOS Gatekeeper).
    
    Key characteristics:
    1. Per-launch verification (latency-sensitive)
    2. User-perceivable delays
    3. Multiple binaries per application
    """
    
    def __init__(self):
        pass
    
    def model_app_launch_delay(self, verify_latency_ms: float, num_binaries: int = 20) -> dict:
        """
        Calculate total launch delay for multi-binary application.
        Default: 20 binaries (typical for large apps like Xcode)
        """
        total_delay_ms = verify_latency_ms * num_binaries
        
        return {
            'single_verify_ms': verify_latency_ms,
            'num_binaries': num_binaries,
            'total_delay_ms': total_delay_ms,
            'total_delay_sec': total_delay_ms / 1000,
            'user_perceptible': total_delay_ms > 100  # >100ms noticeable
        }
