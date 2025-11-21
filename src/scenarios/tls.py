"""
TLS 1.3 handshake scenario modeling.
Implements Chapter 4 of the experimental design.
"""


class TLSScenario:
    """
    Models TLS 1.3 handshake performance impacts.
    
    Key bottlenecks:
    1. Server-side signing (CertificateVerify message)
    2. Certificate chain size vs TCP initcwnd (14KB threshold)
    """
    
    INITCWND_BYTES = 14336  # 14KB typical initial congestion window
    
    def __init__(self):
        pass
    
    def model_server_scalability(self, signing_throughput: dict) -> dict:
        """
        Calculate max handshakes/sec based on signing throughput.
        Input: signing ops/sec from parallelism benchmark
        """
        if not signing_throughput:
            raise ValueError("signing_throughput cannot be empty")

        # Use best-performing worker count
        best_threads, peak_sign_ops = max(signing_throughput.items(), key=lambda kv: kv[1])

        # Baseline for efficiency calculation
        baseline_threads = 1 if 1 in signing_throughput else min(signing_throughput)
        baseline_ops = signing_throughput[baseline_threads]
        efficiency = peak_sign_ops / (baseline_ops * best_threads) if baseline_ops > 0 else None

        # One CertificateVerify per handshake on the server
        return {
            'peak_handshakes_per_sec': peak_sign_ops,
            'best_thread_count': best_threads,
            'baseline_ops_per_sec': baseline_ops,
            'per_thread_efficiency': efficiency,
            'handshake_rtt_bound_ms': 1000.0 / peak_sign_ops if peak_sign_ops > 0 else None,
            'throughput_table': signing_throughput
        }
    
    def model_handshake_size(self, pk_size: int, sig_size: int) -> dict:
        """
        Calculate total handshake crypto bytes.
        Check if exceeds initcwnd threshold (causes extra RTT).
        """
        # Model: 2-level cert chain
        # Total ≈ PK(server) + Sig(CA) + Sig(handshake)
        total_bytes = pk_size + sig_size * 2
        
        return {
            'total_bytes': total_bytes,
            'exceeds_initcwnd': total_bytes > self.INITCWND_BYTES,
            'extra_rtt_risk': total_bytes > self.INITCWND_BYTES
        }
