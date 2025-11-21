"""
API Gateway (JWT) scenario modeling.
Implements Chapter 5 of the experimental design.
"""


class JWTScenario:
    """
    Models API gateway JWT verification performance.
    
    Key characteristics:
    1. High-frequency verification (every API request)
    2. Asymmetric 1:N sign:verify ratio
    3. HTTP header size constraints
    """
    
    HTTP_HEADER_LIMIT = 8192  # 8KB typical limit
    SIGN_VERIFY_RATIO = 1_000_000  # One signing key serves many verifications
    
    def __init__(self, sign_verify_ratio: int = SIGN_VERIFY_RATIO):
        self.sign_verify_ratio = sign_verify_ratio
    
    def model_gateway_capacity(self, verify_throughput: dict) -> dict:
        """
        Calculate max API requests/sec based on verification throughput.
        Input: verification ops/sec from parallelism benchmark
        """
        if not verify_throughput:
            raise ValueError("verify_throughput cannot be empty")

        # Choose peak throughput across worker counts
        best_workers, peak_verify_ops = max(verify_throughput.items(), key=lambda kv: kv[1])
        baseline_workers = 1 if 1 in verify_throughput else min(verify_throughput)
        baseline_ops = verify_throughput[baseline_workers]
        efficiency = peak_verify_ops / (baseline_ops * best_workers) if baseline_ops > 0 else None

        # Signing side: how many issuer signatures per second needed to sustain verification load
        issuer_signs_per_sec = peak_verify_ops / self.sign_verify_ratio

        return {
            'peak_requests_per_sec': peak_verify_ops,
            'best_worker_count': best_workers,
            'baseline_ops_per_sec': baseline_ops,
            'per_worker_efficiency': efficiency,
            'issuer_signs_per_sec_needed': issuer_signs_per_sec,
            'verify_throughput_table': verify_throughput,
            'per_request_budget_ms': 1000.0 / peak_verify_ops if peak_verify_ops > 0 else None
        }
    
    def check_jwt_size(self, sig_size: int, metadata_overhead: int = 200) -> dict:
        """
        Check if JWT token fits in HTTP header limits.
        """
        total_size = sig_size + metadata_overhead
        
        return {
            'signature_bytes': sig_size,
            'total_jwt_bytes': total_size,
            'exceeds_header_limit': total_size > self.HTTP_HEADER_LIMIT,
            'usable': total_size <= self.HTTP_HEADER_LIMIT
        }
