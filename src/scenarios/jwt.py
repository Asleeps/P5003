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
    
    def __init__(self):
        pass
    
    def model_gateway_capacity(self, verify_throughput: dict) -> dict:
        """
        Calculate max API requests/sec based on verification throughput.
        Input: verification ops/sec from parallelism benchmark
        """
        # TODO: Model gateway capacity
        pass
    
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
