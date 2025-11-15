from .base import SignatureAlgorithm
from .classical import RSASignature, ECDSASignature, EdDSASignature

try:
    from .post_quantum import DilithiumSignature, SPHINCSPlusSignature
    PQC_AVAILABLE = True
except (ImportError, RuntimeError):
    PQC_AVAILABLE = False
    DilithiumSignature = None
    SPHINCSPlusSignature = None

__all__ = [
    'SignatureAlgorithm',
    'RSASignature',
    'ECDSASignature',
    'EdDSASignature',
    'DilithiumSignature',
    'SPHINCSPlusSignature',
    'PQC_AVAILABLE',
]
