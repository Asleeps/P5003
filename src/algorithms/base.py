from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any


class SignatureAlgorithm(ABC):
    """Base interface for all signature algorithms."""
    
    def __init__(self, name: str, security_level: str):
        self.name = name
        self.security_level = security_level
        self._public_key = None
        self._private_key = None
    
    @abstractmethod
    def generate_keypair(self) -> Tuple[bytes, bytes]:
        """Generate and return (public_key, private_key) pair."""
        pass
    
    @abstractmethod
    def sign(self, message: bytes) -> bytes:
        """Sign message and return signature."""
        pass
    
    @abstractmethod
    def verify(self, message: bytes, signature: bytes) -> bool:
        """Verify signature on message."""
        pass
    
    @abstractmethod
    def get_sizes(self) -> Dict[str, int]:
        """Return dict with public_key, private_key, and signature sizes in bytes."""
        pass
    
    def load_keypair(self, public_key: bytes, private_key: bytes):
        """Load existing keypair."""
        self._public_key = public_key
        self._private_key = private_key
    
    def get_info(self) -> Dict[str, Any]:
        """Return algorithm metadata."""
        return {
            'name': self.name,
            'security_level': self.security_level,
            'type': self.__class__.__name__
        }
