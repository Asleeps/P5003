try:
    import oqs
    LIBOQS_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    LIBOQS_AVAILABLE = False
    _LIBOQS_ERROR = str(e)

import base64
from typing import Tuple, Dict
from .base import SignatureAlgorithm


class DilithiumSignature(SignatureAlgorithm):
    """Dilithium post-quantum signature implementation using liboqs."""
    
    def __init__(self, name: str, security_level: str, liboqs_name: str):
        if not LIBOQS_AVAILABLE:
            raise RuntimeError(f"liboqs not available: {_LIBOQS_ERROR}")
        super().__init__(name, security_level)
        self.liboqs_name = liboqs_name
        self._signer = None
    
    def generate_keypair(self) -> Tuple[bytes, bytes]:
        self._signer = oqs.Signature(self.liboqs_name)
        public_key = self._signer.generate_keypair()
        private_key = self._signer.export_secret_key()
        
        self._public_key = public_key
        self._private_key = private_key
        return public_key, private_key
    
    def sign(self, message: bytes) -> bytes:
        if self._signer is None:
            if self._private_key is None:
                raise ValueError("No keypair loaded")
            # Handle base64-encoded keys from cache
            key_bytes = base64.b64decode(self._private_key) if isinstance(self._private_key, str) else self._private_key
            self._signer = oqs.Signature(self.liboqs_name, key_bytes)
        
        return self._signer.sign(message)
    
    def verify(self, message: bytes, signature: bytes) -> bool:
        if self._public_key is None:
            raise ValueError("No public key loaded")
        
        # Handle base64-encoded keys from cache
        key_bytes = base64.b64decode(self._public_key) if isinstance(self._public_key, str) else self._public_key
        
        verifier = oqs.Signature(self.liboqs_name)
        return verifier.verify(message, signature, key_bytes)
    
    def get_sizes(self) -> Dict[str, int]:
        if self._signer is None:
            self.generate_keypair()
        
        return {
            'public_key': self._signer.details['length_public_key'],
            'private_key': self._signer.details['length_secret_key'],
            'signature': self._signer.details['length_signature']
        }


class SPHINCSPlusSignature(SignatureAlgorithm):
    """SPHINCS+ post-quantum signature implementation using liboqs."""
    
    def __init__(self, name: str, security_level: str, liboqs_name: str):
        if not LIBOQS_AVAILABLE:
            raise RuntimeError(f"liboqs not available: {_LIBOQS_ERROR}")
        super().__init__(name, security_level)
        self.liboqs_name = liboqs_name
        self._signer = None
    
    def generate_keypair(self) -> Tuple[bytes, bytes]:
        self._signer = oqs.Signature(self.liboqs_name)
        public_key = self._signer.generate_keypair()
        private_key = self._signer.export_secret_key()
        
        self._public_key = public_key
        self._private_key = private_key
        return public_key, private_key
    
    def sign(self, message: bytes) -> bytes:
        if self._signer is None:
            if self._private_key is None:
                raise ValueError("No keypair loaded")
            # Handle base64-encoded keys from cache
            key_bytes = base64.b64decode(self._private_key) if isinstance(self._private_key, str) else self._private_key
            self._signer = oqs.Signature(self.liboqs_name, key_bytes)
        
        return self._signer.sign(message)
    
    def verify(self, message: bytes, signature: bytes) -> bool:
        if self._public_key is None:
            raise ValueError("No public key loaded")
        
        # Handle base64-encoded keys from cache
        key_bytes = base64.b64decode(self._public_key) if isinstance(self._public_key, str) else self._public_key
        
        verifier = oqs.Signature(self.liboqs_name)
        return verifier.verify(message, signature, key_bytes)
    
    def get_sizes(self) -> Dict[str, int]:
        if self._signer is None:
            self.generate_keypair()
        
        return {
            'public_key': self._signer.details['length_public_key'],
            'private_key': self._signer.details['length_secret_key'],
            'signature': self._signer.details['length_signature']
        }
