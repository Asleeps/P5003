from cryptography.hazmat.primitives.asymmetric import rsa, ec, ed25519, ed448
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from typing import Tuple, Dict
from .base import SignatureAlgorithm


class RSASignature(SignatureAlgorithm):
    """RSA signature implementation using cryptography library."""
    
    def __init__(self, name: str, security_level: str, key_size: int):
        super().__init__(name, security_level)
        self.key_size = key_size
        self._private_key_obj = None
        self._public_key_obj = None
    
    def generate_keypair(self) -> Tuple[bytes, bytes]:
        self._private_key_obj = rsa.generate_private_key(
            public_exponent=65537,
            key_size=self.key_size
        )
        self._public_key_obj = self._private_key_obj.public_key()
        
        private_bytes = self._private_key_obj.private_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        public_bytes = self._public_key_obj.public_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        self._private_key = private_bytes
        self._public_key = public_bytes
        return public_bytes, private_bytes
    
    def sign(self, message: bytes) -> bytes:
        if self._private_key_obj is None:
            if self._private_key is None:
                raise ValueError("No keypair loaded")
            self._private_key_obj = serialization.load_der_private_key(
                self._private_key, password=None
            )
        
        signature = self._private_key_obj.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return signature
    
    def verify(self, message: bytes, signature: bytes) -> bool:
        if self._public_key_obj is None:
            if self._public_key is None:
                raise ValueError("No public key loaded")
            self._public_key_obj = serialization.load_der_public_key(self._public_key)
        
        try:
            self._public_key_obj.verify(
                signature,
                message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception:
            return False
    
    def get_sizes(self) -> Dict[str, int]:
        if self._public_key is None or self._private_key is None:
            self.generate_keypair()
        
        test_sig = self.sign(b"test" * 8)
        return {
            'public_key': len(self._public_key),
            'private_key': len(self._private_key),
            'signature': len(test_sig)
        }


class ECDSASignature(SignatureAlgorithm):
    """ECDSA signature implementation using cryptography library."""
    
    def __init__(self, name: str, security_level: str, curve_name: str):
        super().__init__(name, security_level)
        self.curve_name = curve_name
        self.curve = self._get_curve(curve_name)
        self._private_key_obj = None
        self._public_key_obj = None
    
    def _get_curve(self, curve_name: str):
        curves = {
            'secp256r1': ec.SECP256R1(),
            'secp384r1': ec.SECP384R1(),
            'secp521r1': ec.SECP521R1()
        }
        return curves[curve_name]
    
    def generate_keypair(self) -> Tuple[bytes, bytes]:
        self._private_key_obj = ec.generate_private_key(self.curve)
        self._public_key_obj = self._private_key_obj.public_key()
        
        private_bytes = self._private_key_obj.private_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        public_bytes = self._public_key_obj.public_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        self._private_key = private_bytes
        self._public_key = public_bytes
        return public_bytes, private_bytes
    
    def sign(self, message: bytes) -> bytes:
        if self._private_key_obj is None:
            if self._private_key is None:
                raise ValueError("No keypair loaded")
            self._private_key_obj = serialization.load_der_private_key(
                self._private_key, password=None
            )
        
        signature = self._private_key_obj.sign(
            message,
            ec.ECDSA(hashes.SHA256())
        )
        return signature
    
    def verify(self, message: bytes, signature: bytes) -> bool:
        if self._public_key_obj is None:
            if self._public_key is None:
                raise ValueError("No public key loaded")
            self._public_key_obj = serialization.load_der_public_key(self._public_key)
        
        try:
            self._public_key_obj.verify(signature, message, ec.ECDSA(hashes.SHA256()))
            return True
        except Exception:
            return False
    
    def get_sizes(self) -> Dict[str, int]:
        if self._public_key is None or self._private_key is None:
            self.generate_keypair()
        
        test_sig = self.sign(b"test" * 8)
        return {
            'public_key': len(self._public_key),
            'private_key': len(self._private_key),
            'signature': len(test_sig)
        }


class EdDSASignature(SignatureAlgorithm):
    """EdDSA signature implementation using cryptography library."""
    
    def __init__(self, name: str, security_level: str, curve_name: str):
        super().__init__(name, security_level)
        self.curve_name = curve_name
        self._private_key_obj = None
        self._public_key_obj = None
    
    def generate_keypair(self) -> Tuple[bytes, bytes]:
        if self.curve_name == 'ed25519':
            self._private_key_obj = ed25519.Ed25519PrivateKey.generate()
        elif self.curve_name == 'ed448':
            self._private_key_obj = ed448.Ed448PrivateKey.generate()
        else:
            raise ValueError(f"Unsupported curve: {self.curve_name}")
        
        self._public_key_obj = self._private_key_obj.public_key()
        
        private_bytes = self._private_key_obj.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption()
        )
        public_bytes = self._public_key_obj.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        )
        
        self._private_key = private_bytes
        self._public_key = public_bytes
        return public_bytes, private_bytes
    
    def sign(self, message: bytes) -> bytes:
        if self._private_key_obj is None:
            if self._private_key is None:
                raise ValueError("No keypair loaded")
            if self.curve_name == 'ed25519':
                self._private_key_obj = ed25519.Ed25519PrivateKey.from_private_bytes(
                    self._private_key
                )
            else:
                self._private_key_obj = ed448.Ed448PrivateKey.from_private_bytes(
                    self._private_key
                )
        
        return self._private_key_obj.sign(message)
    
    def verify(self, message: bytes, signature: bytes) -> bool:
        if self._public_key_obj is None:
            if self._public_key is None:
                raise ValueError("No public key loaded")
            if self.curve_name == 'ed25519':
                self._public_key_obj = ed25519.Ed25519PublicKey.from_public_bytes(
                    self._public_key
                )
            else:
                self._public_key_obj = ed448.Ed448PublicKey.from_public_bytes(
                    self._public_key
                )
        
        try:
            self._public_key_obj.verify(signature, message)
            return True
        except Exception:
            return False
    
    def get_sizes(self) -> Dict[str, int]:
        if self._public_key is None or self._private_key is None:
            self.generate_keypair()
        
        test_sig = self.sign(b"test" * 8)
        return {
            'public_key': len(self._public_key),
            'private_key': len(self._private_key),
            'signature': len(test_sig)
        }
