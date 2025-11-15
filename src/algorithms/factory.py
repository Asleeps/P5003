import json
import os
from typing import List, Dict, Any
from .classical import RSASignature, ECDSASignature, EdDSASignature
from .post_quantum import DilithiumSignature, SPHINCSPlusSignature


class AlgorithmFactory:
    """Factory for creating signature algorithm instances from configuration."""
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), '..', '..', 'config', 'algorithms.json'
            )
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
    
    def create_algorithm(self, algo_type: str, variant_name: str):
        """Create a signature algorithm instance by type and variant name."""
        
        if algo_type == 'RSA':
            variant = self._find_variant('classical', 'RSA', variant_name)
            return RSASignature(
                name=variant['name'],
                security_level=variant['security_level'],
                key_size=variant['key_size']
            )
        
        elif algo_type == 'ECDSA':
            variant = self._find_variant('classical', 'ECDSA', variant_name)
            return ECDSASignature(
                name=variant['name'],
                security_level=variant['security_level'],
                curve_name=variant['curve']
            )
        
        elif algo_type == 'EdDSA':
            variant = self._find_variant('classical', 'EdDSA', variant_name)
            return EdDSASignature(
                name=variant['name'],
                security_level=variant['security_level'],
                curve_name=variant['curve']
            )
        
        elif algo_type == 'Dilithium':
            variant = self._find_variant('post_quantum', 'Dilithium', variant_name)
            return DilithiumSignature(
                name=variant['name'],
                security_level=variant['security_level'],
                liboqs_name=variant['liboqs_name']
            )
        
        elif algo_type == 'SPHINCS+':
            variant = self._find_variant('post_quantum', 'SPHINCS+', variant_name)
            return SPHINCSPlusSignature(
                name=variant['name'],
                security_level=variant['security_level'],
                liboqs_name=variant['liboqs_name']
            )
        
        else:
            raise ValueError(f"Unknown algorithm type: {algo_type}")
    
    def _find_variant(self, category: str, algo_type: str, variant_name: str) -> Dict:
        """Find variant configuration by name."""
        variants = self.config[category][algo_type]['variants']
        for variant in variants:
            if variant['name'] == variant_name:
                return variant
        raise ValueError(f"Variant {variant_name} not found for {algo_type}")
    
    def get_all_algorithms(self) -> List[Dict[str, Any]]:
        """Get list of all configured algorithms with metadata."""
        algorithms = []
        
        for category in ['classical', 'post_quantum']:
            for algo_type, algo_data in self.config[category].items():
                for variant in algo_data['variants']:
                    algorithms.append({
                        'category': category,
                        'type': algo_type,
                        'name': variant['name'],
                        'security_level': variant['security_level']
                    })
        
        return algorithms
    
    def get_algorithms_by_level(self, level: str) -> List[Dict[str, Any]]:
        """Get all algorithms at a specific security level."""
        all_algos = self.get_all_algorithms()
        return [a for a in all_algos if a['security_level'] == level]
