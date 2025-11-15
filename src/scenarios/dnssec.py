"""
DNSSEC scenario modeling.
Implements Chapter 7 of the experimental design.
"""


class DNSSECScenario:
    """
    Models DNSSEC response packet size impacts.
    
    Key characteristics:
    1. Bandwidth-constrained (not CPU-constrained)
    2. UDP 512B limit and EDNS 1232B practical limit
    3. Response includes A_Record + RRSIG + DNSKEY
    """
    
    UDP_LIMIT_BYTES = 512
    EDNS_LIMIT_BYTES = 1232
    BASE_RESPONSE_BYTES = 80  # DNS header + query + A record
    
    def __init__(self):
        pass
    
    def model_response_size(self, pk_size: int, sig_size: int) -> dict:
        """
        Calculate total DNSSEC response size.
        
        Components:
        - Base response (80B)
        - RRSIG record: signature + metadata (~20B overhead)
        - DNSKEY record: public key + metadata (~20B overhead)
        """
        rrsig_size = sig_size + 20
        dnskey_size = pk_size + 20
        total_size = self.BASE_RESPONSE_BYTES + rrsig_size + dnskey_size
        
        return {
            'base_bytes': self.BASE_RESPONSE_BYTES,
            'rrsig_bytes': rrsig_size,
            'dnskey_bytes': dnskey_size,
            'total_bytes': total_size,
            'fits_udp': total_size <= self.UDP_LIMIT_BYTES,
            'fits_edns': total_size <= self.EDNS_LIMIT_BYTES,
            'requires_tcp': total_size > self.EDNS_LIMIT_BYTES,
            'size_ratio': total_size / self.BASE_RESPONSE_BYTES
        }
