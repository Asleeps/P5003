#!/usr/bin/env python3
"""
Test script for digital signature algorithms.
Tests all configured algorithms and generates a comprehensive report.
"""

import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from algorithms.factory import AlgorithmFactory
from algorithms.base import SignatureAlgorithm
from typing import Dict, List, Tuple
import traceback


def test_algorithm(algo: SignatureAlgorithm, message: bytes = b"test_message" * 4) -> Dict:
    """Test a single algorithm and return results with timing information."""
    result = {
        'name': algo.name,
        'type': algo.__class__.__name__,
        'security_level': algo.security_level,
        'status': 'unknown',
        'error': None,
        'sizes': {},
        'operations': {},
        'timings': {}
    }
    
    try:
        # Test 1: Key generation
        print(f"  Testing keygen...", end=' ')
        start_time = time.perf_counter()
        pub, priv = algo.generate_keypair()
        keygen_time = time.perf_counter() - start_time
        result['operations']['keygen'] = 'OK'
        result['timings']['keygen_ms'] = keygen_time * 1000
        print(f"✓ ({keygen_time*1000:.3f} ms)")
        
        # Test 2: Signing
        print(f"  Testing sign...", end=' ')
        start_time = time.perf_counter()
        signature = algo.sign(message)
        sign_time = time.perf_counter() - start_time
        result['operations']['sign'] = 'OK'
        result['timings']['sign_ms'] = sign_time * 1000
        print(f"✓ ({sign_time*1000:.3f} ms)")
        
        # Test 3: Verification
        print(f"  Testing verify...", end=' ')
        start_time = time.perf_counter()
        is_valid = algo.verify(message, signature)
        verify_time = time.perf_counter() - start_time
        result['timings']['verify_ms'] = verify_time * 1000
        if not is_valid:
            result['operations']['verify'] = 'FAILED (invalid signature)'
            result['status'] = 'failed'
            print("✗")
            return result
        result['operations']['verify'] = 'OK'
        print(f"✓ ({verify_time*1000:.3f} ms)")
        
        # Test 4: Invalid signature rejection
        print(f"  Testing invalid signature...", end=' ')
        bad_sig = signature[:len(signature)//2] + bytes([0] * (len(signature) - len(signature)//2))
        start_time = time.perf_counter()
        is_invalid = algo.verify(message, bad_sig)
        verify_invalid_time = time.perf_counter() - start_time
        result['timings']['verify_invalid_ms'] = verify_invalid_time * 1000
        if is_invalid:
            result['operations']['verify_invalid'] = 'FAILED (accepted bad signature)'
            result['status'] = 'failed'
            print("✗")
            return result
        result['operations']['verify_invalid'] = 'OK'
        print(f"✓ ({verify_invalid_time*1000:.3f} ms)")
        
        # Test 5: Size measurements
        print(f"  Measuring sizes...", end=' ')
        sizes = algo.get_sizes()
        result['sizes'] = sizes
        print("✓")
        
        result['status'] = 'success'
        
    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)
        print(f"✗ {e}")
        if '--verbose' in sys.argv:
            traceback.print_exc()
    
    return result


def print_summary_table(results: List[Dict]):
    """Print formatted summary table with timing information."""
    
    # Map security levels to bit strength
    level_to_bits = {
        'level_1': '128-bit',
        'level_3': '192-bit',
        'level_5': '256-bit'
    }
    
    print("\n" + "="*150)
    print("ALGORITHM SUMMARY TABLE")
    print("="*150)
    
    header = f"{'Algorithm':<32} {'Security':<10} {'S':<3} {'Keygen(ms)':<12} {'Sign(ms)':<12} {'Verify(ms)':<12} {'PubKey(B)':<11} {'PrivKey(B)':<11} {'Sig(B)':<10}"
    print(header)
    print("-"*150)
    
    for r in results:
        status_symbol = "✓" if r['status'] == 'success' else "✗"
        sizes = r.get('sizes', {})
        timings = r.get('timings', {})
        
        security_bits = level_to_bits.get(r['security_level'], r['security_level'])
        
        keygen_ms = f"{timings.get('keygen_ms', 0):.3f}" if timings.get('keygen_ms') is not None else "N/A"
        sign_ms = f"{timings.get('sign_ms', 0):.3f}" if timings.get('sign_ms') is not None else "N/A"
        verify_ms = f"{timings.get('verify_ms', 0):.3f}" if timings.get('verify_ms') is not None else "N/A"
        
        pub_size = sizes.get('public_key', 0) if sizes else 0
        priv_size = sizes.get('private_key', 0) if sizes else 0
        sig_size = sizes.get('signature', 0) if sizes else 0
        
        row = f"{r['name']:<32} {security_bits:<10} {status_symbol:<3} {keygen_ms:<12} {sign_ms:<12} {verify_ms:<12} {pub_size:<11} {priv_size:<11} {sig_size:<10}"
        print(row)
    
    print("="*150)


def print_statistics(results: List[Dict]):
    """Print success/failure statistics."""
    total = len(results)
    success = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] == 'failed')
    error = sum(1 for r in results if r['status'] == 'error')
    
    print("\n" + "="*100)
    print("TEST STATISTICS")
    print("="*100)
    print(f"Total algorithms tested: {total}")
    print(f"Successful: {success} ({success/total*100:.1f}%)")
    print(f"Failed: {failed} ({failed/total*100:.1f}%)")
    print(f"Errors: {error} ({error/total*100:.1f}%)")
    print("="*100)
    
    if failed > 0 or error > 0:
        print("\nFailed/Error algorithms:")
        for r in results:
            if r['status'] in ['failed', 'error']:
                print(f"  - {r['name']}: {r.get('error', 'verification failed')}")


def print_security_level_groups(results: List[Dict]):
    """Print algorithms grouped by security level with detailed comparison."""
    
    # Map security levels to bit strength
    level_to_bits = {
        'level_1': '128-bit',
        'level_3': '192-bit',
        'level_5': '256-bit'
    }
    
    level_names = {
        'level_1': 'NIST Security Level 1 (AES-128 equivalent)',
        'level_3': 'NIST Security Level 3 (AES-192 equivalent)',
        'level_5': 'NIST Security Level 5 (AES-256 equivalent)'
    }
    
    levels = {}
    for r in results:
        level = r['security_level']
        if level not in levels:
            levels[level] = []
        levels[level].append(r)
    
    print("\n" + "="*150)
    print("ALGORITHMS BY SECURITY LEVEL")
    print("="*150)
    
    for level in sorted(levels.keys()):
        print(f"\n{'='*150}")
        print(f"{level_names.get(level, level)}")
        print(f"{'='*150}")
        
        header = f"{'Algorithm':<32} {'S':<3} {'Keygen(ms)':<12} {'Sign(ms)':<12} {'Verify(ms)':<12} {'PubKey(B)':<11} {'PrivKey(B)':<11} {'Sig(B)':<10}"
        print(header)
        print("-"*150)
        
        for r in levels[level]:
            status = "✓" if r['status'] == 'success' else "✗"
            sizes = r.get('sizes', {})
            timings = r.get('timings', {})
            
            keygen_ms = f"{timings.get('keygen_ms', 0):.3f}" if timings.get('keygen_ms') is not None else "N/A"
            sign_ms = f"{timings.get('sign_ms', 0):.3f}" if timings.get('sign_ms') is not None else "N/A"
            verify_ms = f"{timings.get('verify_ms', 0):.3f}" if timings.get('verify_ms') is not None else "N/A"
            
            pub_size = sizes.get('public_key', 0) if sizes else 0
            priv_size = sizes.get('private_key', 0) if sizes else 0
            sig_size = sizes.get('signature', 0) if sizes else 0
            
            row = f"{r['name']:<32} {status:<3} {keygen_ms:<12} {sign_ms:<12} {verify_ms:<12} {pub_size:<11} {priv_size:<11} {sig_size:<10}"
            print(row)


def print_performance_comparison(results: List[Dict]):
    """Print performance comparison highlighting fastest/slowest algorithms."""
    successful = [r for r in results if r['status'] == 'success' and r.get('timings')]
    
    if not successful:
        return
    
    print("\n" + "="*100)
    print("PERFORMANCE HIGHLIGHTS")
    print("="*100)
    
    # Signing performance
    by_sign = sorted(successful, key=lambda x: x['timings'].get('sign_ms', float('inf')))
    print("\n📝 Signing Speed (fastest to slowest):")
    print(f"  🥇 Fastest: {by_sign[0]['name']:<30} {by_sign[0]['timings']['sign_ms']:>10.3f} ms")
    if len(by_sign) > 1:
        print(f"  🥈 Second:  {by_sign[1]['name']:<30} {by_sign[1]['timings']['sign_ms']:>10.3f} ms")
    if len(by_sign) > 2:
        print(f"  🥉 Third:   {by_sign[2]['name']:<30} {by_sign[2]['timings']['sign_ms']:>10.3f} ms")
    print(f"  🐌 Slowest: {by_sign[-1]['name']:<30} {by_sign[-1]['timings']['sign_ms']:>10.3f} ms")
    slowdown = by_sign[-1]['timings']['sign_ms'] / by_sign[0]['timings']['sign_ms']
    print(f"     └─ {slowdown:.1f}x slower than fastest")
    
    # Verification performance
    by_verify = sorted(successful, key=lambda x: x['timings'].get('verify_ms', float('inf')))
    print("\n✅ Verification Speed (fastest to slowest):")
    print(f"  🥇 Fastest: {by_verify[0]['name']:<30} {by_verify[0]['timings']['verify_ms']:>10.3f} ms")
    if len(by_verify) > 1:
        print(f"  🥈 Second:  {by_verify[1]['name']:<30} {by_verify[1]['timings']['verify_ms']:>10.3f} ms")
    if len(by_verify) > 2:
        print(f"  🥉 Third:   {by_verify[2]['name']:<30} {by_verify[2]['timings']['verify_ms']:>10.3f} ms")
    print(f"  🐌 Slowest: {by_verify[-1]['name']:<30} {by_verify[-1]['timings']['verify_ms']:>10.3f} ms")
    slowdown = by_verify[-1]['timings']['verify_ms'] / by_verify[0]['timings']['verify_ms']
    print(f"     └─ {slowdown:.1f}x slower than fastest")
    
    # Signature size
    by_size = sorted(successful, key=lambda x: x['sizes'].get('signature', float('inf')))
    print("\n📦 Signature Size (smallest to largest):")
    print(f"  🥇 Smallest: {by_size[0]['name']:<30} {by_size[0]['sizes']['signature']:>10} bytes")
    if len(by_size) > 1:
        print(f"  🥈 Second:   {by_size[1]['name']:<30} {by_size[1]['sizes']['signature']:>10} bytes")
    print(f"  📦 Largest:  {by_size[-1]['name']:<30} {by_size[-1]['sizes']['signature']:>10} bytes")
    expansion = by_size[-1]['sizes']['signature'] / by_size[0]['sizes']['signature']
    print(f"     └─ {expansion:.1f}x larger than smallest")


def main():
    print("="*100)
    print("DIGITAL SIGNATURE ALGORITHM TEST SUITE")
    print("="*100)
    print()
    
    try:
        factory = AlgorithmFactory()
        print(f"Loaded configuration successfully.\n")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return 1
    
    all_algos = factory.get_all_algorithms()
    print(f"Found {len(all_algos)} algorithms to test.\n")
    
    results = []
    
    for i, algo_info in enumerate(all_algos, 1):
        print(f"[{i}/{len(all_algos)}] Testing {algo_info['name']} ({algo_info['type']})...")
        
        try:
            algo = factory.create_algorithm(algo_info['type'], algo_info['name'])
            result = test_algorithm(algo)
            results.append(result)
        except Exception as e:
            print(f"  ✗ Failed to create algorithm: {e}")
            results.append({
                'name': algo_info['name'],
                'type': algo_info['type'],
                'security_level': algo_info['security_level'],
                'status': 'error',
                'error': f"Creation failed: {e}",
                'sizes': {},
                'operations': {}
            })
            if '--verbose' in sys.argv:
                traceback.print_exc()
        
        print()
    
    print_summary_table(results)
    print_security_level_groups(results)
    print_performance_comparison(results)
    print_statistics(results)
    
    success_count = sum(1 for r in results if r['status'] == 'success')
    if success_count == len(results):
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {len(results) - success_count} tests failed or encountered errors.")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
