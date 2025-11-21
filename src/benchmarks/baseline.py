"""
Baseline performance benchmark module.
Implements Chapter 2 of the experimental design.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from typing import Dict, List, Tuple, Optional
import json
import csv
import time
import statistics
from datetime import datetime
from pathlib import Path

from src.algorithms.factory import AlgorithmFactory
from src.utils.timer import HighPrecisionTimer
from src.utils.stats import StatisticsCollector


def estimate_duration(algo_name: str, operation: str, config: Dict) -> Tuple[int, int, float]:
    """
    Get adaptive iteration counts and estimated duration for an algorithm operation.
    Returns: (warmup_iterations, measure_iterations, estimated_total_seconds)
    """
    baseline_config = config.get('baseline', {})
    iteration_strategy = baseline_config.get('iteration_strategy', {})
    measured_latencies = baseline_config.get('measured_latencies_ms', {})

    if algo_name in iteration_strategy and algo_name in measured_latencies:
        warmup, iterations = iteration_strategy[algo_name].get(operation, (100, 1000))
        latency_ms = measured_latencies[algo_name].get(operation, 1.0)
        estimated_sec = (warmup + iterations) * latency_ms / 1000.0
        return warmup, iterations, estimated_sec
    
    # Default conservative values
    return 100, 1000, 1.0


def format_duration(seconds: float) -> str:
    """Format duration in human-readable form without decimals."""
    if seconds < 60:
        return f"{int(seconds)}s"
    else:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        if secs == 0:
            return f"{mins}m"
        return f"{mins}m{secs}s"


def progress_bar(current: int, total: int, width: int = 40, prefix: str = "") -> str:
    """Generate a progress bar string."""
    filled = int(width * current / total)
    bar = '█' * filled + '░' * (width - filled)
    percent = current / total * 100
    return f"{prefix}[{bar}] {percent:5.1f}% ({current}/{total})"


class BaselineBenchmark:
    """
    Measures core cryptographic operations for all algorithms.
    
    Metrics:
    - Key generation latency
    - Signing latency  
    - Verification latency
    - Memory footprint
    - Size measurements (public key, private key, signature)
    """
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), '..', '..', 'config', 'benchmark.json'
            )
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.params = self.config['experiment_params']
        self.factory = AlgorithmFactory()
        
        # Create output directories
        self.raw_dir = Path('data/raw/baseline')
        self.processed_dir = Path('data/processed')
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def run_all(self) -> Dict:
        """
        Run baseline benchmarks for all configured algorithms.
        Returns comprehensive results dictionary.
        """
        print("="*80)
        print("BASELINE PERFORMANCE BENCHMARK - CHAPTER 2")
        print("="*80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        all_algos = self.factory.get_all_algorithms()
        results = {}
        
        # Calculate total estimated time for progress bar
        total_estimated = 0
        algo_estimated_times = {}
        for algo_info in all_algos:
            algo_time = 0
            for op in ['keygen', 'sign', 'verify']:
                _, _, est = estimate_duration(algo_info['name'], op, self.config)
                algo_time += est
            algo_estimated_times[algo_info['name']] = algo_time
            total_estimated += algo_time
        
        print(f"Testing {len(all_algos)} algorithms")
        print(f"Estimated total time: {format_duration(total_estimated)}")
        print(f"(Actual time may vary based on system load)")
        print("="*80)
        print()
        
        start_time = time.time()
        elapsed_time_sum = 0
        
        for i, algo_info in enumerate(all_algos, 1):
            algo_start = time.time()
            
            print(f"[{i}/{len(all_algos)}] {algo_info['name']}")
            print(f"  Security Level: {algo_info['security_level']}")
            
            try:
                algo = self.factory.create_algorithm(algo_info['type'], algo_info['name'])
                result = self._benchmark_algorithm(algo, algo_info)
                results[algo_info['name']] = result
                
                # Save raw data immediately
                self._save_raw_data(algo_info['name'], result)
                
                algo_duration = time.time() - algo_start
                elapsed_time_sum += algo_duration
                
                print(f"  ✓ Completed in {format_duration(algo_duration)}")
                print(f"    Keygen: {result['keygen_ms_mean']:.3f}±{result['keygen_ms_std']:.3f}ms "
                      f"(p99: {result['keygen_ms_p99']:.3f}ms)")
                print(f"    Sign:   {result['sign_ms_mean']:.3f}±{result['sign_ms_std']:.3f}ms "
                      f"(p99: {result['sign_ms_p99']:.3f}ms)")
                print(f"    Verify: {result['verify_ms_mean']:.3f}±{result['verify_ms_std']:.3f}ms "
                      f"(p99: {result['verify_ms_p99']:.3f}ms)")
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
                results[algo_info['name']] = {'error': str(e)}
                algo_duration = time.time() - algo_start
                elapsed_time_sum += algo_duration
            
            # Show overall progress based on time completed
            total_elapsed = time.time() - start_time
            progress_ratio = elapsed_time_sum / total_estimated if total_estimated > 0 else 0
            progress_ratio = min(progress_ratio, 1.0)  # Cap at 100%
            
            remaining_time = (total_estimated - elapsed_time_sum) if elapsed_time_sum < total_estimated else 0
            
            print(f"  Progress: {progress_bar(int(progress_ratio * 100), 100, width=30)} ({i}/{len(all_algos)})")
            print(f"  Elapsed: {format_duration(total_elapsed)} | "
                  f"Remaining: ~{format_duration(remaining_time)}")
            print()
        
        # Process and save aggregated results
        processed = self._process_results(results)
        self._save_processed_data(processed)
        
        total_time = time.time() - start_time
        
        print("="*80)
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total time: {format_duration(total_time)}")
        print(f"Raw data saved to: {self.raw_dir}")
        print(f"Processed data saved to: {self.processed_dir}")
        print("="*80)
        
        return processed
    
    def _benchmark_algorithm(self, algorithm, algo_info: Dict) -> Dict:
        """Run complete benchmark suite for a single algorithm."""
        message = os.urandom(self.params['message_size_bytes'])
        algo_name = algo_info['name']
        
        # Measure key generation with adaptive iterations
        print(f"  ⏱  Key generation...", end='', flush=True)
        warmup_kg, iter_kg, est_kg = estimate_duration(algo_name, 'keygen', self.config)
        print(f" ({iter_kg} iterations, ~{format_duration(est_kg)})")
        keygen_metrics = self.measure_operation(
            lambda: algorithm.generate_keypair(),
            warmup_kg, iter_kg, "    "
        )
        
        # Generate keys for signing/verification tests
        pub_key, priv_key = algorithm.generate_keypair()
        
        # Measure signing with adaptive iterations
        print(f"  ⏱  Signing...", end='', flush=True)
        warmup_sign, iter_sign, est_sign = estimate_duration(algo_name, 'sign', self.config)
        print(f" ({iter_sign} iterations, ~{format_duration(est_sign)})")
        sign_metrics = self.measure_operation(
            lambda: algorithm.sign(message),
            warmup_sign, iter_sign, "    "
        )
        
        # Generate signature for verification test
        signature = algorithm.sign(message)
        
        # Measure verification with adaptive iterations
        print(f"  ⏱  Verification...", end='', flush=True)
        warmup_verify, iter_verify, est_verify = estimate_duration(algo_name, 'verify', self.config)
        print(f" ({iter_verify} iterations, ~{format_duration(est_verify)})")
        verify_metrics = self.measure_operation(
            lambda: algorithm.verify(message, signature),
            warmup_verify, iter_verify, "    "
        )
        
        # Get size measurements
        sizes = algorithm.get_sizes()
        
        # Compute derived metrics
        asymmetry_ratio = verify_metrics['mean'] / sign_metrics['mean'] if sign_metrics['mean'] > 0 else 0
        sign_ops_per_sec = 1000.0 / sign_metrics['mean'] if sign_metrics['mean'] > 0 else 0
        verify_ops_per_sec = 1000.0 / verify_metrics['mean'] if verify_metrics['mean'] > 0 else 0
        
        return {
            'algorithm': algo_info['name'],
            'security_level': algo_info['security_level'],
            'iterations': {
                'keygen': iter_kg,
                'sign': iter_sign,
                'verify': iter_verify
            },
            
            # Key generation metrics
            'keygen_ms_mean': keygen_metrics['mean'],
            'keygen_ms_std': keygen_metrics['std'],
            'keygen_ms_median': keygen_metrics['median'],
            'keygen_ms_p95': keygen_metrics['p95'],
            'keygen_ms_p99': keygen_metrics['p99'],
            'keygen_samples': keygen_metrics['samples'],
            
            # Signing metrics
            'sign_ms_mean': sign_metrics['mean'],
            'sign_ms_std': sign_metrics['std'],
            'sign_ms_median': sign_metrics['median'],
            'sign_ms_p95': sign_metrics['p95'],
            'sign_ms_p99': sign_metrics['p99'],
            'sign_samples': sign_metrics['samples'],
            
            # Verification metrics
            'verify_ms_mean': verify_metrics['mean'],
            'verify_ms_std': verify_metrics['std'],
            'verify_ms_median': verify_metrics['median'],
            'verify_ms_p95': verify_metrics['p95'],
            'verify_ms_p99': verify_metrics['p99'],
            'verify_samples': verify_metrics['samples'],
            
            # Size metrics
            'pk_bytes': sizes['public_key'],
            'sk_bytes': sizes['private_key'],
            'sig_bytes': sizes['signature'],
            
            # Derived metrics
            'asymmetry_ratio': asymmetry_ratio,
            'sign_ops_per_sec': sign_ops_per_sec,
            'verify_ops_per_sec': verify_ops_per_sec,
            
            # Metadata
            'timestamp': datetime.now().isoformat(),
            'message_size_bytes': self.params['message_size_bytes']
        }
    
    def measure_operation(self, operation, warmup: int, iterations: int, indent: str = "") -> Dict:
        """
        Measure operation performance with warmup and progress tracking.
        """
        # Warmup phase
        if warmup > 0:
            print(f"{indent}Warmup ({warmup} iterations)...", end='', flush=True)
            for _ in range(warmup):
                operation()
            print(" ✓")
        
        # Measurement phase with progress tracking
        samples = []
        print(f"{indent}Measuring:", end='', flush=True)
        
        # Update interval: show progress every 10% or every 100 iterations, whichever is larger
        update_interval = max(iterations // 10, 100) if iterations > 100 else 1
        
        for i in range(iterations):
            start = time.perf_counter()
            result = operation()
            elapsed_ms = (time.perf_counter() - start) * 1000
            samples.append(elapsed_ms)

            # Guard against operations that unexpectedly fail (e.g., verify returns False)
            if isinstance(result, bool) and not result:
                raise RuntimeError("Operation returned False during measurement")
            
            # Show progress
            if (i + 1) % update_interval == 0 or i == iterations - 1:
                print(f"\r{indent}Measuring: {progress_bar(i+1, iterations, width=20, prefix='')}", 
                      end='', flush=True)
        
        print(" ✓")
        return StatisticsCollector.compute_stats(samples)
    
    def _save_raw_data(self, algo_name: str, result: Dict):
        """Save raw per-iteration samples to CSV."""
        if 'error' in result:
            return
        
        csv_path = self.raw_dir / f"{algo_name}.csv"
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['operation', 'sample_ms'])
            
            # Write keygen samples
            for sample in result.get('keygen_samples', []):
                writer.writerow(['keygen', f"{sample:.6f}"])
            
            # Write sign samples
            for sample in result.get('sign_samples', []):
                writer.writerow(['sign', f"{sample:.6f}"])
            
            # Write verify samples
            for sample in result.get('verify_samples', []):
                writer.writerow(['verify', f"{sample:.6f}"])
    
    def _process_results(self, results: Dict) -> Dict:
        """Aggregate results into summary tables."""
        processed = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'platform': self.config['platform'],
                'experiment_params': self.params
            },
            'algorithms': {}
        }
        
        for algo_name, result in results.items():
            if 'error' not in result:
                # Remove raw samples from processed output
                processed_result = {k: v for k, v in result.items() 
                                   if not k.endswith('_samples')}
                processed['algorithms'][algo_name] = processed_result
        
        return processed
    
    def _save_processed_data(self, processed: Dict):
        """Save aggregated results to JSON."""
        json_path = self.processed_dir / 'baseline_summary.json'
        
        with open(json_path, 'w') as f:
            json.dump(processed, f, indent=2)
        
        # Also create CSV summary for easy analysis
        self._create_summary_csv(processed)
    
    def _create_summary_csv(self, processed: Dict):
        """Create summary CSV table."""
        csv_path = self.processed_dir / 'baseline_summary.csv'
        
        with open(csv_path, 'w', newline='') as f:
            fieldnames = [
                'algorithm', 'security_level',
                'keygen_ms_mean', 'keygen_ms_std', 'keygen_ms_p99',
                'sign_ms_mean', 'sign_ms_std', 'sign_ms_p99',
                'verify_ms_mean', 'verify_ms_std', 'verify_ms_p99',
                'pk_bytes', 'sk_bytes', 'sig_bytes',
                'asymmetry_ratio', 'sign_ops_per_sec', 'verify_ops_per_sec'
            ]
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for algo_data in processed['algorithms'].values():
                row = {k: algo_data.get(k, '') for k in fieldnames}
                writer.writerow(row)
        
        print(f"\n✓ Summary CSV created: {csv_path}")


def main():
    """Command-line entry point for baseline benchmarks."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run baseline performance benchmarks')
    parser.add_argument('--config', help='Path to benchmark config file')
    parser.add_argument('--output', help='Output directory for results')
    
    args = parser.parse_args()
    
    benchmark = BaselineBenchmark(config_path=args.config)
    results = benchmark.run_all()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total algorithms benchmarked: {len(results['algorithms'])}")
    print(f"Results saved to: {benchmark.processed_dir / 'baseline_summary.json'}")
    print("="*80)


if __name__ == '__main__':
    main()
