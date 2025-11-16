# Running Baseline Benchmarks (Chapter 2)

## Quick Start

```bash
# Activate virtual environment
source venv/bin/activate

# Run complete baseline benchmarks
python src/benchmarks/baseline.py
```

## Key Features

1. **Adaptive Iteration Strategy**: Different algorithms use different iteration counts based on their measured performance characteristics
   - Fast operations (e.g., Ed25519 verify): 50,000 iterations
   - Medium operations (e.g., RSA-2048 keygen): 100 iterations
   - Slow operations (e.g., RSA-15360 keygen): 10 iterations
   - Based on actual measurements from preliminary tests

2. **Accurate Time Estimation**: Uses measured latencies from `test_algorithms.py`
   - Pre-calculated estimates for each algorithm and operation
   - Time displayed as integers without decimals (e.g., "14s" not "14.8s")
   - Automatic unit conversion: seconds for <60s, minutes for ≥60s (e.g., "1m23s")

3. **Time-Based Progress Tracking**: 
   - Progress bar reflects **time consumed** rather than algorithm count
   - Fast algorithms (Ed25519) show small progress increments (~0.1% per algorithm)
   - Slow algorithms (RSA-15360, SPHINCS+-s) show large progress jumps (~50% when RSA-15360 completes)
   - Remaining time estimates based on actual elapsed time vs. total estimated time
   - Example: After completing 1/17 algorithms, if it was Ed25519 (0.5s), progress shows 0.1%; if it was RSA-15360 (460s), progress shows 51%

4. **Comprehensive Metrics**: For each operation (keygen, sign, verify):
   - Mean latency
   - Standard deviation
   - Percentiles (p50, p95, p99)
   - Key/signature sizes
   - Sample count

5. **Dual Output Format**:
   - **Raw data** (`data/raw/baseline/`): Individual CSV files per algorithm with all samples
   - **Processed data** (`data/processed/`): Aggregated summary in both JSON and CSV formats

## Output Format Example

```
================================================================================
BASELINE PERFORMANCE BENCHMARK - CHAPTER 2
================================================================================
Started at: 2025-11-13 19:45:00

Testing 17 algorithms
Estimated total time: 10m30s
(Actual time may vary based on system load)
================================================================================

[1/17] RSA-2048
  Security Level: 112
  Keygen [████████████████████████████████] (100/100) ~8s
  Sign   [████████████████████████████████] (1000/1000) ~6s
  Verify [████████████████████████████████] (10000/10000) ~1s
  ✓ Completed in 15s
    Keygen: 75.186±5.234ms (p99: 85.432ms)
    Sign:   6.492±0.543ms (p99: 7.876ms)
    Verify: 0.102±0.012ms (p99: 0.125ms)
  Progress: [█░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] (1/17)
  Elapsed: 15s | Remaining: ~10m15s

[2/17] RSA-3072
  Security Level: 128
  Keygen [████████████████████████████████] (50/50) ~9s
  Sign   [████████████████████████████████] (500/500) ~3s
  Verify [████████████████████████████████] (10000/10000) ~1s
  ✓ Completed in 13s
    Keygen: 173.277±8.765ms (p99: 190.123ms)
    Sign:   6.894±0.678ms (p99: 8.234ms)
    Verify: 0.102±0.015ms (p99: 0.130ms)
  Progress: [██░░░░░░░░░░░░░░░░░░░░░░░░░░░░] (2/17)
  Elapsed: 28s | Remaining: ~10m2s

...

[10/17] RSA-15360
  Security Level: 256
  Keygen [████████████████████████████████] (10/10) ~7m37s
  Sign   [████████████████████████████████] (20/20) ~1s
  Verify [████████████████████████████████] (1000/1000) ~0s
  ✓ Completed in 7m38s
    Keygen: 45707.856±1234.567ms (p99: 48234.123ms)
    Sign:   32.085±2.345ms (p99: 36.789ms)
    Verify: 0.448±0.034ms (p99: 0.512ms)
  Progress: [███████████████░░░░░░░░░░░░░░░] (10/17)
  Elapsed: 8m15s | Remaining: ~2m15s

...

================================================================================
Completed at: 2025-11-13 19:55:30
Total time: 10m30s
Raw data saved to: data/raw/baseline
Processed data saved to: data/processed
================================================================================
```

**Note on Progress Bar**: The progress bar shows **time-based progress**, not algorithm count. Notice that after completing 10/17 algorithms (58.8%), the progress bar shows ~50% because RSA-15360 consumed most of the time.

## Expected Runtime

The complete benchmark takes approximately **10-15 minutes** on a modern system:

- Classical algorithms (RSA, ECDSA, EdDSA): ~5-7 minutes
- Post-quantum algorithms (Dilithium, SPHINCS+): ~5-8 minutes

**Slowest operations** (consume most of total runtime):
- **RSA-15360 key generation**: ~45.7 seconds per iteration × 10 iterations = ~7m37s
- **RSA-7680 key generation**: ~5.4 seconds per iteration × 20 iterations = ~1m48s
- **SPHINCS+-192s signing**: ~381ms per iteration × 50 iterations = ~19s
- **SPHINCS+-256s signing**: ~816ms per iteration × 20 iterations = ~16s

**Fastest operations** (negligible runtime):
- **Ed25519/Ed448 verification**: ~0.14-0.17ms per iteration
- **ECDSA verification**: ~0.2-0.3ms per iteration
- These run 5000-50000 iterations but complete in <1 second total

The time-based progress bar will show large jumps when RSA-15360 and RSA-7680 complete.

## Output Files

### Raw Data (`data/raw/baseline/`)

Individual CSV files for each algorithm:

```
data/raw/baseline/
├── RSA-3072.csv
├── ECDSA-P256.csv
├── Ed25519.csv
├── Dilithium2.csv
└── ...
```

Each CSV contains per-iteration samples:
```csv
operation,sample_ms
keygen,173.234567
keygen,174.123456
sign,1.765432
sign,1.771234
verify,0.040123
verify,0.039987
```

### Processed Data (`data/processed/`)

**JSON format** (`baseline_summary.json`):
```json
{
  "metadata": {
    "timestamp": "2025-11-13T19:42:08",
    "platform": { "cpu": "Apple M4", ... }
  },
  "algorithms": {
    "RSA-3072": {
      "keygen_ms_mean": 173.277,
      "keygen_ms_std": 5.234,
      "keygen_ms_p99": 185.123,
      "sign_ms_mean": 1.771,
      ...
    }
  }
}
```

**CSV format** (`baseline_summary.csv`):
```csv
algorithm,security_level,keygen_ms_mean,keygen_ms_std,sign_ms_mean,...
RSA-3072,level_1,173.277,5.234,1.771,0.123,...
ECDSA-P256,level_1,2.395,0.089,5.959,0.234,...
```

## Metrics Collected

For each algorithm and operation:

- **Latency statistics**:
  - Mean (average time)
  - Standard deviation (variability)
  - Median (50th percentile)
  - P95 (95th percentile - SLA threshold)
  - P99 (99th percentile - tail latency)

- **Size measurements**:
  - Public key bytes
  - Private key bytes
  - Signature bytes

- **Derived metrics**:
  - Asymmetry ratio (verify/sign time ratio)
  - Operations per second (throughput)

## Use Cases for Output Data

### 1. Academic Analysis
```python
import json
with open('data/processed/baseline_summary.json') as f:
    data = json.load(f)

# Compare signing latency across security levels
for algo, metrics in data['algorithms'].items():
    print(f"{algo}: {metrics['sign_ms_mean']:.3f}ms")
```

### 2. Performance Analysis
The processed data enables:
- Cross-algorithm latency and throughput comparisons
- Classical vs. PQC performance trade-off analysis
- Security level impact evaluation
- Key/signature size vs. performance correlation

### 3. Visualization
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/processed/baseline_summary.csv')
df.plot(x='algorithm', y='sign_ms_mean', kind='bar')
plt.savefig('results/baseline_signing_latency.png')
```

## Interrupting and Resuming

- **Ctrl+C**: Safely interrupt at any time
- Currently tested algorithms are saved to `data/raw/baseline/`
- Re-run to test remaining algorithms (skips completed ones - *planned feature*)

## Customization

### Adjusting Iteration Counts

Edit iteration counts in `src/benchmarks/baseline.py`:

```python
ITERATION_STRATEGY = {
    'RSA-3072': {
        'keygen': (10, 50),      # (warmup_rounds, measurement_iterations)
        'sign': (100, 500),
        'verify': (1000, 10000)
    },
    # Add or modify algorithms...
}
```

### Understanding Time Estimation

The benchmark pre-calculates total estimated time using measured latencies:

```python
# For each algorithm and operation:
total_estimated = Σ [(warmup + iterations) × measured_latency]
```

Where `measured_latency` values are stored in the `MEASURED_LATENCIES` dictionary (extracted from actual `test_algorithms.py` runs).

**Progress calculation**:
```python
progress_ratio = elapsed_time_sum / total_estimated
```

This ensures the progress bar advances proportionally to **time consumed**, not algorithm count.

### Warmup Rounds

Warmup rounds are executed before measurements to:
- Initialize JIT compilation (if applicable)
- Warm up CPU caches
- Establish stable performance baseline

Warmup iteration counts (per algorithm type):
- Fast operations (Ed25519, ECDSA verify): 5000 warmup rounds
- Medium operations (RSA-2048/3072 keygen): 10-50 warmup rounds
- Slow operations (RSA-15360 keygen): 2 warmup rounds

## Troubleshooting

### Issue: "Module not found"
```bash
# Ensure you're in the project root
cd /path/to/P5003

# Activate environment
source venv/bin/activate
```

### Issue: Memory warnings on large iteration counts
Reduce iterations for memory-constrained systems by editing `ITERATION_STRATEGY`.

### Issue: Thermal throttling on MacBook
Close other applications and ensure good ventilation. The script logs when performance cores are active.

---

**Last Updated**: 2025-11-13
