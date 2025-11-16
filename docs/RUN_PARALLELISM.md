# Running Parallelism Benchmarks (Chapter 3)

## Quick Start

```bash
# 1. Activate the virtual environment and install deps
source venv/bin/activate
pip install -r requirements.txt

# 2. (Optional) Disable powermetrics if you cannot run with sudo
#    jq '.parallelism.powermetrics.enabled=false' config/benchmark.json | sponge config/benchmark.json

# 3a. Run the threading benchmark (GIL-releasing implementations for all algorithms)
sudo -E python -m src.benchmarks.parallelism_threading

# 3b. Run the multiprocessing benchmark (process-per-worker, fully GIL-free)
sudo -E python -m src.benchmarks.parallelism_multiprocess
```

> 💡 **GIL-Releasing Implementations**: All algorithms use GIL-releasing implementations:
> - **Classical (RSA/ECDSA/EdDSA)**: Direct OpenSSL bindings via CFFI (confirmed 85%+ efficiency at 2-4 threads)
> - **PQC (Dilithium/SPHINCS+)**: liboqs native implementations
>
> ⚠️ **Performance Note**: Fast algorithms (ECDSA-P256, Ed25519) may show performance degradation at 8+ threads due to cache contention from algorithm-specific optimizations (precomputed tables), not GIL issues. This reflects real-world multi-core performance characteristics.

> 💡 If you cannot provide sudo credentials, disable `powermetrics.enabled` and `cpu_affinity.enabled` in `config/benchmark.json` and run without `sudo`.

## Overview

Chapter 3 provides two execution models:

1. **Threading mode** (`parallelism_threading.py`): Python threads with GIL-releasing implementations (CFFI for classical, liboqs for PQC)
2. **Multiprocessing mode** (`parallelism_multiprocess.py`): Independent OS processes with no GIL contention

Both modes test scalability across `[1, 2, 4, 6, 8, 10]` workers to understand:

1. **Inter-operation parallelism**: How well independent signing/verification operations scale (performance cores first, then efficiency cores)  
2. **GIL impact**: Threading vs multiprocessing comparison reveals GIL overhead and cache contention patterns
3. **Intra-operation insights**: SPHINCS+ runs expose per-signature latency distributions for later pipeline optimizations  
4. **Scheduler and power coupling**: We log CPU utilization, RSS, temperature, and energy/operation so Chapters 4–5 can reason about both capacity and efficiency

## Key Features

1. **Multiprocessing workers** (spawn) → no GIL bottleneck; each worker loads pre-generated keys and runs an independent loop pinned to a core when enabled.  
2. **Extended worker set** `[1, 2, 4, 6, 8, 10]` aligned with the Apple M4 topology (6 performance + 4 efficiency).  
3. **Deterministic timing**: 1 s warmup, 12 s measurement window, middle 10 s used for statistics to avoid ramp-up/down bias.  
4. **Rich telemetry**:  
   - Per-worker CSV samples (ops counters, RSS, timestamps).  
   - System JSON snapshots (per-core CPU%, powermetrics logs, temperature).  
   - Latency stats (p50/p95/p99), scaling efficiency, CPU utilization, memory footprint, and energy/operation.
5. **Key caching**: `data/cache/` stores serialized keypairs/signatures so repeated runs skip expensive key generation.  
6. **Graceful fallbacks**: Powermetrics and affinity degrade cleanly if unsupported; warnings are recorded once per run.

## Expected Runtime

Plan for **35–45 minutes** on the Apple M4 reference host:

- **Per algorithm**: 12 configurations (6 worker counts × 2 operations) × 12 seconds + warmup/coordination overhead ≈ 3 minutes.  
- **Full suite (≈17 algorithms)**: 17 × 3 minutes ≈ 45–50 minutes depending on powermetrics overhead.  
- **Slowest configs**: RSA-15360 signing (ops/sec < 0.05) and SPHINCS+-256 variants due to large signatures.  
- **Fastest configs**: Ed25519/ECDSA verification where the processes remain throughput-bound even at 10 workers.

## Output Format Example

```
================================================================================
PARALLELISM THROUGHPUT BENCHMARK - MULTIPROCESS MODE
================================================================================
Started at: 2025-11-20 09:00:00

Testing 17 algorithms
Worker counts: [1, 2, 4, 6, 8, 10]
Duration per config: 12s (analysis window: 10s, warmup: 1s)
Estimated total time: ~45m
================================================================================

[1/17] RSA-2048 (Level 112)
  Sign:
     1t →      124.3 ops/sec | p99: 8.01 ms
     2t →      247.8 ops/sec | p99: 8.22 ms
     4t →      490.6 ops/sec | p99: 8.41 ms
     6t →      731.0 ops/sec | p99: 8.95 ms
     8t →      954.2 ops/sec | p99: 9.32 ms
    10t →     1152.9 ops/sec | p99: 9.88 ms
  Verify:
     1t →     9860.5 ops/sec | p99: 0.11 ms
     2t →    19512.2 ops/sec | p99: 0.12 ms
     4t →    38221.3 ops/sec | p99: 0.13 ms
     6t →    56532.8 ops/sec | p99: 0.15 ms
     8t →    73610.4 ops/sec | p99: 0.17 ms
    10t →    88412.6 ops/sec | p99: 0.19 ms
  ✓ Completed in 3m02s
    Sign  :   124.3 →   1152.9 ops/sec (efficiency: 92.8%)
    Verify:  9860.5 →  88412.6 ops/sec (efficiency: 89.7%)
    Avg CPU (system): 94.2% | Avg Temp: 66.1 °C | Energy/op: 0.34 mJ

[2/17] ECDSA-P256 (Level 128)
  ...
```

**What to expect**:

1. Each worker count is printed once the configuration finishes; values already reflect the 10-second analysis window.  
2. If powermetrics is enabled, the summary line includes temperature and energy/operation; otherwise those fields show `N/A`.  
3. Affinity diagnostics display once per run if macOS cannot honor strict pinning (informational only).  
4. Progress shows elapsed vs estimated time so you can plan multi-algorithm sessions.

## Output Files

### Raw Data (`data/raw/parallelism/`)

For every `(algorithm, operation, worker_count)` we emit:

```
RSA-2048_sign_t1.csv
RSA-2048_sign_t1_system.json
RSA-2048_sign_t2.csv
...
```

`*.csv` columns:

| Column | Meaning |
| --- | --- |
| `timestamp` | wall-clock (`time.time()`) when the worker snapshot was recorded |
| `worker_id` | worker/process index |
| `ops_total` | cumulative operations executed since the start event |
| `analysis_ops` | operations that fell inside the 10 s analysis window |
| `rss_bytes` | worker RSS at sample time (requires `psutil`) |
| `system_cpu_percent` | average CPU percent across all cores (from psutil sampler) |
| `cpu_temperature_c` | average CPU die temperature across powermetrics samples |

`*_system.json` combines system-level telemetry:

```json
{
  "cpu_samples": [
    {"timestamp": 1699872001.123, "percpu": [95.1, 94.8, ...], "avg": 93.9},
    ...
  ],
  "power": {
    "energy_joules": 12.3,
    "avg_power_w": 24.6,
    "avg_temperature_c": 66.1,
    "log_path": "results/power/powermetrics_RSA-2048_sign_t4_20251120-090025.log",
    "sample_count": 11
  }
}
```

### Processed Data (`data/processed/`)

**JSON** (`parallelism.json`):

```json
{
  "metadata": {
    "timestamp": "2025-11-20T09:46:00",
    "platform": { "cpu": "Apple M4 (10-core)", ... },
    "thread_counts": [1, 2, 4, 6, 8, 10],
    "duration_sec": 12,
    "analysis_window_sec": 10
  },
  "algorithms": {
    "RSA-3072": {
      "sign_ops_per_sec": {"1": 149.8, "2": 296.2, ..., "10": 1483.4},
      "sign_scaling_efficiency": {"1": 1.0, "2": 0.99, ..., "10": 0.99},
      "sign_latency_ms": {"1": 6.7, "2": 6.9, ..., "10": 7.6},
      "sign_energy_per_op": {"1": 0.00037, "10": 0.00042},
      "verify_ops_per_sec": { ... },
      "verify_scaling_efficiency": { ... },
      "verify_latency_ms": { ... },
      "verify_energy_per_op": { ... }
    }
  }
}
```

**CSV** (`parallelism_summary.csv`):

```csv
algorithm,operation,worker_count,ops_per_sec,scaling_efficiency,p99_latency_ms
RSA-3072,sign,1,149.8,1.0,6.7
RSA-3072,sign,2,296.2,0.99,6.9
RSA-3072,sign,4,590.5,0.98,7.1
RSA-3072,sign,6,884.3,0.98,7.3
RSA-3072,sign,8,1180.1,0.99,7.4
RSA-3072,sign,10,1483.4,0.99,7.6
RSA-3072,verify,1,9822.7,1.0,0.11
...
```

## Understanding Metrics

### Throughput (`ops_per_sec`)

We divide operations that landed inside the 10-second analysis window by 10.  
Example: 10 workers complete 14,800 sign operations → `14800 / 10 = 1480 ops/sec`.

### Scaling Efficiency

`efficiency = ops_per_sec(worker_count) / (worker_count × ops_per_sec(1 worker))`

- >90% → near-linear scaling (CPU-bound).  
- 70–90% → mild contention / memory pressure.  
- <70% → investigate (SPHINCS+ hashing depth, cache thrash, or scheduler noise).

### Latency statistics

Each worker records up to `max_latency_samples` during the analysis window:
- `p50/p95/p99` highlight contention outliers (especially important for SPHINCS+).  
- If you see `None`, the worker completed too few ops (common for RSA-15360); consider running longer.

### CPU and Memory Telemetry

- `avg_cpu_utilization` is derived from per-worker CPU time divided by the analysis window. Expect ~0.95 for CPU-saturated configs.  
- `avg_rss_bytes` indicates steady-state memory footprint per worker; useful for projecting container sizing.

### Energy per Operation

When powermetrics runs with sudo:

```
energy_per_op = (ΔCPU energy joules) / analysis_ops
```

Use this to compare “ops per watt” trade-offs across algorithms or to model server sizing.

## Use Cases for Output Data

### 1. Throughput Analysis
```python
import json
with open('data/processed/parallelism_multiprocess.json') as f:
    data = json.load(f)

# Calculate signing throughput with 8 workers
for algo in ['RSA-3072', 'ECDSA-P256', 'Dilithium2']:
    signing_ops = data['algorithms'][algo]['sign_ops_per_sec'][str(8)]
    print(f"{algo}: {signing_ops:.0f} ops/sec @ 8 workers")
```

### 2. Verification Throughput
```python
# Compare verification performance with 4 workers
workers = 4
for algo in ['ECDSA-P256', 'Ed25519', 'Dilithium2']:
    verify_ops = data['algorithms'][algo]['verify_ops_per_sec'][str(workers)]
    print(f"{algo}: {verify_ops:.0f} ops/sec ({workers} workers)")
```

### 3. Scaling + Energy Analysis

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/processed/parallelism_summary.csv')

# Plot scaling curves for signing
for algo in ['RSA-3072', 'ECDSA-P256', 'Dilithium2', 'SPHINCS+-128s']:
    subset = df[(df['algorithm'] == algo) & (df['operation'] == 'sign')]
    plt.plot(subset['worker_count'], subset['ops_per_sec'], label=algo, marker='o')

plt.xlabel('Worker Count')
plt.ylabel('Signing Throughput (ops/sec)')
plt.legend()
plt.grid(True)
plt.savefig('results/parallelism_signing_scaling.png')

# Optional: energy per operation from JSON
with open('data/processed/parallelism_multiprocess.json') as f:
    data = json.load(f)
energy = data['algorithms']['Dilithium2']['sign_energy_per_op'][str(10)]
print(f"Dilithium2 energy/op @10 workers: {energy * 1e3:.2f} mJ")
```

## Interrupting and Resuming

- **Ctrl+C** at any point — all worker processes terminate and already-finished algorithms retain their outputs.  
- Key material lives in `data/cache/`, so reruns skip key generation even after interruptions.  
- Re-running currently executes the full suite; to resume a subset, temporarily edit `ParallelismBenchmark.run_all()` to filter `all_algos`.

## Customization

### Adjust Worker Sets and Telemetry

`config/benchmark.json` → `parallelism` block:

```json
"parallelism": {
  "thread_counts": [1, 2, 4, 6, 8, 10],
  "duration_sec": 12,
  "analysis_window_sec": 10,
  "warmup_duration_sec": 1,
  "sample_interval_sec": 0.25,
  "max_latency_samples": 1024,
  "cache_dir": "data/cache",
  "cpu_affinity": {
    "enabled": true,
    "performance_cores": [0, 1, 2, 3, 4, 5],
    "efficiency_cores": [6, 7, 8, 9],
    "strategy": "performance_first"
  },
  "measurement": {
    "collect_latencies": true,
    "collect_cpu_percent": true,
    "collect_memory": true
  },
  "powermetrics": {
    "enabled": true,
    "binary": "/usr/bin/powermetrics",
    "samplers": ["smc"],
    "sample_interval_sec": 1.0
  }
}
```

Tuning tips:

- `thread_counts`: match your host topology; include efficiency cores only if you want heterogenous scaling data.  
- `analysis_window_sec`: shrink to 6–8 s when running on noisy laptops to reduce tail impact; keep ≥ half of `duration_sec`.  
- `sample_interval_sec`: increase (e.g., to 1.0) to reduce telemetry overhead on very slow algorithms.  
- `powermetrics.enabled`: set to `false` if you cannot run with sudo; energy-related columns will show `null`.  
- `cpu_affinity.enabled`: macOS cannot enforce strict pinning; disable to suppress warnings if the extra logging is noisy.

### Test Specific Algorithms

Modify `src/benchmarks/parallelism_threading.py` or `src/benchmarks/parallelism_multiprocess.py`:

```python
# In run_all() method, filter algorithms:
all_algos = self.factory.get_all_algorithms()
all_algos = [a for a in all_algos if a['name'] in ['RSA-3072', 'Dilithium2']]
```

## Troubleshooting

### Powermetrics Permission Error

```
[powermetrics] powermetrics requires sudo access; disable powermetrics or re-run with sudo.
```

- Either re-run with sudo (e.g., `sudo -E python src/benchmarks/parallelism_threading.py`), or set `"powermetrics.enabled": false` in `config/benchmark.json`.  
- When disabled, `energy_per_op` and temperature fields become `null`.

### CPU Affinity Warning on macOS

```
[affinity] macOS does not expose strict per-core pinning...
```

- Informational only; macOS cannot hard-pin processes. Keep background load low for repeatable data.  
- To silence the warning, set `"cpu_affinity.enabled": false`.

### `psutil` Missing

If you see `ModuleNotFoundError: No module named 'psutil'`, install dependencies:

```bash
pip install -r requirements.txt
```

Telemetry falls back to empty values without psutil, but you lose CPU/RSS data.

### Low Scaling Efficiency

Symptoms: `scaling_efficiency` < 0.7 for otherwise fast algorithms.

Checklist:
1. Ensure no other heavy workloads are running (Activity Monitor / `top`)  
2. Verify powermetrics isn't saturating I/O (`sample_interval_sec >= 1.0` for slow systems)  
3. For SPHINCS+, expect 0.65–0.8 due to hash-tree memory churn; align with design doc  
4. **Threading mode only**: Fast algorithms (ECDSA-P256, Ed25519) may show <0.4 efficiency at 8+ threads due to cache contention from OpenSSL's precomputed lookup tables. This is expected behavior, not a bug:
   - 2-4 threads: Efficiency >85% confirms GIL is released
   - 6 threads: Moderate degradation (50-70%) is acceptable
   - 8-10 threads: Severe degradation (<40%) reflects cache line contention
   - Slower algorithms (RSA, ECDSA-P384) maintain better scaling as operation time dominates cache effects

### Very Slow Configurations

RSA-15360 and SPHINCS+-256 take a long time to accumulate operations. Seeing `<5 ops` in a 10-second window is normal; the analysis window still records accurate ops/sec and latency numbers. Increase `duration_sec` if you need more stable stats.

## Performance Tips

1. **Disable Turbo Boost** (for consistency):
   ```bash
   # macOS: Not easily controllable
   # Linux: echo "1" > /sys/devices/system/cpu/intel_pstate/no_turbo
   ```

2. **Monitor CPU usage**:
   ```bash
   # In another terminal
   top -pid $(pgrep -f 'parallelism_(threading|multiprocess).py')
   ```

3. **Check thermal state** (macOS):
   ```bash
   sudo powermetrics --samplers smc -i 1000 -n 1 | grep -i "CPU die temperature"
   ```

---

**Last Updated**: 2025-11-20
