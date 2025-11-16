# P5003: Benchmarking Classical and Post-Quantum Digital Signatures

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform: macOS](https://img.shields.io/badge/platform-macOS-lightgrey.svg)](https://www.apple.com/macos/)

A reproducible benchmark suite for classical (RSA, ECDSA, EdDSA) and post-quantum (CRYSTALS‑Dilithium, SPHINCS+) digital signatures on Apple M4 hardware.

---

## What This Project Provides

- **Baseline performance**: Single-core latency, throughput, and key/signature sizes for keygen/sign/verify operations.
- **Parallel scaling**: Threading and multiprocessing experiments (1–10 workers) with GIL-releasing implementations, covering 17 algorithms (8 classical + 9 PQC).
- **Reproducibility**: All experiments defined via `config/*.json`, with raw and processed data stored in `data/`.

## Current Status

Completed baseline, threading, and multiprocessing benchmarks for 17 signature algorithms on Apple M4 hardware. Processed data available in `data/processed/` as JSON and CSV files.

## Project Structure

```
P5003/
├── config/              # Experiment parameters (algorithms, threads, iterations)
├── data/                # Benchmark data (raw measurements and processed summaries)
│   ├── raw/             # Per-algorithm CSV files from individual runs
│   ├── processed/       # Aggregated JSON/CSV summaries
│   └── cache/           # Cached baseline results for quick lookup
├── docs/                # Documentation (setup guides, benchmark instructions)
├── results/             # Analysis outputs and figures
├── src/                 # Core source code
│   ├── algorithms/      # Algorithm adapters (classical & PQC)
│   ├── benchmarks/      # Benchmarking logic (baseline, parallelism)
│   └── utils/           # Helper utilities (timer, stats, affinity, power)
├── scripts/             # Automation scripts for running experiments
├── tests/               # Unit tests for algorithm correctness
├── setup.sh             # Project setup and installation script
└── requirements.txt     # Python dependencies
```

## Getting Started

### Prerequisites

- **OS**: macOS
- **Python**: 3.9+
- **Toolchain**: Apple Clang / LLVM (Xcode Command Line Tools)

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/Asleeps/P5003.git
cd P5003
```

2. **Run the setup script** (creates `venv` and installs deps, including `liboqs`)

```bash
bash setup.sh
```

If you open a new terminal later, reactivate with:

```bash
source venv/bin/activate
```

## How to Run Experiments

All entrypoints live in `scripts/` and `src/benchmarks/`.

### Baseline and Parallelism Benchmarks

Run individual parallelism studies if needed:

```bash
# Threading with GIL-releasing implementations (OpenSSL via CFFI for classical,
# liboqs for PQC)
python -m src.benchmarks.parallelism_threading

# Multiprocessing (one process per worker)
python -m src.benchmarks.parallelism_multiprocess
```

## Configuration

Experiment parameters can be easily modified without changing the source code:

- **`config/algorithms.json`**: Add or remove cryptographic algorithms to be tested.
- **`config/benchmark.json`**: Adjust thread counts, iteration budgets, message sizes, and scenario-specific constants.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{p5003_Digital_Signatures,
  author = {Chen, Xinzhe},
  title = {P5003: Classical and Post-Quantum Digital Signatures},
  year = {2025},
  url = {https://github.com/Asleeps/P5003}
}
```

## Contact

For questions or feedback, please open an issue on [GitHub](https://github.com/Asleeps/P5003/issues).

