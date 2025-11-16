# P5003: Benchmarking Classical and Post-Quantum Digital Signatures

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform: macOS](https://img.shields.io/badge/platform-macOS-lightgrey.svg)](https://www.apple.com/macos/)

A reproducible benchmark suite for classical (RSA, ECDSA, EdDSA) and post-quantum (CRYSTALS‑Dilithium, SPHINCS+) digital signatures, with scenario models for TLS, JWT, code signing, and DNSSEC.

---

## What This Project Provides

- **Baseline performance**: Single-core latency, throughput, and size for keygen/sign/verify.
- **Parallel scaling**: Threading (GIL-releasing implementations) and multiprocessing benchmarks across multiple cores.
- **Scenario modeling**: TLS 1.3, JWT API gateways, macOS code signing, and DNSSEC response sizing.
- **Reproducibility**: All experiments defined via `config/*.json`, results stored in `data/` and `results/` with metadata.

## Project Structure

```
P5003/
├── config/              # Experiment parameters (algorithms, threads, iterations)
├── data/                # Raw and processed measurement data
│   ├── raw/
│   └── processed/
├── results/             # Generated reports and figures
├── scripts/             # Automation scripts for running experiments and reports
├── src/                 # Core source code
│   ├── algorithms/      # Algorithm adapters (classical & PQC)
│   ├── benchmarks/      # Benchmarking logic (baseline, parallelism)
│   ├── scenarios/       # Real-world scenario models (TLS, JWT, etc.)
│   └── utils/           # Helper utilities (timer, stats, power)
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

2. **Run the setup script** (creates `.venv` and installs deps, including `liboqs`)

```bash
bash setup.sh
```

If you open a new terminal later, reactivate with:

```bash
source .venv/bin/activate
```

## How to Run Experiments

All entrypoints live in `scripts/` and `src/benchmarks/`.

### 1. Baseline and Parallelism Benchmarks

Run the full experiment suite (baseline + all parallel modes) as defined in `config/benchmark.json`:

```bash
python -m scripts.run_all_benchmarks
```

Run individual parallelism studies if needed:

```bash
# Threading with GIL-releasing implementations (OpenSSL via CFFI for classical,
# liboqs for PQC)
python -m src.benchmarks.parallelism_threading

# Multiprocessing (one process per worker)
python -m src.benchmarks.parallelism_multiprocess
```

### 2. Generate Scenario Reports

After benchmarks complete, build the scenario models (TLS, JWT, code signing, DNSSEC) and figures:

```bash
python -m scripts.generate_report
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
@software{p5003_pqc_benchmark,
  author = {Chen, Xinzhe},
  title = {P5003: Benchmarking Classical and Post-Quantum Digital Signatures},
  year = {2025},
  url = {https://github.com/Asleeps/P5003}
}
```

## Contact

For questions or feedback, please open an issue on [GitHub](https://github.com/Asleeps/P5003/issues).

