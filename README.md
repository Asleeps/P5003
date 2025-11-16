# P5003: Benchmarking Classical and Post-Quantum Digital Signatures

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform: macOS](https://img.shields.io/badge/platform-macOS-lightgrey.svg)](https://www.apple.com/macos/)

A comprehensive benchmarking framework for classical (RSA, ECDSA, EdDSA) and post-quantum (CRYSTALS-Dilithium, SPHINCS+) digital signature algorithms, with real-world scenario modeling for TLS, JWT, code signing, and DNSSEC.

---

## Key Features

- **Baseline Performance**: Measures latency, throughput, and memory for key generation, signing, and verification.
- **System-Level Parallelism**: Analyzes multi-core scaling using both `threading` (with GIL-releasing implementations via CFFI+OpenSSL for classical algorithms and liboqs for PQC) and `multiprocessing`. Tests across `[1, 2, 4, 6, 8, 10]` workers.
- **GIL-Free Classical Algorithms**: Direct OpenSSL bindings via CFFI enable classical algorithms (RSA/ECDSA/EdDSA) to release the Python GIL during cryptographic operations, achieving 85%+ efficiency at 2-4 threads. Performance at 8+ threads may be limited by cache contention (algorithm-specific), not GIL.
- **Real-World Scenario Modeling**: Evaluates algorithm suitability for:
  - TLS 1.3 Handshakes (CPU and bandwidth bottlenecks)
  - JWT API Gateways (verification throughput and header size)
  - macOS Code Signing (user-perceived launch delay)
  - DNSSEC (response size and protocol fallback)
- **Reproducibility**: All experiments are defined by configuration files and include metadata (`git_hash`, library versions) for full data provenance.

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

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Asleeps/P5003.git
    cd P5003
    ```

2.  **Run the setup script:**
    This script will create a Python virtual environment (`.venv`), activate it, and install all required dependencies, including `liboqs`.
    ```bash
    bash setup.sh
    ```
    *If you open a new terminal, reactivate the environment with `source .venv/bin/activate`.*

## How to Run Experiments

All experiments are orchestrated via scripts in the `scripts/` directory.

1.  **Run Baseline & Parallelism Benchmarks:**
  This command executes the core performance measurements (baseline + all parallelism modes) as defined in `config/benchmark.json`. Results are saved to `data/raw/` and `data/processed/`.
  ```bash
  python -m scripts.run_all_benchmarks
  ```

  To run the parallelism studies individually:
  ```bash
  # Threading with GIL-releasing implementations (all algorithms)
  # Classical: CFFI + direct OpenSSL bindings
  # PQC: liboqs library
  python -m src.benchmarks.parallelism_threading

  # Multiprocessing (process-per-worker, fully GIL-free)
  python -m src.benchmarks.parallelism_multiprocess
  ```

2.  **Generate Reports and Figures:**
    After the benchmarks are complete, this command runs the scenario models (TLS, JWT, etc.) and generates all tables and figures for the final report, saving them to `results/`.
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

