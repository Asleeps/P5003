# P5003: Benchmarking Classical and Post-Quantum Digital Signatures

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)

This project provides a comprehensive framework for benchmarking and analyzing the performance of classical (RSA, ECDSA, EdDSA) and post-quantum (CRYSTALS-Dilithium, SPHINCS+) digital signature algorithms. It goes beyond simple performance metrics by modeling their impact in real-world scenarios like TLS, JWT, code signing, and DNSSEC, with a special focus on the constraints of Python's Global Interpreter Lock (GIL).

The full experimental design is detailed in `experiment_design.md`.

---

## Key Features

- **Baseline Performance**: Measures latency, throughput, and memory for key generation, signing, and verification.
- **System-Level Parallelism**: Analyzes multi-core scaling using both `threading` and `multiprocessing` to quantify the GIL's impact.
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
├── experiment_design.md # In-depth experimental design document
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
    git clone <your-repo-url>
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
    This command executes the core performance measurements as defined in `config/benchmark.json`. Results are saved to `data/raw/` and `data/processed/`.
    ```bash
    python -m scripts.run_all_benchmarks
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

## Citing This Work

If you use this project in your research, please consider citing it. (Details to be added).

