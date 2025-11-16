# Environment Setup Guide

> One-command installation to get started with digital signature algorithm benchmarking

## Table of Contents

- [Quick Start](#quick-start)
- [System Requirements](#system-requirements)
- [Manual Installation](#manual-installation)
- [Troubleshooting](#troubleshooting)
- [Verification](#verification)
- [Next Steps](#next-steps)
- [References](#references)

---

## Quick Start

### Automated Installation

```bash
# Run the automated setup script
bash setup.sh
```

The script automatically:
- ✅ Verifies Python 3.9+ is installed
- ✅ Creates virtual environment in `venv/`
- ✅ Installs all Python dependencies
- ✅ Validates installation success

### Verify Installation

```bash
# Test all algorithms
python test_algorithms.py
```

**Expected output:** `17/17 algorithms passed ✓`

---

## System Requirements

### Operating System

- **macOS**: 10.15+ (Catalina or later)
- **Linux**: Ubuntu 20.04+, Debian 11+, or equivalent
- **Windows**: 10/11 (with WSL2 recommended)

### Python

- **Version**: 3.9 or higher
- **Check your version**:
  ```bash
  python3 --version
  ```

### System Dependencies

#### macOS

```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install liboqs library
brew install liboqs
```

#### Linux (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake liboqs-dev
```

#### Linux (Fedora/RHEL)

```bash
sudo dnf install -y gcc gcc-c++ cmake liboqs-devel
```

### Python Dependencies

All dependencies are specified in `requirements.txt`:

```
cryptography>=46.0.0    # Baseline benchmarks (cryptography library)
cffi>=1.16.0            # Direct OpenSSL bindings for GIL-releasing parallelism
liboqs-python>=0.14.0   # Post-quantum algorithms (Dilithium, SPHINCS+)
numpy>=2.0.0            # Statistical analysis
psutil>=5.9.0           # System telemetry
```

**Implementation Note:**
- **Baseline benchmarks**: Use `cryptography` library (Python wrapper)
- **Parallelism benchmarks**: Use direct OpenSSL bindings via CFFI for classical algorithms (GIL-releasing) and liboqs for PQC algorithms

**Installation:**
```bash
pip install -r requirements.txt
```

---

## Manual Installation

If the automated script fails or you prefer manual setup:

### Step 1: Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows PowerShell)
venv\Scripts\Activate.ps1

# Activate (Windows Command Prompt)
venv\Scripts\activate.bat
```

### Step 2: Upgrade pip

```bash
pip install --upgrade pip
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import cryptography, oqs, numpy; print('✓ All packages installed successfully')"
```

---

## Troubleshooting

### Issue 1: liboqs-python Import Error

**Error message:**
```
ImportError: liboqs shared library not found
```

**Root cause:** System liboqs library is missing or was built without shared library support.

**Solution:** Build liboqs from source with shared library flag enabled

```bash
# Step 1: Clone liboqs repository
cd ~
git clone https://github.com/open-quantum-safe/liboqs.git
cd liboqs

# Step 2: Create build directory
mkdir build && cd build

# Step 3: Configure with shared library support
# For macOS:
cmake -DCMAKE_INSTALL_PREFIX=/opt/homebrew \
      -DBUILD_SHARED_LIBS=ON \
      -DCMAKE_BUILD_TYPE=Release \
      ..

# For Linux:
cmake -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DBUILD_SHARED_LIBS=ON \
      -DCMAKE_BUILD_TYPE=Release \
      ..

# Step 4: Build (use all CPU cores)
# macOS:
make -j$(sysctl -n hw.ncpu)

# Linux:
make -j$(nproc)

# Step 5: Install
sudo make install

# Step 6: Verify installation
# macOS:
ls -la /opt/homebrew/lib/liboqs.*

# Linux:
ls -la /usr/local/lib/liboqs.*

# You should see liboqs.dylib (macOS) or liboqs.so (Linux)

# Step 7: Update library path (Linux only)
sudo ldconfig

# Step 8: Reinstall Python bindings
cd /path/to/P5003
source venv/bin/activate
pip install --force-reinstall liboqs-python

# Step 9: Test
python -c "import oqs; print('liboqs version:', oqs.__version__)"
```

### Issue 2: cryptography Build Error

**Error message:**
```
error: can't find Rust compiler
```

**Root cause:** The `cryptography` package requires Rust compiler for building native extensions.

**Solution:** Install Rust toolchain

```bash
# Install Rust (macOS/Linux)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Verify Rust installation
rustc --version

# Reinstall cryptography
pip install --upgrade --force-reinstall cryptography
```

**Alternative:** Use pre-built wheels

```bash
# Upgrade pip to ensure binary wheels are available
pip install --upgrade pip setuptools wheel

# Try installing again (should use binary wheel)
pip install cryptography
```

### Issue 3: Python Version Too Old

**Error message:**
```
ERROR: Python 3.9+ required
```

**Solution:** Install a newer Python version

**macOS:**
```bash
# Using Homebrew
brew install python@3.13

# Create virtual environment with new Python
python3.13 -m venv venv
source venv/bin/activate
```

**Linux (Ubuntu/Debian):**
```bash
# Add deadsnakes PPA for newer Python versions
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update

# Install Python 3.13
sudo apt-get install python3.13 python3.13-venv

# Create virtual environment
python3.13 -m venv venv
source venv/bin/activate
```

**Linux (from source):**
```bash
# Download and build Python
cd /tmp
wget https://www.python.org/ftp/python/3.13.0/Python-3.13.0.tgz
tar -xzf Python-3.13.0.tgz
cd Python-3.13.0
./configure --enable-optimizations
make -j$(nproc)
sudo make altinstall

# Create virtual environment
python3.13 -m venv ~/P5003/venv
```

### Issue 4: Permission Denied on setup.sh

**Error message:**
```
Permission denied: ./setup.sh
```

**Solution:**
```bash
chmod +x setup.sh
./setup.sh
```

### Issue 5: Virtual Environment Activation Fails (Windows)

**Error message:**
```
cannot be loaded because running scripts is disabled on this system
```

**Solution:**
```powershell
# Open PowerShell as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Retry activation
venv\Scripts\Activate.ps1
```

---

## Verification

### Test Suite Results

After running `python test_algorithms.py`, you should see:

```
====================================================================================================
DIGITAL SIGNATURE ALGORITHM TEST SUITE
====================================================================================================

Found 17 algorithms to test.

[1/17] Testing RSA-3072 (RSA)...
  Testing keygen... ✓ (129.112 ms)
  Testing sign... ✓ (5.328 ms)
  Testing verify... ✓ (0.119 ms)
  Testing invalid signature... ✓ (0.325 ms)
  Measuring sizes... ✓

...

====================================================================================================
ALGORITHM SUMMARY TABLE
====================================================================================================
Algorithm                    Security    S   Keygen(ms)   Sign(ms)     Verify(ms)   PubKey(B)   PrivKey(B)  Sig(B)
------------------------------------------------------------------------------------------------------------------
Ed25519                      128-bit     ✓   0.123        0.045        0.089        32          32          64
ECDSA-P256                   128-bit     ✓   1.234        0.789        0.456        91          138         70
RSA-3072                     128-bit     ✓   129.112      5.328        0.119        422         1793        384
Dilithium2                   128-bit     ✓   0.567        0.234        0.123        1312        2528        2420
SPHINCS+-SHA2-128s-simple    128-bit     ✓   0.089        45.678       0.234        32          64          7856
SPHINCS+-SHA2-128f-simple    128-bit     ✓   0.091        12.345       0.245        32          64          17088
...

====================================================================================================
TEST STATISTICS
====================================================================================================
Total algorithms tested: 17
Successful: 17 (100.0%)
Failed: 0 (0.0%)
Errors: 0 (0.0%)
====================================================================================================

✓ All tests passed!
```

### Performance Comparison Highlights

| Algorithm | Security | Public Key | Private Key | Signature | Sign Time |
|-----------|----------|------------|-------------|-----------|-----------|
| **Ed25519** | 128-bit | 32 B | 32 B | 64 B | ~0.05 ms |
| **ECDSA-P256** | 128-bit | 91 B | 138 B | 70 B | ~0.8 ms |
| **RSA-3072** | 128-bit | 422 B | 1793 B | 384 B | ~5.3 ms |
| **Dilithium2** | 128-bit | 1312 B | 2528 B | 2420 B | ~0.23 ms |
| **SPHINCS+-128s** | 128-bit | 32 B | 64 B | 7856 B | ~45 ms |
| **SPHINCS+-128f** | 128-bit | 32 B | 64 B | 17088 B | ~12 ms |

**Key observations:**
- ✅ Classical algorithms: Ed25519 has smallest signature size (64 B)
- ✅ PQC algorithms: Dilithium signatures 3-7x smaller than SPHINCS+
- ✅ SPHINCS+ "fast" variant: 2x larger signatures but 4x faster signing

---

## Next Steps

Once installation is complete:

### 1. Explore Project Structure

```bash
# View comprehensive project documentation
cat README.md

# Examine algorithm configurations
cat config/algorithms.json

# Check benchmark parameters
cat config/benchmark.json
```

### 2. Run Benchmarks

```bash
# Run baseline benchmarks (single-core performance)
python -m src.benchmarks.baseline

# Run parallel benchmarks (multi-core scaling)
python -m src.benchmarks.parallelism_threading
python -m src.benchmarks.parallelism_multiprocess
```

### 3. Analyze Results

```bash
# View processed summaries
cat data/processed/baseline_summary.json
cat data/processed/parallelism_multiprocess.json
cat data/processed/parallelism_threading.json

# Results and visualizations are saved to results/
ls results/
```

---

## Using External Virtual Environment

If you prefer to create the virtual environment outside the project directory:

```bash
# Create external environment
python3 -m venv ~/envs/P5003_env

# Activate
source ~/envs/P5003_env/bin/activate

# Navigate to project
cd /path/to/P5003

# Install dependencies
pip install -r requirements.txt

# Run tests
python test_algorithms.py
```

---

## Docker Alternative

For consistent cross-platform environments:

```dockerfile
# Dockerfile (example)
FROM python:3.13-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Build liboqs from source
RUN git clone https://github.com/open-quantum-safe/liboqs.git && \
    cd liboqs && \
    mkdir build && cd build && \
    cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=/usr/local .. && \
    make -j$(nproc) && \
    make install && \
    ldconfig

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["python", "test_algorithms.py"]
```

**Build and run:**
```bash
docker build -t pqc-benchmark .
docker run --rm pqc-benchmark
```

---

## References

### Official Documentation

- **liboqs**: https://github.com/open-quantum-safe/liboqs
- **liboqs-python**: https://github.com/open-quantum-safe/liboqs-python
- **cryptography.io**: https://cryptography.io/en/latest/
- **Open Quantum Safe**: https://openquantumsafe.org/

### Troubleshooting Resources

- **liboqs Installation Guide**: https://github.com/open-quantum-safe/liboqs/wiki/Customizing-liboqs
- **Python Virtual Environments**: https://docs.python.org/3/library/venv.html
- **Rust Installation**: https://www.rust-lang.org/tools/install

---

**Last Updated:** November 13, 2025  
**Feedback:** If you encounter issues not covered here, please document and report them.
