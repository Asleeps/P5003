#!/bin/bash
# Setup script for P5003 Digital Signature Algorithm Project

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/venv"

echo "=========================================="
echo "P5003 Project Environment Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version || {
    echo "Error: Python 3 not found"
    exit 1
}

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo "✓ Virtual environment created at $VENV_DIR"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet

# Install dependencies
echo "Installing Python packages..."
pip install -r "$PROJECT_DIR/requirements.txt"

# Quick sanity checks for native dependencies
echo "Checking native dependencies..."
python - <<'PY'
import shutil
import sys

def log(msg: str) -> None:
    print(msg)

try:
    import oqs  # type: ignore
    log("✓ liboqs-python import succeeded")
except Exception as exc:
    log(f"⚠ liboqs-python import failed: {exc}")
    log("  Ensure system liboqs is installed (brew install liboqs or apt-get liboqs-dev).")

openssl = shutil.which("openssl")
if openssl:
    log(f"✓ OpenSSL found at {openssl}")
else:
    log("⚠ OpenSSL binary not found in PATH; OpenSSL-backed threading benches may be unavailable.")

if sys.platform == "darwin":
    pm = "/usr/bin/powermetrics"
    if shutil.which(pm):
        log(f"ℹ powermetrics available at {pm} (requires sudo to collect power samples).")
    else:
        log("ℹ powermetrics not found; power sampling will be skipped.")
PY

echo ""
echo "=========================================="
echo "Environment setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment in the future:"
echo "  source venv/bin/activate"
echo ""
echo "To test all algorithms:"
echo "  python test_algorithms.py"
echo ""
