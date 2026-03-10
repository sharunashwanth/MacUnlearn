#!/bin/bash
# =============================================================================
# Colab Setup Script for SimNPO+SAM Experiments
# Run this ONCE at the start of each Colab session.
# =============================================================================
#
# Usage (in a Colab cell):
#   !bash scripts/colab_setup.sh
#
# This script:
#   1. Creates persistent directories on GDrive
#   2. Symlinks saves/ and HF cache to GDrive (survives reconnection)
#   3. Installs dependencies
#   4. Downloads eval data
# =============================================================================

set -e

GDRIVE_BASE="/content/drive/MyDrive/MacUnlearn"
REPO_DIR="/content/MacUnlearn"

echo "============================================"
echo "  SimNPO+SAM Colab Setup"
echo "============================================"

# --- 1. Create persistent GDrive directories ---
echo "[1/5] Creating GDrive directories..."
mkdir -p "${GDRIVE_BASE}/saves/unlearn"
mkdir -p "${GDRIVE_BASE}/saves/eval"
mkdir -p "${GDRIVE_BASE}/hf_cache"
mkdir -p "${GDRIVE_BASE}/data"

# --- 2. Symlink saves/ to GDrive ---
echo "[2/5] Symlinking saves/ and caches to GDrive..."
# If saves exists as a directory, move its content to GDrive then replace with symlink
if [ -d "${REPO_DIR}/saves" ] && [ ! -L "${REPO_DIR}/saves" ]; then
    cp -rn "${REPO_DIR}/saves/"* "${GDRIVE_BASE}/saves/" 2>/dev/null || true
    rm -rf "${REPO_DIR}/saves"
fi
ln -sfn "${GDRIVE_BASE}/saves" "${REPO_DIR}/saves"

if [ -d "${REPO_DIR}/data" ] && [ ! -L "${REPO_DIR}/data" ]; then
    cp -rn "${REPO_DIR}/data/"* "${GDRIVE_BASE}/data/" 2>/dev/null || true
    rm -rf "${REPO_DIR}/data"
fi
ln -sfn "${GDRIVE_BASE}/data" "${REPO_DIR}/data"

export HF_HOME="${GDRIVE_BASE}/hf_cache"

# --- 3. Install dependencies & Fix Torchvision ---
echo "[3/5] Installing dependencies..."
cd "${REPO_DIR}"
# Install everything except torch first to avoid conflicts, then fix torch/torchvision
pip install -q -e . 
# Fix the "torchvision::nms" error by ensuring compatible versions
pip install -q --no-deps torchvision==0.19.1 torch==2.4.1

# --- 4. Download eval logs & datasets ---
echo "[4/5] Downloading data (if needed)..."
if [ ! -f "${GDRIVE_BASE}/saves/eval/.downloaded" ]; then
    python setup_data.py --eval_logs
    touch "${GDRIVE_BASE}/saves/eval/.downloaded"
fi
if [ ! -f "${GDRIVE_BASE}/data/.idk_downloaded" ]; then
    python setup_data.py --idk
    touch "${GDRIVE_BASE}/data/.idk_downloaded"
fi

# --- 5. Verify setup ---
echo "[5/5] Verifying..."
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
import os
print(f'  saves/ -> {os.path.realpath(\"saves\")}')
print(f'  HF_HOME: {os.environ.get(\"HF_HOME\", \"default\")}')
"

echo ""
echo "============================================"
echo "  Setup complete! Ready to run experiments."
echo "============================================"
echo ""
echo "Quick start:"
echo "  python scripts/run_phi_experiments.py --batch_size 2 --grad_accum 8"
echo ""
echo "Resume after reconnection:"
echo "  1. Mount GDrive"
echo "  2. Clone repo (or it's already there)"  
echo "  3. Run: bash scripts/colab_setup.sh"
echo "  4. Run: python scripts/run_phi_experiments.py --resume --batch_size 2 --grad_accum 8"
