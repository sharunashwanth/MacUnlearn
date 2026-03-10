#!/bin/bash
# =============================================================================
# Kaggle Setup Script for SimNPO+SAM Experiments
# Run this ONCE at the start of each Kaggle session.
# =============================================================================
#
# Usage (in a Kaggle notebook cell):
#   !bash scripts/kaggle_setup.sh
#
# Kaggle directory structure:
#   /kaggle/working/       <- persisted as notebook output (20GB limit)
#   /kaggle/input/         <- read-only datasets you attach
#   /kaggle/tmp/           <- scratch space, cleared on restart
#
# Persistence strategy:
#   saves/ and hf_cache/ live in /kaggle/working/ so they survive as output.
#   On new sessions, attach previous output as a dataset input to resume.
# =============================================================================

set -e

KAGGLE_WORK="/kaggle/working"
REPO_DIR="/kaggle/working/MacUnlearn"

echo "============================================"
echo "  SimNPO+SAM Kaggle Setup"
echo "============================================"

# --- 1. Clone or update repo ---
echo "[1/5] Setting up repository..."
if [ -d "${REPO_DIR}/.git" ]; then
    cd "${REPO_DIR}" && git pull
else
    cd "${KAGGLE_WORK}"
    git clone https://github.com/sharunashwanth/MacUnlearn.git
fi
cd "${REPO_DIR}"

# --- 2. Create persistent directories in /kaggle/working ---
echo "[2/5] Creating persistent directories..."
mkdir -p "${KAGGLE_WORK}/saves/unlearn"
mkdir -p "${KAGGLE_WORK}/saves/eval"
mkdir -p "${KAGGLE_WORK}/hf_cache"
mkdir -p "${KAGGLE_WORK}/data"

# Symlink saves/ to working dir (if not already)
if [ -d "${REPO_DIR}/saves" ] && [ ! -L "${REPO_DIR}/saves" ]; then
    cp -rn "${REPO_DIR}/saves/"* "${KAGGLE_WORK}/saves/" 2>/dev/null || true
    rm -rf "${REPO_DIR}/saves"
fi
ln -sfn "${KAGGLE_WORK}/saves" "${REPO_DIR}/saves"

if [ -d "${REPO_DIR}/data" ] && [ ! -L "${REPO_DIR}/data" ]; then
    cp -rn "${REPO_DIR}/data/"* "${KAGGLE_WORK}/data/" 2>/dev/null || true
    rm -rf "${REPO_DIR}/data"
fi
ln -sfn "${KAGGLE_WORK}/data" "${REPO_DIR}/data"

export HF_HOME="${KAGGLE_WORK}/hf_cache"

# --- 3. Restore from previous session (if attached as input dataset) ---
echo "[3/5] Checking for previous session data..."
# Kaggle input datasets are at /kaggle/input/<dataset-name>/
# Look for any attached dataset that contains our saves
for input_dir in /kaggle/input/*/; do
    if [ -d "${input_dir}saves" ]; then
        echo "  Found previous saves in ${input_dir}"
        cp -rn "${input_dir}saves/"* "${KAGGLE_WORK}/saves/" 2>/dev/null || true
        echo "  Restored checkpoints from previous session!"
    fi
    if [ -d "${input_dir}hf_cache" ]; then
        echo "  Found previous HF cache in ${input_dir}"
        cp -rn "${input_dir}hf_cache/"* "${KAGGLE_WORK}/hf_cache/" 2>/dev/null || true
        echo "  Restored HF cache from previous session!"
    fi
done

# --- 4. Install dependencies ---
echo "[4/5] Installing dependencies..."
cd "${REPO_DIR}"
pip install -q -e ".[lm-eval]"
# Kaggle's torch/torchvision are usually compatible, but fix just in case
pip install -q --no-deps torchvision==0.19.1 torch==2.4.1 2>/dev/null || true

# Download eval data if needed
if [ ! -f "${KAGGLE_WORK}/saves/eval/.downloaded" ]; then
    python setup_data.py --eval_logs 2>/dev/null || echo "  [WARN] eval_logs download skipped"
    touch "${KAGGLE_WORK}/saves/eval/.downloaded"
fi
if [ ! -f "${KAGGLE_WORK}/data/.idk_downloaded" ]; then
    python setup_data.py --idk 2>/dev/null || echo "  [WARN] idk download skipped"
    touch "${KAGGLE_WORK}/data/.idk_downloaded"
fi

# --- 5. Verify setup ---
echo "[5/5] Verifying..."
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
import psutil
print(f'  System RAM: {psutil.virtual_memory().total / 1e9:.1f} GB')
import os
print(f'  saves/ -> {os.path.realpath(\"saves\")}')
print(f'  HF_HOME: {os.environ.get(\"HF_HOME\", \"default\")}')
"

echo ""
echo "============================================"
echo "  Setup complete! Ready to run experiments."
echo "============================================"
echo ""
echo "Run experiments:"
echo "  cd /kaggle/working/MacUnlearn"
echo "  python scripts/run_phi_experiments.py --methods SimNPO_SAM --batch_size 1 --grad_accum 16 trainer.args.num_train_epochs=3"
echo ""
echo "Resume after restart:"
echo "  1. Save this notebook's output as a dataset"
echo "  2. In new session, attach that dataset as input"
echo "  3. Run: bash scripts/kaggle_setup.sh"
echo "  4. Run: python scripts/run_phi_experiments.py --resume --methods SimNPO_SAM --batch_size 1 --grad_accum 16 trainer.args.num_train_epochs=3"
