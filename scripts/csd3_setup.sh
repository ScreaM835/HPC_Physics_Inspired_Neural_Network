#!/bin/bash
# ============================================================
# CSD3 One-Time Setup Script for QNM PINN Project
# ============================================================
# Run this ONCE on a CSD3 login node to set up the environment.
#
# Usage:
#   ssh <CRSid>@login-cpu.hpc.cam.ac.uk
#   cd /home/<CRSid>
#   # Upload project via scp/rsync first (see instructions below)
#   bash project32_qnm_pinn/scripts/csd3_setup.sh
#
# Upload project to CSD3 (run from your LOCAL machine):
#   scp -r project32_qnm_pinn <CRSid>@login-cpu.hpc.cam.ac.uk:~/
#   # OR with rsync (faster for updates):
#   rsync -avz --exclude '__pycache__' --exclude '*.pyc' --exclude 'outputs/' \
#     project32_qnm_pinn/ <CRSid>@login-cpu.hpc.cam.ac.uk:~/project32_qnm_pinn/
# ============================================================

set -e

echo "=== CSD3 Setup for QNM PINN Project ==="

# Load the Ampere (A100 GPU) default environment
module purge
module load rhel8/default-amp
module load python/3.8.11/gcc-9.4.0-yb6rzr6

echo "[1/3] Creating Python virtual environment..."
cd ~/project32_qnm_pinn

# Create venv if it doesn't exist
if [ ! -d "venv_csd3" ]; then
    python -m venv venv_csd3
    echo "  Created venv_csd3"
else
    echo "  venv_csd3 already exists, skipping creation"
fi

source venv_csd3/bin/activate

echo "[2/3] Installing dependencies..."
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install numpy scipy pyyaml tqdm matplotlib

echo "[3/3] Verifying installation..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
else:
    print('NOTE: CUDA not available on login node (expected).')
    print('      GPU will be available when running via SLURM on ampere partition.')
print('All imports OK')
"

echo ""
echo "=== Setup Complete ==="
echo "To submit a training job, run:"
echo "  sbatch scripts/csd3_submit.sh"
echo ""
echo "To check your GPU hour balance:"
echo "  mybalance"
