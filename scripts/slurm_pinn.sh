#!/bin/bash
# ============================================================
# SLURM Job: PINN Training (CPU) + QNM extraction
# Supports checkpointing: resubmit with same script to resume.
# ============================================================
#SBATCH --job-name=qnm_pinn
#SBATCH --output=qnm_pinn_%j.out
#SBATCH --error=qnm_pinn_%j.err
#SBATCH --account=fergusson-sl3-cpu
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@300

# Flush Python output immediately (so tqdm progress appears in .err)
export PYTHONUNBUFFERED=1

set -e

WORKDIR=/home/ycc44/project32_qnm_pinn_repo_fd_refinement/project32_qnm_pinn
CONFIG=configs/zerilli_l2.yaml
CKPT_DIR="$WORKDIR/outputs/pinn/zerilli_l2/checkpoints"

# ---- Signal handler: save checkpoint on approaching time limit ----
requeue_handler() {
    echo "[SIGNAL] Caught USR1 — job approaching time limit."
    echo "[SIGNAL] Checkpoint should already be saved periodically."
    echo "[SIGNAL] Resubmitting job to continue training..."
    sbatch "$WORKDIR/scripts/slurm_pinn.sh"
    exit 0
}
trap requeue_handler USR1

echo "============================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURM_NODELIST"
echo "Started:       $(date)"
echo "============================================"

cd "$WORKDIR"

# --- Environment setup ---
module purge
module load rhel8/default-amp
module load python/3.11.0-icl

if [ ! -f "venv_csd3/bin/activate" ]; then
    echo "[SETUP] Creating venv..."
    python3 -m venv venv_csd3
fi

source venv_csd3/bin/activate
echo "[SETUP] Python: $(python --version)"

# Install deps (will be fast if already installed by the FD job)
pip install --quiet .

# GPU check
echo "[GPU] Checking GPU availability..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available:  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU device:      {torch.cuda.get_device_name(0)}')
    print(f'GPU memory:      {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"
nvidia-smi 2>/dev/null || echo "(No GPU driver — running on CPU)"

# --- Determine whether to resume ---
RESUME_FLAG=""
if ls "$CKPT_DIR"/model*.pt 1>/dev/null 2>&1; then
    echo "[CKPT] Found existing checkpoint — will resume training"
    RESUME_FLAG="--resume"
fi

# --- Train PINN ---
echo ""
echo "============================================"
echo "[PINN] Training PINN (zerilli_l2.yaml)..."
echo "  Framework: DeepXDE (PyTorch backend)"
echo "  Config: 10k Adam + 15k L-BFGS"
echo "  Nr=32000, Ni=800, Nb=400"
echo "  Gradient balancing: enabled (Wang et al. 2021)"
echo "  Checkpoint every 500 iters"
if [ -n "$RESUME_FLAG" ]; then
    echo "  >>> RESUMING from checkpoint <<<"
fi
echo "============================================"
python scripts/run_pinn.py --config "$CONFIG" --checkpoint-every 500 $RESUME_FLAG

# --- Extract QNMs from PINN ---
echo ""
echo "============================================"
echo "[QNM] Extracting QNMs from PINN output..."
echo "============================================"
python scripts/extract_qnm.py --config "$CONFIG" --source pinn

echo ""
echo "============================================"
echo "Finished: $(date)"
echo "============================================"
