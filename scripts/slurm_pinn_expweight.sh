#!/bin/bash
# ============================================================
# SLURM Job: PINN Training (CPU) + QNM extraction
# Base model + exponential reweighting (no RAD).
# ============================================================
#SBATCH --job-name=qnm_pinn_exp
#SBATCH --output=qnm_pinn_%j.out
#SBATCH --error=qnm_pinn_%j.err
#SBATCH --account=fergusson-sl3-cpu
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --cpus-per-task=72
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@300

# Flush Python output immediately (so tqdm progress appears in .err)
export PYTHONUNBUFFERED=1

set -e

WORKDIR=/home/ycc44/project32_qnm_pinn_repo_fd_refinement/project32_qnm_pinn_improved
VENV_DIR=/home/ycc44/project32_qnm_pinn_repo_fd_refinement/project32_qnm_pinn/venv_csd3
CONFIG=configs/zerilli_l2_expweight.yaml
CKPT_DIR="$WORKDIR/outputs/pinn/zerilli_l2_expweight/checkpoints"

# ---- Signal handler: save checkpoint on approaching time limit ----
requeue_handler() {
    echo "[SIGNAL] Caught USR1 — job approaching time limit."
    echo "[SIGNAL] Checkpoint should already be saved periodically."
    echo "[SIGNAL] Resubmitting job to continue training..."
    sbatch "$WORKDIR/scripts/slurm_pinn_expweight.sh"
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

# Use the shared venv from the benchmark repo (no separate venv needed)
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "[ERROR] Shared venv not found at $VENV_DIR" >&2
    exit 1
fi

source "$VENV_DIR/bin/activate"
echo "[SETUP] Python: $(python --version)"

# Use all allocated cores for PyTorch intra-op parallelism
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "[SETUP] OMP_NUM_THREADS=$OMP_NUM_THREADS"

# Install the improved package into the shared venv
rm -rf build/ dist/ src/*.egg-info
export TMPDIR="$WORKDIR/.pip_tmp"
mkdir -p "$TMPDIR"
pip install --quiet .
rm -rf "$TMPDIR" build/ dist/
unset TMPDIR

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
echo "[PINN] Training FNN + Exp Reweighting (no RAD)..."
echo "  Framework: DeepXDE (PyTorch backend)"
echo "  Model: FNN [2,80,40,20,10,1], tanh, Glorot uniform"
echo "  Config: 10k Adam + 15k L-BFGS"
echo "  Nr=32000, Ni=800, Nb=400"
echo "  Lambda: [100,100,100,1,100,1,1] (paper weights)"
echo "  Uniform resampling: period=100"
echo "  Exponential reweighting: tau_est=11.241"
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
