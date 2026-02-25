#!/bin/bash
# ============================================================
# SLURM Job: PINN Training (CPU) + QNM extraction
# RAD tuned: k=2.0, c=0.1 (sharper residual concentration)
# ============================================================
#SBATCH --job-name=qnm_pinn_k2
#SBATCH --output=qnm_pinn_%j.out
#SBATCH --error=qnm_pinn_%j.err
#SBATCH --account=fergusson-sl3-cpu
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --cpus-per-task=72
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@300

export PYTHONUNBUFFERED=1

set -e

WORKDIR=/home/ycc44/project32_qnm_pinn_repo_fd_refinement/project32_qnm_pinn_improved
VENV_DIR=/home/ycc44/project32_qnm_pinn_repo_fd_refinement/project32_qnm_pinn/venv_csd3
CONFIG=configs/zerilli_l2_rad_k2.yaml
CKPT_DIR="$WORKDIR/outputs/pinn/zerilli_l2_rad_k2/checkpoints"

requeue_handler() {
    echo "[SIGNAL] Caught USR1 — job approaching time limit."
    echo "[SIGNAL] Resubmitting job to continue training..."
    sbatch "$WORKDIR/scripts/slurm_pinn_rad_k2.sh"
    exit 0
}
trap requeue_handler USR1

echo "============================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURM_NODELIST"
echo "Started:       $(date)"
echo "============================================"

cd "$WORKDIR"

module purge
module load rhel8/default-amp
module load python/3.11.0-icl

if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "[ERROR] Shared venv not found at $VENV_DIR" >&2
    exit 1
fi

source "$VENV_DIR/bin/activate"
echo "[SETUP] Python: $(python --version)"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "[SETUP] OMP_NUM_THREADS=$OMP_NUM_THREADS"

rm -rf build/ dist/ src/*.egg-info
export TMPDIR="$WORKDIR/.pip_tmp_$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
pip install --quiet .
rm -rf "$TMPDIR" build/ dist/
unset TMPDIR

echo "[GPU] Checking GPU availability..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available:  {torch.cuda.is_available()}')
"
nvidia-smi 2>/dev/null || echo "(No GPU driver — running on CPU)"

RESUME_FLAG=""
if ls "$CKPT_DIR"/model*.pt 1>/dev/null 2>&1; then
    echo "[CKPT] Found existing checkpoint — will resume training"
    RESUME_FLAG="--resume"
fi

echo ""
echo "============================================"
echo "[PINN] Training FNN + RAD tuned (k=2, c=0.1)..."
echo "  Model: FNN [2,80,40,20,10,1], tanh, Glorot uniform"
echo "  Config: 10k Adam + 15k L-BFGS"
echo "  Nr=32000, Ni=800, Nb=400"
echo "  Lambda: [100,100,100,1,100,1,1] (paper weights)"
echo "  RAD: period=1000, k=2.0, c=0.1, candidates=100000"
echo "  Checkpoint every 500 iters"
if [ -n "$RESUME_FLAG" ]; then
    echo "  >>> RESUMING from checkpoint <<<"
fi
echo "============================================"
python scripts/run_pinn.py --config "$CONFIG" --checkpoint-every 500 $RESUME_FLAG

echo ""
echo "============================================"
echo "[QNM] Extracting QNMs from PINN output..."
echo "============================================"
python scripts/extract_qnm.py --config "$CONFIG" --source pinn

echo ""
echo "============================================"
echo "Finished: $(date)"
echo "============================================"
