#!/bin/bash
# ============================================================
# SLURM Job: Setup venv + Run refined FD + FD comparison
# ============================================================
#SBATCH --job-name=qnm_fd
#SBATCH --output=qnm_fd_%j.out
#SBATCH --error=qnm_fd_%j.err
#SBATCH --account=fergusson-sl3-cpu
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00

set -e

echo "============================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURM_NODELIST"
echo "Started:       $(date)"
echo "============================================"

cd /home/ycc44/project32_qnm_pinn_repo_fd_refinement/project32_qnm_pinn

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

echo "[SETUP] Installing dependencies..."
pip install --quiet .

echo "[SETUP] Installed packages:"
pip list | grep -iE "torch|numpy|scipy|matplotlib|tqdm|yaml"

# --- Run refined FD ---
echo ""
echo "============================================"
echo "[FD] Running refined FD solver (dx=0.1, dt=0.05)..."
echo "============================================"
python scripts/run_fd.py --config configs/zerilli_l2_fd_refined.yaml

# --- Run FD comparison ---
echo ""
echo "============================================"
echo "[FD] Running FD refinement comparison..."
echo "============================================"
python scripts/compare_fd_refinement.py \
  --fd_coarse  outputs/fd/zerilli_l2_fd.npz \
  --fd_refined outputs/fd/zerilli_l2_fd_refined_fd.npz \
  --times 10,20,30,40 \
  --xlim -20 20

# --- Extract QNMs from refined FD ---
echo ""
echo "============================================"
echo "[QNM] Extracting QNMs from refined FD..."
echo "============================================"
python scripts/extract_qnm.py --config configs/zerilli_l2_fd_refined.yaml --source fd

echo ""
echo "============================================"
echo "Finished: $(date)"
echo "============================================"
