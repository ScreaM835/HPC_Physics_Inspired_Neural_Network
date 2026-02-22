#!/bin/bash
# ============================================================
# CSD3 SLURM Submission Script — QNM PINN Training
# ============================================================
# Submit with:  sbatch scripts/csd3_submit.sh
# Monitor with: squeue -u $USER
# Cancel with:  scancel <jobid>
# Check output: tail -f slurm-<jobid>.out
# ============================================================

#SBATCH --job-name=qnm_pinn
#SBATCH --output=qnm_pinn_%j.out
#SBATCH --error=qnm_pinn_%j.err

#! --- CHANGE THIS to your GPU project name ---
#SBATCH --account=CHANGEME-GPU

#! Ampere A100 GPU partition
#SBATCH --partition=ampere

#! Request 1 GPU (A100 80GB) — sufficient for this single-model training
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

#! Wallclock time: 2 hours should be plenty with A100 FP64 performance
#! (The A100 has ~19.5 TFLOPS FP64 vs ~0.5 on consumer GPUs)
#! Adjust upward if needed; max is 36h for SL1/SL2
#SBATCH --time=02:00:00

# ============================================================
# Environment Setup
# ============================================================
module purge
module load rhel8/default-amp

# Activate the virtual environment created by csd3_setup.sh
source ~/project32_qnm_pinn/venv_csd3/bin/activate

# Move to project directory
cd ~/project32_qnm_pinn

# ============================================================
# Pre-flight checks
# ============================================================
echo "============================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURM_NODELIST"
echo "Started:       $(date)"
echo "Working dir:   $(pwd)"
echo "============================================"

python -c "
import torch
print(f'PyTorch:       {torch.__version__}')
print(f'CUDA:          {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU:           {torch.cuda.get_device_name(0)}')
    print(f'VRAM:          {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
    print(f'FP64 capable:  Yes (A100 has 19.5 TFLOPS FP64)')
"

echo "============================================"
echo "Starting PINN training with causal weighting"
echo "Config: configs/zerilli_l2.yaml"
echo "============================================"

# ============================================================
# Run Training
# ============================================================
python scripts/run_pinn.py --config configs/zerilli_l2.yaml

echo "============================================"
echo "Training complete: $(date)"
echo "============================================"

# ============================================================
# Optional: Run QNM extraction on PINN output
# ============================================================
# Uncomment the following line to automatically extract QNMs after training:
# python scripts/extract_qnm.py --config configs/zerilli_l2.yaml --source pinn

echo "Outputs saved to: outputs/pinn/zerilli_l2/"
echo "Download results with:"
echo "  scp -r <CRSid>@login-cpu.hpc.cam.ac.uk:~/project32_qnm_pinn/outputs/pinn/zerilli_l2/ ."
