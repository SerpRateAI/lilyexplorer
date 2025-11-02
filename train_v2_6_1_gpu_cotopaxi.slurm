#!/bin/bash
#SBATCH --job-name=vae_v2_6_1_gpu
#SBATCH --output=vae_v2_6_1_training_gpu.log
#SBATCH --error=vae_v2_6_1_training_gpu.err
#SBATCH --nodelist=cotopaxi
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo

# Check GPU
echo "=========================================="
echo "GPU Information"
echo "=========================================="
nvidia-smi
echo

# Run training
echo "=========================================="
echo "Starting VAE v2.6.1 Training on GPU"
echo "=========================================="
cd /home/utig5/johna/bhai
python3 train_v2_6_1.py

echo
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
