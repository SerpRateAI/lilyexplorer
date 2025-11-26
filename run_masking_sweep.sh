#!/bin/bash
# Masking Hyperparameter Sweep - Parallel Execution Across 4 GPUs

# Create checkpoint directory
mkdir -p ml_models/checkpoints/masking_sweep

# Remove old results file if exists
rm -f masking_sweep_results.csv

# Array of masking percentages (0% to 50% in 1% increments)
mask_probs=()
for i in $(seq 0 50); do
    mask_probs+=($(echo "scale=2; $i/100" | bc))
done

# Function to run training on specific GPU
run_on_gpu() {
    local gpu_id=$1
    local mask_prob=$2
    CUDA_VISIBLE_DEVICES=$gpu_id python3 -u train_vae_masking_sweep.py $mask_prob
}

export -f run_on_gpu

# Progress tracking
total_jobs=${#mask_probs[@]}
completed=0

echo "Starting masking sweep: ${total_jobs} jobs across 4 GPUs"
echo "Mask probabilities: 0.00 to 0.50 (1% increments)"
echo "=========================================="

# Launch jobs in parallel (4 at a time, one per GPU)
for mask_prob in "${mask_probs[@]}"; do
    # Wait for a GPU to become available if all 4 are busy
    while [ $(jobs -r | wc -l) -ge 4 ]; do
        sleep 1
    done

    # Assign to GPU (round-robin)
    gpu_id=$((completed % 4))

    # Launch training in background
    run_on_gpu $gpu_id $mask_prob &

    ((completed++))
    echo "[$completed/$total_jobs] Launched: mask_prob=$mask_prob on GPU $gpu_id"
done

# Wait for all jobs to complete
wait

echo "=========================================="
echo "Sweep complete! Results saved to masking_sweep_results.csv"
echo "Generating visualization..."

# Generate plot
python3 plot_masking_sweep_results.py

echo "Done! Plot saved to masking_sweep_results.png"
