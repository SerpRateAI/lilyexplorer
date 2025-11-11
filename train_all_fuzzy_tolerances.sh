#!/bin/bash
#
# Train VAE v2.6.8 on all fuzzy matching tolerances in parallel
# Uses 4x A100 GPUs on cotopaxi
#

echo "=================================================================================================="
echo "VAE v2.6.8 - PARALLEL FUZZY TOLERANCE TRAINING"
echo "=================================================================================================="
echo ""
echo "Training tolerances: ±10cm, ±20cm, ±50cm, ±1m, ±2m"
echo "Parallel execution on 4 GPUs"
echo ""

# Check if datasets exist
for tol in 10 20 50 100 200; do
    if [ "$tol" -eq 20 ]; then
        # Baseline uses original dataset
        if [ ! -f "vae_training_data_v2_20cm.csv" ]; then
            echo "ERROR: Baseline dataset vae_training_data_v2_20cm.csv not found"
            exit 1
        fi
    else
        dataset="vae_training_data_fuzzy_${tol}cm.csv"
        if [ ! -f "$dataset" ]; then
            echo "ERROR: Dataset $dataset not found"
            echo "Run test_fuzzy_matching_tolerances.py first to create datasets"
            exit 1
        fi
    fi
done

echo "✓ All datasets found"
echo ""

# Create checkpoints directory if it doesn't exist
mkdir -p ml_models/checkpoints

# Train in parallel on different GPUs
echo "Starting parallel training..."
echo ""

# GPU 0: ± 10cm
echo "GPU 0: Training ±10cm tolerance..."
python3 train_vae_v2_6_8_fuzzy_comparison.py --tolerance_cm 10 --gpu_id 0 > vae_v2_6_8_fuzzy_10cm_training.log 2>&1 &
PID1=$!

# GPU 1: ± 20cm (baseline)
echo "GPU 1: Training ±20cm tolerance (fuzzy baseline)..."
python3 train_vae_v2_6_8_fuzzy_comparison.py --tolerance_cm 20 --gpu_id 1 > vae_v2_6_8_fuzzy_20cm_training.log 2>&1 &
PID2=$!

# GPU 2: ± 50cm
echo "GPU 2: Training ±50cm tolerance..."
python3 train_vae_v2_6_8_fuzzy_comparison.py --tolerance_cm 50 --gpu_id 2 > vae_v2_6_8_fuzzy_50cm_training.log 2>&1 &
PID3=$!

# GPU 3: ± 1m
echo "GPU 3: Training ±1m tolerance..."
python3 train_vae_v2_6_8_fuzzy_comparison.py --tolerance_cm 100 --gpu_id 3 > vae_v2_6_8_fuzzy_100cm_training.log 2>&1 &
PID4=$!

echo ""
echo "All training jobs started!"
echo "  PID $PID1: ±10cm on GPU 0"
echo "  PID $PID2: ±20cm on GPU 1"
echo "  PID $PID3: ±50cm on GPU 2"
echo "  PID $PID4: ±1m on GPU 3"
echo ""

# Monitor progress
echo "Monitoring progress (Ctrl+C to stop monitoring, jobs will continue)..."
echo ""

while true; do
    sleep 30
    clear
    echo "=================================================================================================="
    echo "VAE v2.6.8 TRAINING PROGRESS"
    echo "=================================================================================================="
    echo ""
    date
    echo ""

    # Check each log file
    for tol in 10 20 50 100; do
        logfile="vae_v2_6_8_fuzzy_${tol}cm_training.log"
        if [ -f "$logfile" ]; then
            echo "± ${tol}cm:"
            tail -3 "$logfile" | grep -E "(Epoch|Early stopping|completed)" | tail -1 || echo "  Still loading data..."
        fi
    done

    echo ""

    # Check if all jobs finished
    jobs_running=0
    for pid in $PID1 $PID2 $PID3 $PID4; do
        if kill -0 $pid 2>/dev/null; then
            jobs_running=$((jobs_running + 1))
        fi
    done

    if [ $jobs_running -eq 0 ]; then
        echo "✓ All training jobs completed!"
        break
    else
        echo "$jobs_running / 4 jobs still running..."
    fi
done

echo ""
echo "=================================================================================================="
echo "TRAINING COMPLETE"
echo "=================================================================================================="
echo ""

# Train ±2m separately (reuse GPU 0 after ±10cm finishes)
echo "Training ±2m tolerance on GPU 0..."
python3 train_vae_v2_6_8_fuzzy_comparison.py --tolerance_cm 200 --gpu_id 0 > vae_v2_6_8_fuzzy_200cm_training.log 2>&1

echo "✓ ±2m tolerance completed"
echo ""

# Collect all results
echo "Collecting results..."
cat vae_v2_6_8_fuzzy_*cm_results.csv > vae_v2_6_8_all_tolerances_results.csv

echo "✓ Results saved to vae_v2_6_8_all_tolerances_results.csv"
echo ""
echo "=================================================================================================="
echo "DONE"
echo "=================================================================================================="
