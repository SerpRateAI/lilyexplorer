#!/bin/bash
# Monitor masking sweep progress

echo "Masking Hyperparameter Sweep Progress"
echo "======================================"

# Count completed jobs
if [ -f masking_sweep_results.csv ]; then
    completed=$(tail -n +2 masking_sweep_results.csv 2>/dev/null | wc -l)
else
    completed=0
fi

total=51
pct=$(echo "scale=1; $completed * 100 / $total" | bc)

echo "Completed: $completed / $total ($pct%)"
echo ""

# Show running jobs
running=$(ps aux | grep "train_vae_masking_sweep.py" | grep -v grep | wc -l)
echo "Currently training: $running models"
echo ""

# Show recent completions
if [ -f masking_sweep_results.csv ] && [ $completed -gt 0 ]; then
    echo "Most recent completions:"
    tail -5 masking_sweep_results.csv | awk -F',' '{printf "  mask_prob=%.2f: R²=[%.3f, %.3f, %.3f, %.3f, %.3f, %.3f]\n", $1, $2, $3, $4, $5, $6, $7}'
fi
