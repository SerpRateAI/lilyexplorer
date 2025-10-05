#!/bin/bash
#
# Script to generate all paper plots from the LILY database
# This script runs all Python figure generation scripts in the paper_plots_code directory
#

echo "========================================="
echo "Generating Paper Plots from LILY Database"
echo "========================================="
echo ""

# Change to the repository root directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "Working directory: $(pwd)"
echo ""

# Check if required directories exist
if [ ! -d "paper_plots_code" ]; then
    echo "Error: paper_plots_code directory not found"
    exit 1
fi

if [ ! -d "datasets" ]; then
    echo "Error: datasets directory not found"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p paper_plots

# Count total figures
total_figures=$(ls paper_plots_code/figure_*.py 2>/dev/null | wc -l)
echo "Found $total_figures figure scripts to run"
echo ""

# Initialize counters
success_count=0
error_count=0

# Run each figure script from the repo root
for script in paper_plots_code/figure_*.py; do
    if [ -f "$script" ]; then
        figure_name=$(basename "$script" .py)
        echo "----------------------------------------"
        echo "Running: $figure_name"
        echo "----------------------------------------"

        # Run the Python script from the repo root
        if python3 "$script"; then
            echo "✓ $figure_name completed successfully"
            ((success_count++))
        else
            echo "✗ $figure_name failed with error code $?"
            ((error_count++))
        fi
        echo ""
    fi
done

echo "========================================="
echo "Summary"
echo "========================================="
echo "Total figures: $total_figures"
echo "Successful: $success_count"
echo "Errors: $error_count"
echo ""

# List generated plots
if [ -d "paper_plots" ]; then
    plot_count=$(ls paper_plots/*.png 2>/dev/null | wc -l)
    echo "Generated $plot_count PNG files in paper_plots/"
    echo ""
    if [ $plot_count -gt 0 ]; then
        echo "Generated files:"
        ls -lh paper_plots/*.png
    else
        echo "No PNG files found"
    fi
fi

echo ""
echo "========================================="
echo "Plot generation complete!"
echo "========================================="

# Exit with error if any scripts failed
if [ $error_count -gt 0 ]; then
    exit 1
fi

exit 0
