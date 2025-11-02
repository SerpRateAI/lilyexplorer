#!/bin/bash

# Copy VAE v2.6.6 model and files to oceanic-agartha

AGARTHA="/home/utig5/johna/oceanic-agartha"
BHAI="/home/utig5/johna/bhai"

echo "Copying VAE v2.6.6 files to oceanic-agartha..."

# Create checkpoints directory if it doesn't exist
mkdir -p "$AGARTHA/checkpoints"

# Copy v2.6.6 model checkpoint
echo "  - Copying v2.6.6 checkpoint..."
cp "$BHAI/ml_models/checkpoints/vae_gra_v2_6_6_latent10.pth" "$AGARTHA/checkpoints/"

# Copy v2.6.6 training script
echo "  - Copying training script..."
cp "$BHAI/train_vae_v2_6_6.py" "$AGARTHA/"

# Copy v2.6.6 clustering results
echo "  - Copying clustering results..."
cp "$BHAI/vae_v2_6_6_clustering_results.csv" "$AGARTHA/"

# Copy v2.6.6 training log
echo "  - Copying training log..."
cp "$BHAI/vae_v2_6_6_training.log" "$AGARTHA/"

# Copy latent dimensionality experiment files (for reference)
echo "  - Copying latent dimensionality experiment..."
cp "$BHAI/test_latent_dimensionality.py" "$AGARTHA/"
cp "$BHAI/latent_dim_test.log" "$AGARTHA/"
cp "$BHAI/latent_dimensionality_comparison.csv" "$AGARTHA/"

echo ""
echo "✓ Copy complete!"
echo ""
echo "Files copied to $AGARTHA:"
echo "  checkpoints/vae_gra_v2_6_6_latent10.pth"
echo "  train_vae_v2_6_6.py"
echo "  vae_v2_6_6_clustering_results.csv"
echo "  vae_v2_6_6_training.log"
echo "  test_latent_dimensionality.py"
echo "  latent_dim_test.log"
echo "  latent_dimensionality_comparison.csv"
