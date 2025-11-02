# Development Environment Setup

This document describes the development environment, dependencies, and system-specific configurations for the LILY Database project.

## Python Setup

- **Python version**: 3.11+ (see `.python-version`)
- **Project configuration**: `pyproject.toml`
- **Main entry point**: `main.py` (currently minimal)

## Python Dependencies

Core dependencies defined in `pyproject.toml`:

**Data manipulation**:
- pandas, numpy, xarray, netcdf4

**Visualization**:
- matplotlib, seaborn, hvplot, datashader, cartopy

**Scientific computing**:
- scipy

**Machine learning**:
- scikit-learn, catboost, torch

**Development**:
- jupyter

## GPU Setup for PyTorch Training

**Important:** GPU acceleration only available on **cotopaxi**, not smokey.

### Current Environment

- PyTorch 2.8.0 with CUDA 12.8 support installed
- On **smokey**: No NVIDIA driver loaded, `torch.cuda.is_available()` returns False
- On **cotopaxi**: NVIDIA GPU available, can use `nvidia-smi` directly (no slurm needed)
- Both `/home/utig5/johna` and `/home/other/johna` are the same NFS mount (utig5.ig.utexas.edu:/bigpool/home)

### Running VAE Training on GPU

1. **Log into cotopaxi:**
```bash
ssh cotopaxi
cd /home/other/johna/bhai  # or /home/utig5/johna/bhai (same filesystem)
```

2. **Verify GPU access:**
```bash
nvidia-smi
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

3. **Run training:**
```bash
# Background execution with logging
python3 train_beta_annealing.py > beta_annealing.log 2>&1 &

# Monitor progress
tail -f beta_annealing.log
```

### Automatic GPU Detection

All VAE training scripts include automatic GPU detection:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

If running on smokey, training will use CPU (significantly slower but functional for testing).

## UMAP Visualization Issues

### Known Issue on smokey

UMAP import fails due to numba/numpy version conflicts:
```
SystemError: initialization of _internal failed without raising an exception
```

### Workarounds

1. Run analysis notebooks on **cotopaxi** (where UMAP works)
2. Use Jupyter on a different machine with properly configured umap-learn
3. For quick testing, notebooks can fall back to PCA (less effective visualization)

### Affected Notebooks

- `vae_v2_6_7_analysis.ipynb` - **Production model analysis** (10D latent space visualization)
- `vae_v2_6_6_analysis.ipynb` - v2.6.6 analysis (10D latent space visualization)
- Other VAE pipeline notebooks with UMAP projections

The notebooks are fully functional on systems with working UMAP installations.

## System-Specific Notes

### Smokey (CPU-only)
- Good for: Dataset creation, analysis scripts, notebook development
- Limitations: No GPU, UMAP import issues
- Use for: Quick tests, data exploration

### Cotopaxi (GPU + UMAP)
- Good for: VAE training, full notebook analysis with UMAP
- Hardware: NVIDIA GPU with CUDA 12.8
- Use for: Production training, complete visualizations

### Shared Filesystem
Both systems share the same NFS mount, so:
- Code changes on one system are immediately visible on the other
- Model checkpoints can be trained on cotopaxi and analyzed on smokey
- No need to copy files between systems
