# Compatibility wrapper for loading v3 checkpoints
# This makes the module loadable by torch
import sys
import os

# Add ml_models to path
ml_models_path = os.path.join(os.path.dirname(__file__), 'ml_models')
if ml_models_path not in sys.path:
    sys.path.insert(0, ml_models_path)

# Import the actual module components
import importlib.util
spec = importlib.util.spec_from_file_location(
    "vae_gra_v3_actual",
    os.path.join(ml_models_path, 'vae_lithology_gra_v3_model.py')
)
vae_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vae_module)

# Export the classes
DualEncoderVAE = vae_module.DualEncoderVAE
DistributionAwareScaler = vae_module.DistributionAwareScaler
LithologyDataset = vae_module.LithologyDataset
vae_loss = vae_module.vae_loss

__all__ = ['DualEncoderVAE', 'DistributionAwareScaler', 'LithologyDataset', 'vae_loss']
