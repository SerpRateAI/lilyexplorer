import torch
import sys
sys.path.insert(0, '/home/utig5/johna/bhai/ml_models')
from vae_lithology_gra_v2_5_model import VAE, DistributionAwareScaler

print("Testing checkpoint load...")

try:
    checkpoint = torch.load('ml_models/checkpoints/vae_gra_v2_6_6_latent10.pth')
    print("✓ Checkpoint loaded successfully")
    print(f"Keys: {checkpoint.keys()}")

    model = VAE(input_dim=6, latent_dim=10, hidden_dims=[32, 16])
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✓ Model state loaded successfully")

    scaler = checkpoint['scaler']
    print(f"✓ Scaler loaded: {type(scaler)}")
    print(f"Scaler attributes: {dir(scaler)}")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
