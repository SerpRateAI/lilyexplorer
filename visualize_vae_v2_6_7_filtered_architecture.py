"""
Visualize VAE GRA v2.6.7 FILTERED Architecture
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

class VAE(nn.Module):
    """VAE v2.6.7 architecture (same as v2.6.6, simplified - no BatchNorm/Dropout)"""

    def __init__(self, input_dim=6, latent_dim=10, hidden_dims=[32, 16]):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder (simplified - no BatchNorm/Dropout)
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU()
            ])
            prev_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Latent space
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        # Decoder (simplified - no BatchNorm/Dropout)
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU()
            ])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def create_architecture_diagram():
    """Create a detailed architecture diagram"""

    fig, ax = plt.subplots(figsize=(18, 10))
    ax.set_xlim(-0.5, 17.5)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Color scheme
    color_input = '#E8F4F8'
    color_encoder = '#B3D9E6'
    color_latent = '#FFD700'
    color_decoder = '#FFB3BA'
    color_output = '#BAFFC9'

    # Layer positions (x, y, width, height)
    layers = []

    # Input layer
    layers.append({
        'name': 'Input\n6D Features',
        'details': 'GRA, MS, NGR,\nR, G, B',
        'pos': (0.5, 4, 1.2, 2),
        'color': color_input
    })

    # Encoder Layer 1
    layers.append({
        'name': 'Linear(6→32)',
        'details': 'ReLU\n(no BatchNorm\nor Dropout)',
        'pos': (2.5, 3.5, 1.5, 3),
        'color': color_encoder
    })

    # Encoder Layer 2
    layers.append({
        'name': 'Linear(32→16)',
        'details': 'ReLU\n(simplified)',
        'pos': (4.8, 4, 1.5, 2),
        'color': color_encoder
    })

    # Latent mu
    layers.append({
        'name': 'fc_mu',
        'details': 'Linear(16→10)\n(6 collapsed)',
        'pos': (7, 6, 1.2, 1.2),
        'color': color_latent
    })

    # Latent logvar
    layers.append({
        'name': 'fc_logvar',
        'details': 'Linear(16→10)\n(6 collapsed)',
        'pos': (7, 3, 1.2, 1.2),
        'color': color_latent
    })

    # Reparameterization
    layers.append({
        'name': 'Reparameterize',
        'details': 'z = μ + σ·ε\nε ~ N(0,I)\n10D → 4D eff.',
        'pos': (9, 4, 1.5, 2),
        'color': color_latent
    })

    # Decoder Layer 1
    layers.append({
        'name': 'Linear(10→16)',
        'details': 'ReLU\n(simplified)',
        'pos': (11.3, 4, 1.5, 2),
        'color': color_decoder
    })

    # Decoder Layer 2
    layers.append({
        'name': 'Linear(16→32)',
        'details': 'ReLU\n(no BatchNorm\nor Dropout)',
        'pos': (13.5, 3.5, 1.5, 3),
        'color': color_decoder
    })

    # Output layer
    layers.append({
        'name': 'Output\nLinear(32→6)',
        'details': 'Reconstructed\nFeatures',
        'pos': (15.3, 4, 1.4, 2),
        'color': color_output
    })

    # Draw layers
    for layer in layers:
        x, y, w, h = layer['pos']
        box = FancyBboxPatch((x, y), w, h,
                            boxstyle="round,pad=0.05",
                            edgecolor='black',
                            facecolor=layer['color'],
                            linewidth=2)
        ax.add_patch(box)

        # Layer name
        ax.text(x + w/2, y + h - 0.3, layer['name'],
               ha='center', va='top', fontsize=10, fontweight='bold')

        # Layer details
        ax.text(x + w/2, y + h/2 - 0.1, layer['details'],
               ha='center', va='center', fontsize=8, style='italic')

    # Draw arrows
    arrows = [
        ((0.5 + 1.2, 5), (2.5, 5)),           # Input → Enc1
        ((2.5 + 1.5, 5), (4.8, 5)),           # Enc1 → Enc2
        ((4.8 + 1.5, 6), (7, 6.6)),           # Enc2 → mu (fixed: top of Enc2 at y=6)
        ((4.8 + 1.5, 4.2), (7, 3.6)),         # Enc2 → logvar
        ((7 + 1.2, 6.6), (9, 5.5)),           # mu → Reparam
        ((7 + 1.2, 3.6), (9, 4.5)),           # logvar → Reparam
        ((9 + 1.5, 5), (11.3, 5)),            # Reparam → Dec1
        ((11.3 + 1.5, 5), (13.5, 5)),         # Dec1 → Dec2
        ((13.5 + 1.5, 5), (15.3, 5)),         # Dec2 → Output
    ]

    for (x1, y1), (x2, y2) in arrows:
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                              arrowstyle='->', mutation_scale=20,
                              linewidth=2, color='black')
        ax.add_patch(arrow)

    # Add title
    ax.text(8, 9.2, 'VAE GRA v2.6.7 FILTERED Architecture',
           ha='center', fontsize=18, fontweight='bold')

    # Add subtitle
    ax.text(8, 8.7, 'β-VAE for Lithology Clustering (238K samples, 296 boreholes, 12 classes)',
           ha='center', fontsize=11, style='italic')

    # Add sections labels
    ax.text(3.5, 7.8, 'ENCODER', ha='center', fontsize=12,
           fontweight='bold', color='#2E86AB')
    ax.text(8, 7.8, 'LATENT SPACE', ha='center', fontsize=12,
           fontweight='bold', color='#A23B72')
    ax.text(12.5, 7.8, 'DECODER', ha='center', fontsize=12,
           fontweight='bold', color='#C73E1D')

    # Add legend
    legend_y = 1.5
    ax.text(0.5, legend_y + 0.5, 'Model Details:', fontsize=10, fontweight='bold')
    ax.text(0.5, legend_y + 0.2, '• Total Parameters: 2,010', fontsize=9)
    ax.text(0.5, legend_y - 0.1, '• Latent Dimension: 10D (4D effective)', fontsize=9)
    ax.text(0.5, legend_y - 0.4, '• Training: β annealing (1e-10→0.75)', fontsize=9)
    ax.text(0.5, legend_y - 0.7, '• Filtered: ≥100 samples per class', fontsize=9)

    ax.text(5, legend_y + 0.5, 'Performance (5-fold CV):', fontsize=10, fontweight='bold')
    ax.text(5, legend_y + 0.2, '• ARI: 0.2134 ± 0.0476', fontsize=9)
    ax.text(5, legend_y - 0.1, '• Improvement vs original: +8.9%', fontsize=9)
    ax.text(5, legend_y - 0.4, '• Removed noise classes', fontsize=9)
    ax.text(5, legend_y - 0.7, '• (Ultramafic n=81, Diamict n=66)', fontsize=9)

    ax.text(10, legend_y + 0.5, 'Input Features (6D):', fontsize=10, fontweight='bold')
    ax.text(10, legend_y + 0.2, '• GRA: Bulk density (g/cm³)', fontsize=9)
    ax.text(10, legend_y - 0.1, '• MS: Magnetic susceptibility', fontsize=9)
    ax.text(10, legend_y - 0.4, '• NGR: Gamma ray counts (cps)', fontsize=9)
    ax.text(10, legend_y - 0.7, '• RGB: Color channels (0-255)', fontsize=9)

    plt.tight_layout(pad=1.5)
    plt.savefig('vae_v2_6_7_filtered_architecture_diagram.png', dpi=300,
                bbox_inches='tight', pad_inches=0.3,
                facecolor='white', edgecolor='none')
    print("✓ Architecture diagram saved: vae_v2_6_7_filtered_architecture_diagram.png")
    plt.close()

if __name__ == "__main__":
    print("="*80)
    print("VAE GRA v2.6.7 FILTERED Architecture Visualization")
    print("="*80)

    # Create custom detailed diagram
    print("\n1. Creating custom architecture diagram...")
    create_architecture_diagram()

    print("\n" + "="*80)
    print("Visualization complete!")
    print("="*80)
