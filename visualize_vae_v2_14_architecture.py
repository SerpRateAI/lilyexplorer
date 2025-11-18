"""
Visualize Semi-Supervised VAE v2.14 Architecture
Shows the classification head branching from latent space
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
import numpy as np

# Set font to monospace (DejaVu Sans Mono is typically available on Linux)
plt.rcParams['font.family'] = 'monospace'

class SemiSupervisedVAE(nn.Module):
    """VAE v2.14 semi-supervised architecture"""

    def __init__(self, input_dim=6, latent_dim=10, n_classes=139,
                 encoder_dims=[32, 16], decoder_dims=[16, 32], classifier_hidden=32):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_classes = n_classes

        # Encoder (same as v2.6.7)
        layers = []
        prev_dim = input_dim
        for h_dim in encoder_dims:
            layers.extend([nn.Linear(prev_dim, h_dim), nn.ReLU()])
            prev_dim = h_dim
        self.encoder = nn.Sequential(*layers)

        self.fc_mu = nn.Linear(encoder_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(encoder_dims[-1], latent_dim)

        # Decoder (same as v2.6.7)
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in decoder_dims:
            decoder_layers.extend([nn.Linear(prev_dim, h_dim), nn.ReLU()])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(decoder_dims[-1], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        # Classification head (NEW)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(classifier_hidden, n_classes)
        )

def create_architecture_diagram():
    """Create a detailed architecture diagram for semi-supervised VAE"""

    fig, ax = plt.subplots(figsize=(36, 16))
    ax.set_xlim(-1, 35)
    ax.set_ylim(0, 17)
    ax.axis('off')

    # Color scheme
    color_input = '#E8F4F8'
    color_scaler = '#D4E6F1'
    color_encoder = '#B3D9E6'
    color_latent = '#FFD700'
    color_decoder = '#FFB3BA'
    color_output = '#BAFFC9'
    color_classifier = '#E1BEE7'  # Light purple for classification head
    color_loss = '#FFCCBC'

    # Helper function to draw box
    def draw_box(x, y, width, height, color, label, details='', fontsize=20):
        box = FancyBboxPatch((x, y), width, height,
                             boxstyle="round,pad=0.05",
                             edgecolor='black', facecolor=color, linewidth=2)
        ax.add_patch(box)
        ax.text(x + width/2, y + height/2 + 0.2, label,
                ha='center', va='center', fontsize=fontsize, weight='bold')
        if details:
            ax.text(x + width/2, y + height/2 - 0.3, details,
                    ha='center', va='center', fontsize=fontsize-4, style='italic')

    # Helper function to draw trapezoid (encoder: narrows, decoder: widens)
    def draw_trapezoid(x, y, width, height, color, label, details='', fontsize=20, direction='narrow'):
        """
        direction: 'narrow' for encoder (wider left, narrower right)
                   'widen' for decoder (narrower left, wider right)
        """
        center_y = y + height/2
        if direction == 'narrow':
            # Encoder: wider on left, narrower on right
            left_top = height * 0.8
            left_bottom = height * 0.8
            right_top = height * 0.5
            right_bottom = height * 0.5
        else:  # widen
            # Decoder: narrower on left, wider on right
            left_top = height * 0.5
            left_bottom = height * 0.5
            right_top = height * 0.8
            right_bottom = height * 0.8

        vertices = [
            (x, center_y - left_bottom/2),           # bottom left
            (x, center_y + left_top/2),              # top left
            (x + width, center_y + right_top/2),     # top right
            (x + width, center_y - right_bottom/2),  # bottom right
        ]
        trapezoid = Polygon(vertices, closed=True, edgecolor='black',
                           facecolor=color, linewidth=2)
        ax.add_patch(trapezoid)
        ax.text(x + width/2, center_y + 0.1, label,
                ha='center', va='center', fontsize=fontsize, weight='bold')
        if details:
            ax.text(x + width/2, center_y - 0.4, details,
                    ha='center', va='center', fontsize=fontsize-4, style='italic')

    # Helper function to draw arrow
    def draw_arrow(x1, y1, x2, y2, label='', color='black', linewidth=2):
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                               arrowstyle='->', mutation_scale=20,
                               color=color, linewidth=linewidth)
        ax.add_patch(arrow)
        if label:
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x, mid_y + 0.3, label, ha='center', fontsize=14,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Main VAE path (vertical center at y=10)
    main_y = 10.0

    # 1. Raw Input
    draw_box(0.0, main_y - 1.25, 2.0, 2.5, color_input, 'Raw Input\n6D Features',
             'GRA, MS, NGR,\nR, G, B', fontsize=16)

    # 2. Scaler (forward)
    draw_box(2.5, main_y - 1.5, 2.4, 3.0, color_scaler, 'Distribution-\nAware Scaler',
             'Sign×log\n+ Median-IQR', fontsize=16)
    draw_arrow(2.0, main_y, 2.5, main_y)

    # 3. Scaled Input
    draw_box(5.5, main_y - 1.25, 2.0, 2.5, color_input, 'Scaled\nInput',
             '6D\n(normalized)', fontsize=16)
    draw_arrow(4.9, main_y, 5.5, main_y)

    # 4. Combined Encoder (6 -> 32 -> 16 -> 10)
    draw_trapezoid(8.5, main_y - 1.75, 4.5, 3.5, color_encoder, 'Encoder\n6 → [32,16] → 10',
                   '(fc_mu, fc_logvar)', fontsize=18, direction='narrow')
    draw_arrow(7.5, main_y, 8.5, main_y)

    # 5. Latent z (reparameterization)
    latent_x = 14.0
    draw_box(latent_x, main_y - 1.25, 2.2, 2.5, color_latent, 'Latent z',
             '10D\n(μ + σε)', fontsize=18)
    draw_arrow(13.0, main_y, latent_x, main_y)

    # ========================================================================
    # MAIN PATH: Decoder → Inverse Scaler → Output
    # ========================================================================

    # 6. Decoder (10 -> 16 -> 32 -> 6)
    decoder_x = 17.0
    draw_trapezoid(decoder_x, main_y - 1.75, 4.5, 3.5, color_decoder, 'Decoder\n10 → [16,32] → 6',
                   'ReLU+ReLU+Linear', fontsize=18, direction='widen')
    draw_arrow(latent_x + 2.2, main_y, decoder_x, main_y)

    # 7. Inverse scaler
    inv_scaler_x = 22.5
    draw_box(inv_scaler_x, main_y - 1.5, 2.4, 3.0, color_scaler, 'Inverse\nScaler',
             'Denormalize\nto original\nscale', fontsize=16)
    draw_arrow(decoder_x + 4.5, main_y, inv_scaler_x, main_y)

    # 8. Final reconstructed output
    final_x = 26.0
    draw_box(final_x, main_y - 1.25, 2.2, 2.5, color_output, 'Reconstructed\nOutput',
             'GRA\', MS\',\nNGR\', R\',\nG\', B\'', fontsize=16)
    draw_arrow(inv_scaler_x + 2.4, main_y, final_x, main_y)

    # ========================================================================
    # CLASSIFICATION HEAD (branches downward from latent)
    # ========================================================================

    class_y = 5.5  # Below the main path

    # Arrow from latent z down to classification head
    draw_arrow(latent_x + 1.1, main_y - 1.25, latent_x + 1.1, class_y + 1.1,
               '10D', color='#7B1FA2', linewidth=3)

    # Classification layer 1: Linear(10 -> 32) + ReLU (trapezoid expanding)
    class_x1 = 13.0
    draw_trapezoid(class_x1 - 1.5, class_y - 0.8, 3.0, 1.6, color_classifier,
                   'Linear(10→32)\n+ ReLU', fontsize=16, direction='widen')
    draw_arrow(latent_x + 1.1, class_y + 1.1, class_x1 - 1.5, class_y + 0.0,
               '', color='#7B1FA2', linewidth=2)

    # Dropout layer
    class_x2 = 15.5
    draw_box(class_x2, class_y - 0.8, 2.5, 1.6, color_classifier, 'Dropout\n(p=0.2)',
             fontsize=16)
    draw_arrow(class_x1 + 1.5, class_y + 0.0, class_x2, class_y + 0.0,
               '32D', color='#7B1FA2', linewidth=2)

    # Classification output: Linear(32 -> 139)
    class_x3 = 19.0
    draw_box(class_x3, class_y - 0.8, 3.0, 1.6, color_classifier, 'Linear(32→139)\nLogits',
             fontsize=16)
    draw_arrow(class_x2 + 2.5, class_y + 0.0, class_x3, class_y + 0.0,
               '32D', color='#7B1FA2', linewidth=2)

    # Final classification predictions
    class_x4 = 23.5
    draw_box(class_x4, class_y - 0.8, 2.5, 1.6, color_classifier, 'Lithology\nPredictions',
             '139 classes', fontsize=16)
    draw_arrow(class_x3 + 3.0, class_y + 0.0, class_x4, class_y + 0.0,
               '', color='#7B1FA2', linewidth=2)

    # ========================================================================
    # LOSS FUNCTIONS (bottom)
    # ========================================================================

    loss_y = 1.8

    # Reconstruction loss
    draw_box(1.0, loss_y, 3.5, 1.2, color_loss, 'L_recon = MSE', fontsize=14)
    # Arrow from input (goes down around left side)
    draw_arrow(1.0, main_y - 1.25, 1.0, loss_y + 1.2, '', color='#555', linewidth=1)
    # Arrow from output (goes down around right side)
    draw_arrow(final_x + 1.1, main_y - 1.25, final_x + 1.1, 3.5, '', color='#555', linewidth=1)
    draw_arrow(final_x + 1.1, 3.5, 4.5, 3.5, '', color='#555', linewidth=1)
    draw_arrow(4.5, 3.5, 4.5, loss_y + 1.2, '', color='#555', linewidth=1)

    # KL divergence loss
    draw_box(5.5, loss_y, 3.5, 1.2, color_loss, 'L_KL = β×KL',
             'β: 1e-10→0.75', fontsize=14)
    # Arrow from latent (goes down left side of decoder)
    draw_arrow(latent_x - 1.1, main_y, latent_x - 1.1, loss_y + 1.2, '', color='#555', linewidth=1)

    # Classification loss
    draw_box(10.0, loss_y, 4.0, 1.2, color_loss, 'L_class = α×CE',
             'α = 0.1 (fixed)', fontsize=14)
    draw_arrow(class_x4 + 1.25, class_y + 0.0, class_x4 + 1.25, loss_y + 1.2, '', color='#7B1FA2', linewidth=1)

    # Total loss
    draw_box(15.5, loss_y, 7.0, 1.2, color_loss,
             'L_total = L_recon + β×L_KL + α×L_class',
             fontsize=16)
    draw_arrow(4.5, loss_y + 0.6, 15.5, loss_y + 0.6, '', color='black', linewidth=2)
    draw_arrow(9.0, loss_y + 0.6, 15.5, loss_y + 0.6, '', color='black', linewidth=2)
    draw_arrow(14.0, loss_y + 0.6, 15.5, loss_y + 0.6, '', color='black', linewidth=2)

    # ========================================================================
    # ANNOTATIONS
    # ========================================================================

    # Add title
    ax.text(14, 16.2, 'Semi-Supervised VAE v2.14 Architecture (α=0.1)',
            ha='center', fontsize=28, weight='bold')
    ax.text(14, 15.5, 'Guided Representation Learning with Classification Head',
            ha='center', fontsize=20, style='italic', color='#555')

    plt.tight_layout()
    plt.savefig('vae_v2_14_semisupervised_architecture_diagram.png', dpi=300, bbox_inches='tight')
    print("✓ Diagram saved as 'vae_v2_14_semisupervised_architecture_diagram.png'")
    plt.close()

if __name__ == "__main__":
    create_architecture_diagram()
    print("\n" + "="*70)
    print("Semi-Supervised VAE v2.14 Architecture Diagram Complete!")
    print("="*70)
    print("Shows:")
    print("  • Encoder: 6D → [32, 16] → 10D latent (same as v2.6.7)")
    print("  • Decoder: 10D → [16, 32] → 6D (same as v2.6.7)")
    print("  • Classification head: 10D → [32, ReLU, Dropout(0.2)] → 139 classes")
    print("  • Three-part loss: L_recon + β×L_KL + α×L_class")
    print("  • β annealing: 1e-10 → 0.75 (over 50 epochs)")
    print("  • α = 0.1 (fixed, optimal from grid search)")
    print("  • Performance: ARI=0.285 (+45.6% vs v2.6.7)")
    print("  • Parameters: 6,949 (vs 2,010 for v2.6.7)")
