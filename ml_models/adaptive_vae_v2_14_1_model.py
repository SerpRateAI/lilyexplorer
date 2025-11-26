#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adaptive VAE v2.14.1 - Transformer-based encoder for variable feature inputs

Architecture:
- Feature embeddings: Each of 6 features gets learned embedding (dim=16)
- Input: variable-length sequence of (feature_id, value) pairs
- Transformer encoder: Self-attention learns feature relationships
- Output: 10D latent space
- Multi-decoder: Reconstructs only present features
- Classification head: 10D → 139 lithology classes

Key innovation: Can train on ALL boreholes regardless of missing measurements
"""

import torch
import torch.nn as nn
import math


class FeatureEmbedding(nn.Module):
    """Learned embeddings for each feature type + value projection"""
    def __init__(self, n_features=6, embed_dim=16):
        super().__init__()
        self.embed_dim = embed_dim

        # Learned embedding for each feature type (GRA, MS, NGR, R, G, B)
        self.feature_type_embed = nn.Embedding(n_features, embed_dim)

        # Project normalized feature value to embedding space
        self.value_proj = nn.Linear(1, embed_dim)

    def forward(self, feature_ids, feature_values):
        """
        Args:
            feature_ids: (batch, seq_len) - which features are present [0-5]
            feature_values: (batch, seq_len, 1) - normalized values
        Returns:
            embeddings: (batch, seq_len, embed_dim)
        """
        # Get type embeddings
        type_embeds = self.feature_type_embed(feature_ids)  # (batch, seq_len, embed_dim)

        # Get value embeddings
        value_embeds = self.value_proj(feature_values)  # (batch, seq_len, embed_dim)

        # Combine: type + value
        return type_embeds + value_embeds


class TransformerEncoder(nn.Module):
    """Transformer encoder for variable-length feature sequences"""
    def __init__(self, embed_dim=16, n_heads=4, n_layers=2, latent_dim=10):
        super().__init__()
        self.embed_dim = embed_dim

        # Positional encoding (learnable, since order doesn't matter much)
        self.pos_encoding = nn.Parameter(torch.randn(6, embed_dim))  # max 6 features

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=64,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Pooling: aggregate sequence to single vector
        self.pool = nn.Linear(embed_dim, embed_dim)

        # Project to latent parameters
        self.fc_mu = nn.Linear(embed_dim, latent_dim)
        self.fc_logvar = nn.Linear(embed_dim, latent_dim)

    def forward(self, embeddings, mask=None):
        """
        Args:
            embeddings: (batch, seq_len, embed_dim)
            mask: (batch, seq_len) - True for valid positions
        Returns:
            mu, logvar: (batch, latent_dim)
        """
        batch_size, seq_len, _ = embeddings.shape

        # Add positional encoding
        pos_enc = self.pos_encoding[:seq_len].unsqueeze(0)  # (1, seq_len, embed_dim)
        x = embeddings + pos_enc

        # Create attention mask (True = ignore, False = attend)
        if mask is not None:
            attn_mask = ~mask  # Invert: True positions should be attended to
        else:
            attn_mask = None

        # Transformer encoding
        encoded = self.transformer(x, src_key_padding_mask=attn_mask)  # (batch, seq_len, embed_dim)

        # Global pooling: mean over sequence (only valid positions)
        if mask is not None:
            # Masked mean
            mask_expanded = mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
            pooled = (encoded * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            pooled = encoded.mean(dim=1)  # (batch, embed_dim)

        # Apply pooling layer
        pooled = torch.relu(self.pool(pooled))

        # Latent parameters
        mu = self.fc_mu(pooled)
        logvar = self.fc_logvar(pooled)

        return mu, logvar


class FeatureDecoder(nn.Module):
    """Separate decoder for each feature type"""
    def __init__(self, latent_dim=10, hidden_dim=32):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, z):
        return self.decoder(z)


class AdaptiveVAE(nn.Module):
    """
    Adaptive VAE v2.14.1 with transformer encoder

    Can handle any subset of the 6 features:
    0: GRA (bulk density)
    1: MS (magnetic susceptibility)
    2: NGR (natural gamma radiation)
    3: R (red channel)
    4: G (green channel)
    5: B (blue channel)
    """
    def __init__(self, n_features=6, embed_dim=16, n_heads=4, n_layers=2,
                 latent_dim=10, n_classes=139, decoder_hidden=32):
        super().__init__()
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.n_classes = n_classes

        # Feature embedding
        self.feature_embed = FeatureEmbedding(n_features, embed_dim)

        # Transformer encoder
        self.encoder = TransformerEncoder(embed_dim, n_heads, n_layers, latent_dim)

        # Multi-decoder: separate decoder for each feature
        self.decoders = nn.ModuleList([
            FeatureDecoder(latent_dim, decoder_hidden) for _ in range(n_features)
        ])

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, n_classes)
        )

    def encode(self, feature_ids, feature_values, mask):
        """
        Args:
            feature_ids: (batch, max_seq_len) - feature indices [0-5], padded with 0
            feature_values: (batch, max_seq_len, 1) - normalized values, padded with 0
            mask: (batch, max_seq_len) - True for valid positions
        Returns:
            mu, logvar: (batch, latent_dim)
        """
        # Get embeddings
        embeddings = self.feature_embed(feature_ids, feature_values)

        # Encode to latent
        mu, logvar = self.encoder(embeddings, mask)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, feature_ids):
        """
        Decode latent to reconstruct features

        Args:
            z: (batch, latent_dim)
            feature_ids: (batch, seq_len) - which features to reconstruct
        Returns:
            reconstructions: (batch, seq_len, 1)
        """
        batch_size, seq_len = feature_ids.shape

        # Reconstruct each feature
        recons = []
        for i in range(seq_len):
            feat_id = feature_ids[:, i]  # (batch,)
            # Apply corresponding decoder
            recon_i = torch.stack([
                self.decoders[fid](z[j].unsqueeze(0))
                for j, fid in enumerate(feat_id)
            ]).squeeze(1)  # (batch, 1)
            recons.append(recon_i)

        return torch.stack(recons, dim=1)  # (batch, seq_len, 1)

    def forward(self, feature_ids, feature_values, mask):
        """
        Forward pass

        Args:
            feature_ids: (batch, seq_len) - which features are present
            feature_values: (batch, seq_len, 1) - normalized values
            mask: (batch, seq_len) - True for valid positions
        Returns:
            reconstructions, mu, logvar, logits
        """
        # Encode
        mu, logvar = self.encode(feature_ids, feature_values, mask)

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Decode (only for present features)
        recons = self.decode(z, feature_ids)

        # Classify
        logits = self.classifier(z)

        return recons, mu, logvar, logits

    def get_embedding(self, feature_ids, feature_values, mask):
        """Get latent embedding (deterministic, using mu only)"""
        mu, _ = self.encode(feature_ids, feature_values, mask)
        return mu


# Loss function
def adaptive_vae_loss(recons, targets, mu, logvar, logits, labels,
                      mask, beta=0.75, alpha=0.1):
    """
    Loss for adaptive VAE

    Args:
        recons: (batch, seq_len, 1) - reconstructed values
        targets: (batch, seq_len, 1) - true values
        mu, logvar: (batch, latent_dim) - latent parameters
        logits: (batch, n_classes) - classification logits
        labels: (batch,) - lithology labels
        mask: (batch, seq_len) - True for valid positions
        beta: KL weight
        alpha: classification weight
    """
    # Reconstruction loss (MSE, only on valid positions)
    recon_loss = ((recons - targets) ** 2 * mask.unsqueeze(-1)).sum() / mask.sum()

    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)

    # Classification loss (cross-entropy)
    class_loss = nn.CrossEntropyLoss()(logits, labels)

    # Total loss
    total_loss = recon_loss + beta * kl_loss + alpha * class_loss

    return total_loss, recon_loss, kl_loss, class_loss
