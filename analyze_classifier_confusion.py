"""
Analyze why some lithology classes have 0% accuracy.
Show what the classifier is predicting for these misclassified classes.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import sys

sys.path.insert(0, '/home/utig5/johna/bhai/ml_models')
from vae_lithology_gra_v2_5_model import DistributionAwareScaler

# VAE architecture
class VAE(nn.Module):
    def __init__(self, input_dim=6, latent_dim=10, hidden_dims=[32, 16]):
        super().__init__()
        self.latent_dim = latent_dim

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h_dim), nn.ReLU()])
            prev_dim = h_dim
        self.encoder = nn.Sequential(*layers)

        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            layers.extend([nn.Linear(prev_dim, h_dim), nn.ReLU()])
            prev_dim = h_dim
        self.decoder = nn.Sequential(*layers)
        self.fc_out = nn.Linear(hidden_dims[0], input_dim)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

# Classifier architecture
class LithologyClassifier(nn.Module):
    def __init__(self, input_dim=10, num_classes=139, hidden_dims=[64, 32]):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = h_dim

        self.classifier = nn.Sequential(*layers)
        self.fc_out = nn.Linear(hidden_dims[-1], num_classes)

    def forward(self, x):
        h = self.classifier(x)
        return self.fc_out(h)

print("="*100)
print("ANALYZING CLASSIFIER CONFUSION FOR 0% ACCURACY CLASSES")
print("="*100)
print()

# Load data
df = pd.read_csv('vae_training_data_v2_20cm.csv')

feature_cols = ['Bulk density (GRA)', 'Magnetic susceptibility (instr. units)',
                'NGR total counts (cps)', 'R', 'G', 'B']

X = df[feature_cols].values
y = df['Principal'].values

# Create label mapping
unique_labels = sorted(np.unique(y))
label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}
y_encoded = np.array([label_to_idx[label] for label in y])

# Same split as training
unique_boreholes = df['Borehole_ID'].unique()
train_boreholes, test_boreholes = train_test_split(
    unique_boreholes, train_size=0.85, random_state=42
)
train_boreholes, val_boreholes = train_test_split(
    train_boreholes, train_size=0.8235, random_state=42
)

train_mask = df['Borehole_ID'].isin(train_boreholes)
test_mask = df['Borehole_ID'].isin(test_boreholes)

X_train, y_train = X[train_mask], y_encoded[train_mask]
X_test, y_test = X[test_mask], y_encoded[test_mask]

# Load VAE and scaler
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load('ml_models/checkpoints/vae_gra_v2_6_7_final.pth',
                       map_location=device, weights_only=False)

vae_model = VAE(input_dim=6, latent_dim=10, hidden_dims=[32, 16]).to(device)
vae_model.load_state_dict(checkpoint['model_state_dict'])
vae_model.eval()

scaler = checkpoint['scaler']

# Scale and extract embeddings
X_test_scaled = scaler.transform(X_test)
X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)

with torch.no_grad():
    mu, _ = vae_model.encode(X_test_tensor)
    test_embeddings = mu.cpu().numpy()

# Load classifier
classifier = LithologyClassifier(input_dim=10, num_classes=len(unique_labels),
                                hidden_dims=[64, 32]).to(device)
classifier.load_state_dict(torch.load('ml_models/checkpoints/lithology_classifier_v2_6_7_best.pth',
                                     map_location=device))
classifier.eval()

# Get predictions
test_embeddings_tensor = torch.FloatTensor(test_embeddings).to(device)
with torch.no_grad():
    outputs = classifier(test_embeddings_tensor)
    _, test_preds = outputs.max(1)
    test_preds = test_preds.cpu().numpy()

# Analyze 0% accuracy classes
zero_acc_classes = ['silty claystone', 'gabbro', 'mud', 'chalk']

for class_name in zero_acc_classes:
    class_idx = label_to_idx[class_name]
    class_mask = y_test == class_idx
    class_samples = class_mask.sum()
    class_preds = test_preds[class_mask]

    print(f"\n{'='*100}")
    print(f"CLASS: {class_name.upper()}")
    print(f"{'='*100}")
    print(f"Test samples: {class_samples}")
    print(f"Correct predictions: 0 (0.00%)")
    print()

    # What is it being predicted as?
    pred_counter = Counter(class_preds)
    print("Top 10 predictions (what the classifier thinks it is):")
    for pred_idx, count in pred_counter.most_common(10):
        pred_name = idx_to_label[pred_idx]
        pct = 100 * count / class_samples
        print(f"  {pred_name:40s}: {count:5d} samples ({pct:5.1f}%)")

    # Check training data
    train_class_mask = y_train == class_idx
    train_samples = train_class_mask.sum()
    total_train = len(y_train)
    print()
    print(f"Training data for '{class_name}':")
    print(f"  Training samples: {train_samples:,} ({100*train_samples/total_train:.2f}% of training data)")

    # Check physical properties
    class_features = X[df['Borehole_ID'].isin(test_boreholes) & (df['Principal'] == class_name)]
    if len(class_features) > 0:
        print()
        print(f"Physical properties (test set, median values):")
        print(f"  GRA: {np.median(class_features[:, 0]):.3f} g/cm³")
        print(f"  MS:  {np.median(class_features[:, 1]):.1f} instr. units")
        print(f"  NGR: {np.median(class_features[:, 2]):.1f} cps")
        print(f"  RGB: ({np.median(class_features[:, 3]):.0f}, "
              f"{np.median(class_features[:, 4]):.0f}, "
              f"{np.median(class_features[:, 5]):.0f})")

print()
print("="*100)
print("ANALYSIS COMPLETE")
print("="*100)
