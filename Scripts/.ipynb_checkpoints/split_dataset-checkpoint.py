#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3
"""
Simple data splitting script for PerturbMap KP1-2 data
"""

import numpy as np
import scanpy as sc
from sklearn.model_selection import train_test_split
import os

# Create output directory if it doesn't exist
output_dir = '../Data/UnitedNet/input_data'
os.makedirs(output_dir, exist_ok=True)

print("Loading data...")

# Load the data files
rna = sc.read_h5ad('../Data/processed_data/adata_KP_1-2_hvg.h5ad')
niche_hvg = sc.read_h5ad('../Data/processed_data/niche_kp12_dc_hvg_matched.h5ad')  # UPDATED PATH
adata_hvg2k = sc.read_h5ad('../Data/processed_data/activity_kp12_dc_hvg.h5ad')

print(f"Data loaded - RNA: {rna.shape}, Niche: {niche_hvg.shape}, Activity: {adata_hvg2k.shape}")

# Assign modalities
adata_rna = rna
adata_niche = niche_hvg
adata_activity = adata_hvg2k

# Define the split ratio
train_size = 0.8

# Get total number of observations (cells)
n_samples = adata_rna.n_obs
print(f"Total samples: {n_samples}")

# Generate indices
train_idx, test_idx = train_test_split(
    np.arange(n_samples),
    train_size=train_size,
    random_state=42
)

print(f"Train samples: {len(train_idx)}, Test samples: {len(test_idx)}")

# --- RNA modality ---
print("Splitting RNA modality...")
adata_rna_train = adata_rna[train_idx].copy()
adata_rna_test = adata_rna[test_idx].copy()
adata_rna_train.write_h5ad(f'{output_dir}/adata_rna_train_perturbmap_KP2_1.h5ad')
adata_rna_test.write_h5ad(f'{output_dir}/adata_rna_test_perturbmap_KP2_1.h5ad')

# --- Niche modality ---
print("Splitting Niche modality...")
adata_niche_train = adata_niche[train_idx].copy()
adata_niche_test = adata_niche[test_idx].copy()
adata_niche_train.write_h5ad(f'{output_dir}/adata_niche_train_perturbmap_KP2_1.h5ad')
adata_niche_test.write_h5ad(f'{output_dir}/adata_niche_test_perturbmap_KP2_1.h5ad')

# --- Activity modality ---
print("Splitting Activity modality...")
adata_activity_train = adata_activity[train_idx].copy()
adata_activity_test = adata_activity[test_idx].copy()
adata_activity_train.write_h5ad(f'{output_dir}/adata_activity_train_perturbmap_KP2_1.h5ad')
adata_activity_test.write_h5ad(f'{output_dir}/adata_activity_test_perturbmap_KP2_1.h5ad')

print("Data splitting completed successfully!")
print(f"Files saved to: {output_dir}")
print("Output files:")
print(f"  - adata_rna_train_perturbmap_KP2_1.h5ad")
print(f"  - adata_rna_test_perturbmap_KP2_1.h5ad")
print(f"  - adata_niche_train_perturbmap_KP2_1.h5ad")
print(f"  - adata_niche_test_perturbmap_KP2_1.h5ad")
print(f"  - adata_activity_train_perturbmap_KP2_1.h5ad")
print(f"  - adata_activity_test_perturbmap_KP2_1.h5ad")

