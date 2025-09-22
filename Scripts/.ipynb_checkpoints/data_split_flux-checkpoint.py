#!/usr/bin/env python
# coding: utf-8
# In[ ]:
#!/usr/bin/env python3
"""
Simple data splitting script for PerturbMap KP1-2 data with Flux modality
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
niche_hvg = sc.read_h5ad('../Data/processed_data/niche_kp12_dc_hvg_matched.h5ad')
flux_hvg = sc.read_h5ad('../Data/processed_data/flux_kp12.h5ad')  # HARMONIZED PATH

print(f"Data loaded - RNA: {rna.shape}, Niche: {niche_hvg.shape}, Flux: {flux_hvg.shape}")

# Assign modalities
adata_rna = rna
adata_niche = niche_hvg
adata_flux = flux_hvg  # CHANGED FROM ACTIVITY TO FLUX

# Verify all datasets have the same number of cells
assert adata_rna.n_obs == adata_niche.n_obs == adata_flux.n_obs, \
    f"Cell counts don't match: RNA={adata_rna.n_obs}, Niche={adata_niche.n_obs}, Flux={adata_flux.n_obs}"

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

# --- Flux modality --- (CHANGED FROM ACTIVITY)
print("Splitting Flux modality...")
adata_flux_train = adata_flux[train_idx].copy()
adata_flux_test = adata_flux[test_idx].copy()

adata_flux_train.write_h5ad(f'{output_dir}/adata_flux_train_perturbmap_KP2_1.h5ad')
adata_flux_test.write_h5ad(f'{output_dir}/adata_flux_test_perturbmap_KP2_1.h5ad')

print("Data splitting completed successfully!")
print(f"Files saved to: {output_dir}")

print("Output files:")
print(f"  - adata_rna_train_perturbmap_KP2_1.h5ad")
print(f"  - adata_rna_test_perturbmap_KP2_1.h5ad")
print(f"  - adata_niche_train_perturbmap_KP2_1.h5ad")
print(f"  - adata_niche_test_perturbmap_KP2_1.h5ad")
print(f"  - adata_flux_train_perturbmap_KP2_1.h5ad")  # CHANGED FROM ACTIVITY
print(f"  - adata_flux_test_perturbmap_KP2_1.h5ad")   # CHANGED FROM ACTIVITY

# Optional: Print flux data information
print(f"\nFlux modality details:")
print(f"  - Metabolites: {adata_flux.n_vars}")
if 'metabolite_category' in adata_flux.var.columns:
    print(f"  - Metabolite categories: {adata_flux.var['metabolite_category'].value_counts().to_dict()}")
if 'has_flux_data' in adata_flux.obs.columns:
    print(f"  - Cells with actual flux data: {adata_flux.obs['has_flux_data'].sum()}")
    print(f"  - Cells with filled zeros: {(~adata_flux.obs['has_flux_data']).sum()}")

# Verify the splits maintain the same cell order
print(f"\nVerification:")
print(f"Train split - RNA cells: {adata_rna_train.n_obs}, Niche cells: {adata_niche_train.n_obs}, Flux cells: {adata_flux_train.n_obs}")
print(f"Test split - RNA cells: {adata_rna_test.n_obs}, Niche cells: {adata_niche_test.n_obs}, Flux cells: {adata_flux_test.n_obs}")

# Check if cell barcodes match (first few)
print(f"\nSample train cell barcodes match:")
print(f"RNA train[0]: {adata_rna_train.obs_names[0]}")
print(f"Niche train[0]: {adata_niche_train.obs_names[0]}")
print(f"Flux train[0]: {adata_flux_train.obs_names[0]}")
print(f"Match: {adata_rna_train.obs_names[0] == adata_niche_train.obs_names[0] == adata_flux_train.obs_names[0]}")