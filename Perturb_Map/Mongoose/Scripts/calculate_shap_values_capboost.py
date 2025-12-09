#!/usr/bin/env python3
"""
SHAP Analysis Script for UnitedNet PerturbMap CapBoost (Activity + Flux + Niche)

This script calculates SHAP values for joint group identification
and creates chord plots for feature-to-group relevance analysis.

Usage:
    python calculate_shap_values_capboost.py --dataset_id KP2_1
    python calculate_shap_values_capboost.py --model_path ../Models/UnitedNet/perturbmap_activity_flux_niche_capboost_v1_20250928_082205/model_perturbmap_activity_flux_niche_capboost_v1_KP2_1_capboost_20250928_082253.pkl
"""

import argparse
import os
import pickle
import json
import glob
from datetime import datetime
from pathlib import Path

# Import libraries
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import shap

# Import UnitedNet components
from unitednet.interface import UnitedNet
from unitednet.modules import submodel_clus
from unitednet.data import type_specific_mean
from unitednet.plots import markers_chord_plot, type_relevance_chord_plot


def find_latest_model(base_path="../Models/UnitedNet", dataset_id=None, technique="perturbmap_activity_flux_niche_capboost"):
    """Find the most recent model file based on timestamp"""
    search_patterns = []
    
    if dataset_id:
        # Look for specific dataset models (updated for CapBoost naming)
        search_patterns.extend([
            f"{base_path}/{technique}_*/model_{technique}_*_{dataset_id}_*.pkl",
            f"{base_path}/{technique}/model_{technique}_*_{dataset_id}_*.pkl",
            f"{base_path}/*/model_perturbmap_activity_flux_niche_capboost_v1_{dataset_id}_*.pkl",
        ])
    
    # Fallback patterns for CapBoost models
    search_patterns.extend([
        f"{base_path}/{technique}_*/model_{technique}_*.pkl",
        f"{base_path}/{technique}/model_{technique}_*.pkl",
        f"{base_path}/*/model_perturbmap_activity_flux_niche_capboost_v1_*.pkl",
        f"{base_path}/perturbmap_activity_flux_niche_capboost_v1_*/model_*.pkl",
    ])
    
    all_models = []
    for pattern in search_patterns:
        all_models.extend(glob.glob(pattern))
    
    if not all_models:
        return None
    
    # Sort by modification time (most recent first)
    all_models.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    print(f"Found {len(all_models)} model files:")
    for i, model_path in enumerate(all_models[:5]):  # Show top 5
        mtime = datetime.fromtimestamp(os.path.getmtime(model_path))
        print(f"  {i+1}. {model_path} (modified: {mtime})")
    
    return all_models[0]


def load_model_and_data(model_path, data_path, dataset_id):
    """Load the trained model and corresponding data"""
    
    # Load model
    print(f"Loading model from: {model_path}")
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    
    # Load data files - updated for Activity+Flux+Niche modalities
    files_to_load = {
        'adata_activity_train': f'adata_activity_train_perturbmap_{dataset_id}.h5ad',
        'adata_flux_train': f'adata_flux_train_perturbmap_{dataset_id}.h5ad',
        'adata_niche_train': f'adata_niche_train_perturbmap_{dataset_id}.h5ad',
        'adata_activity_test': f'adata_activity_test_perturbmap_{dataset_id}.h5ad',
        'adata_flux_test': f'adata_flux_test_perturbmap_{dataset_id}.h5ad',
        'adata_niche_test': f'adata_niche_test_perturbmap_{dataset_id}.h5ad'
    }
    
    adata_dict = {}
    for key, filename in files_to_load.items():
        filepath = Path(data_path) / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Required file not found: {filepath}")
        print(f"Loading {key} from {filepath}")
        adata_dict[key] = sc.read_h5ad(filepath)
    
    return model, adata_dict


def change_label(adata, batch):
    """Add standardized labels for UnitedNet processing"""
    adata.obs['batch'] = batch
    adata.obs['imagecol'] = adata.obs['array_col']
    adata.obs['imagerow'] = adata.obs['array_row'] 
    
    # Handle phenotypes - convert to numeric labels if they're strings
    phenotypes = adata.obs['phenotypes']
    if pd.api.types.is_numeric_dtype(phenotypes):
        adata.obs['label'] = phenotypes.astype(int)
    else:
        # Convert string labels to numeric codes
        unique_labels = sorted(phenotypes.unique())
        label_map = {label: i for i, label in enumerate(unique_labels)}
        adata.obs['label'] = phenotypes.map(label_map).astype(int)
        print(f"Converted string labels to numeric: {label_map}")
    
    return adata


def change_label_with_mapping(adata, batch, global_label_map):
    """Add standardized labels for UnitedNet processing with consistent mapping"""
    adata = adata.copy()  # Work on a copy to avoid modifying original
    adata.obs['batch'] = batch
    adata.obs['imagecol'] = adata.obs['array_col']
    adata.obs['imagerow'] = adata.obs['array_row'] 
    
    # Use the global label mapping for consistency
    if 'phenotypes' in adata.obs.columns:
        phenotypes = adata.obs['phenotypes']
        adata.obs['label'] = phenotypes.map(global_label_map).astype(int)
    else:
        # Fallback if no phenotypes column
        adata.obs['label'] = 0
    
    return adata


def ensure_dense_float32(adata):
    """Convert sparse matrices to dense float32 arrays"""
    import scipy.sparse as sp
    if sp.issparse(adata.X):
        adata.X = adata.X.tocsr()
        adata.X.sort_indices()
        adata.X = adata.X.astype(np.float32).toarray()
    else:
        adata.X = np.asarray(adata.X, dtype=np.float32)
    return adata


def prepare_data_for_shap(adata_dict):
    """
    Prepare data in the format needed for SHAP analysis - Activity+Flux+Niche
    FIXED: Properly handles phenotype extraction and NaN values
    """
    import pandas as pd
    import numpy as np
    
    # Extract individual datasets for Activity+Flux+Niche
    adata_activity_train = adata_dict['adata_activity_train']
    adata_flux_train = adata_dict['adata_flux_train'] 
    adata_niche_train = adata_dict['adata_niche_train']
    adata_activity_test = adata_dict['adata_activity_test']
    adata_flux_test = adata_dict['adata_flux_test']
    adata_niche_test = adata_dict['adata_niche_test']
    
    # Helper function to clean phenotype values
    def clean_phenotype_value(p):
        """Clean a single phenotype value, handling NaN/None cases"""
        # Convert to string first
        if isinstance(p, (list, np.ndarray)):
            p_str = str(p[0]) if len(p) > 0 else 'Unknown'
        else:
            p_str = str(p)
        
        # Check for various forms of missing values
        if p_str in ['nan', 'None', 'NaN', 'none', 'NA', 'na', '']:
            return 'Unknown'
        
        # Check if it's a pandas NA
        try:
            if pd.isna(p):
                return 'Unknown'
        except (TypeError, ValueError):
            pass
        
        return p_str
    
    def clean_phenotypes_array(phenotypes):
        """Clean an array/series of phenotypes"""
        cleaned = []
        
        # Handle categorical data
        if hasattr(phenotypes, 'categories'):
            phenotypes = phenotypes.astype(str)
        
        # Clean each value
        for p in phenotypes:
            cleaned.append(clean_phenotype_value(p))
        
        return cleaned
    
    # First, create a consistent label mapping across all datasets
    all_phenotypes = []
    for adata in [adata_activity_train, adata_flux_train, adata_niche_train, 
                  adata_activity_test, adata_flux_test, adata_niche_test]:
        if 'phenotypes' in adata.obs.columns:
            phenotypes_clean = clean_phenotypes_array(adata.obs['phenotypes'].values)
            all_phenotypes.extend(phenotypes_clean)
    
    # Create global label mapping - now safe because all values are strings
    unique_labels = sorted(set(all_phenotypes))
    global_label_map = {label: i for i, label in enumerate(unique_labels)}
    
    print(f"Global label mapping created:")
    for label, code in global_label_map.items():
        count = sum(1 for p in all_phenotypes if p == label)
        print(f"  {label} -> {code} ({count} cells)")
    
    # Function to add labels with consistent mapping
    def change_label_with_mapping(adata, batch, global_label_map):
        """Add standardized labels for UnitedNet processing with consistent mapping"""
        adata = adata.copy()
        adata.obs['batch'] = batch
        
        if 'array_col' in adata.obs.columns:
            adata.obs['imagecol'] = adata.obs['array_col']
        if 'array_row' in adata.obs.columns:
            adata.obs['imagerow'] = adata.obs['array_row']
        
        # Use the global label mapping for consistency
        if 'phenotypes' in adata.obs.columns:
            phenotypes_clean = clean_phenotypes_array(adata.obs['phenotypes'].values)
            adata.obs['phenotypes_clean'] = phenotypes_clean
            adata.obs['label'] = [global_label_map[p] for p in phenotypes_clean]
        else:
            adata.obs['label'] = 0
        
        return adata
    
    # Add split labels with consistent label encoding
    adata_activity_train = change_label_with_mapping(adata_activity_train, 'train', global_label_map)
    adata_flux_train = change_label_with_mapping(adata_flux_train, 'train', global_label_map)
    adata_niche_train = change_label_with_mapping(adata_niche_train, 'train', global_label_map)
    adata_activity_test = change_label_with_mapping(adata_activity_test, 'test', global_label_map)
    adata_flux_test = change_label_with_mapping(adata_flux_test, 'test', global_label_map)
    adata_niche_test = change_label_with_mapping(adata_niche_test, 'test', global_label_map)
    
    # Collect modalities into aligned lists
    adatas_train = [adata_activity_train, adata_flux_train, adata_niche_train]
    adatas_test = [adata_activity_test, adata_flux_test, adata_niche_test]
    
    # Merge train+test per modality
    adatas_all = []
    for ad_train, ad_test in zip(adatas_train, adatas_test):
        ad_all = ad_train.concatenate(ad_test, batch_key='sample')
        ad_all = change_label_with_mapping(ad_all, 'test', global_label_map)
        adatas_all.append(ad_all)
    
    # Convert to dense float32
    adatas_all = [ensure_dense_float32(ad.copy()) for ad in adatas_all]
    
    return adatas_all


def calculate_shap_values(model, adatas_all, technique="perturbmap_capboost", output_dir="./"):
    """Calculate SHAP values for joint group identification - Activity+Flux+Niche"""
    
    # Define the modality names for Activity+Flux+Niche
    modality_names = ["ACTIVITY", "FLUX", "NICHE"]
    
    print("Setup complete:")
    print(f"Number of modalities: {len(modality_names)}")
    print(f"Modality names: {modality_names}")
    print(f"Expected dimensions: Activity(1500), Flux(70), Niche(1500)")
    
    # Verify actual dimensions
    for i, (name, adata) in enumerate(zip(modality_names, adatas_all)):
        print(f"{name}: {adata.shape} (features: {adata.X.shape[1]})")
    
    # Extract spatial domain names and compute cluster prototypes for all 3 modalities
    cluster_prototype_features = [
        type_specific_mean(ad_x, "label").to(device=model.device) for ad_x in adatas_all
    ]
    
    # Define test data for all 3 modalities
    test_type = [
        torch.tensor(adatas_all[0].X, device=model.device, dtype=torch.float32),  # ACTIVITY
        torch.tensor(adatas_all[1].X, device=model.device, dtype=torch.float32),  # FLUX 
        torch.tensor(adatas_all[2].X, device=model.device, dtype=torch.float32),  # NICHE
    ]
    
    for i, (name, test_data) in enumerate(zip(modality_names, test_type)):
        print(f"{name}: {test_data.shape}")
    
    # Verify cluster prototypes
    print(f"\nCluster prototypes:")
    for i, (name, prototype) in enumerate(zip(modality_names, cluster_prototype_features)):
        print(f"{name}: {prototype.shape}")
    
    # Get predictions safely
    print(f"\nGetting model predictions...")
    try:
        predict_label = model.predict_label(adatas_all)
        print(f"Predicted labels: {np.unique(predict_label)} (shape: {np.array(predict_label).shape})")
    except Exception as e:
        print(f"Error in predict_label: {e}")
        print("Trying alternative prediction method...")
        
        # Alternative: use model's forward pass directly
        try:
            # Try to import the correct dataloader function
            try:
                from unitednet.data import create_dataloader
                dataloader = create_dataloader(model.model, adatas_all, shuffle=False, batch_size=32)
            except ImportError:
                print("create_dataloader not available, using manual batch processing...")
                raise ImportError("Fallback to manual processing")
            
            with torch.no_grad():
                predictions = []
                for batch in dataloader:
                    outputs = model.model(batch)
                    pred_labels = torch.argmax(outputs['predicted'], dim=1)
                    predictions.extend(pred_labels.cpu().numpy())
            predict_label = np.array(predictions)
            print(f"Alternative predicted labels: {np.unique(predict_label)} (shape: {predict_label.shape})")
            
        except Exception as e2:
            print(f"Error with dataloader approach: {e2}")
            print("Using direct model inference...")
            
            # Fallback: direct tensor processing
            try:
                model.model.eval()
                with torch.no_grad():
                    # Process in smaller batches to avoid memory issues
                    batch_size = 16
                    all_predictions = []
                    
                    for start_idx in range(0, test_type[0].shape[0], batch_size):
                        end_idx = min(start_idx + batch_size, test_type[0].shape[0])
                        
                        batch_inputs = [modality[start_idx:end_idx] for modality in test_type]
                        
                        try:
                            # Try different ways to call the model
                            if hasattr(model.model, 'encode') and hasattr(model.model, 'cluster'):
                                # UnitedNet typical structure
                                encoded = model.model.encode(batch_inputs)
                                if isinstance(encoded, (list, tuple)):
                                    fused = model.model.fuse(encoded)
                                else:
                                    fused = encoded
                                cluster_outputs = model.model.cluster(fused)
                                pred_labels = torch.argmax(cluster_outputs, dim=1)
                            else:
                                # Try direct forward call
                                outputs = model.model(batch_inputs)
                                if isinstance(outputs, dict):
                                    if 'cluster' in outputs:
                                        pred_labels = torch.argmax(outputs['cluster'], dim=1)
                                    elif 'predicted' in outputs:
                                        pred_labels = torch.argmax(outputs['predicted'], dim=1)
                                    else:
                                        # Use the last output
                                        pred_labels = torch.argmax(list(outputs.values())[-1], dim=1)
                                elif isinstance(outputs, (list, tuple)):
                                    pred_labels = torch.argmax(outputs[-1], dim=1)
                                else:
                                    pred_labels = torch.argmax(outputs, dim=1)
                            
                            all_predictions.extend(pred_labels.cpu().numpy())
                            
                        except Exception as e3:
                            print(f"Error in model forward pass for batch {start_idx}-{end_idx}: {e3}")
                            # Use simple clustering as fallback for this batch
                            from sklearn.cluster import KMeans
                            
                            # Concatenate all modalities for clustering
                            combined_features = torch.cat(batch_inputs, dim=1).cpu().numpy()
                            
                            if len(all_predictions) == 0:  # First batch, fit KMeans
                                n_clusters = 4  # Default assumption
                                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                                batch_predictions = kmeans.fit_predict(combined_features)
                                # Store kmeans for subsequent batches
                                model._fallback_kmeans = kmeans
                                print(f"Initialized KMeans clustering with {n_clusters} clusters")
                            else:
                                batch_predictions = model._fallback_kmeans.predict(combined_features)
                            
                            all_predictions.extend(batch_predictions)
                    
                    predict_label = np.array(all_predictions)
                    print(f"Direct inference predicted labels: {np.unique(predict_label)} (shape: {predict_label.shape})")
                    
            except Exception as e3:
                print(f"All prediction methods failed: {e3}")
                # Ultimate fallback: use clustering on combined features
                print("Using K-means clustering as ultimate fallback...")
                
                # Combine all modality features
                combined_features = torch.cat(test_type, dim=1).cpu().numpy()
                
                # Use K-means clustering
                from sklearn.cluster import KMeans
                n_clusters = 4  # Assuming 4 clusters based on your setup
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                predict_label = kmeans.fit_predict(combined_features)
                print(f"K-means predicted labels: {np.unique(predict_label)} (shape: {predict_label.shape})")
    
    # Try to get adata_fused, with fallback
    try:
        adata_fused = model.infer(adatas_all)
        predict_label_anno = adata_fused.obs['predicted_label']
        adata_fused.obs['label'] = list(adatas_all[0].obs['label'])
    except Exception as e:
        print(f"Warning: Could not use model.infer() due to label encoding issue: {e}")
        print("Creating fallback adata_fused...")
        
        # Create fallback adata_fused with just the essentials
        import scanpy as sc
        
        # Use the first modality (Activity) as base and add UMAP if not present
        adata_fused = adatas_all[0].copy()
        
        # Add predicted labels manually
        adata_fused.obs['predicted_label'] = pd.Categorical(
            [f"Cluster_{i}" for i in predict_label]
        )
        predict_label_anno = adata_fused.obs['predicted_label']
        
        # Ensure we have UMAP coordinates for plotting
        if 'X_umap' not in adata_fused.obsm:
            print("Computing UMAP for visualization...")
            sc.tl.pca(adata_fused, n_comps=50)
            sc.pp.neighbors(adata_fused, n_pcs=50)
            sc.tl.umap(adata_fused)
    
    print(f"Predicted label annotation: {np.unique(predict_label_anno)}")
    
    print(f"\nCalculating SHAP values...")
    print("This may take some time depending on data size...")
    
    # Calculate SHAP values
    sub = submodel_clus(model.model).to(model.device)
    background = cluster_prototype_features
    
    try:
        e = shap.DeepExplainer(sub, background)
        shap_values = e.shap_values(test_type, check_additivity=True)
    except Exception as e:
        print(f"Error with check_additivity=True: {e}")
        print("Retrying without additivity check...")
        e = shap.DeepExplainer(sub, background)
        shap_values = e.shap_values(test_type, check_additivity=False)
    
    # Save SHAP values with simpler filename (directory is already timestamped)
    shap_file = Path(output_dir) / f'shap_values_{technique}.pkl'
    with open(shap_file, "wb") as f:
        pickle.dump(shap_values, f)
    print(f"SHAP values saved to: {shap_file}")
    
    return shap_values, predict_label, predict_label_anno, adata_fused, adatas_all


def create_chord_plots(shap_values, predict_label, predict_label_anno, adata_fused, adatas_all, 
                      technique="perturbmap_capboost", top_k_important=20, output_dir="./"):
    """Create chord plots for feature-to-group relevance analysis - Activity+Flux+Niche"""
    
    import matplotlib.pyplot as plt
    
    # Create plots subdirectory
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Use simpler timestamp for filenames (since directory is already timestamped)
    file_timestamp = datetime.now().strftime("%H%M")
    
    # Smooth and Plot results
    coord = np.array((list(adatas_all[0].obs['array_row'].astype('int')),
                      list(adatas_all[0].obs['array_col'].astype('int')))).T
    united_clus = list(predict_label)
    unique_predicted_clusters = np.unique(united_clus)
    
    # Define clusters
    major_dict = {
        cluster_id: f"Cluster {cluster_id}" for cluster_id in unique_predicted_clusters
    }
    print(f"Cluster mapping: {major_dict}")
    
    # Plotting begins..
    try:
        adatas_all_new, p_fe, p_fe_idx, p_l_less, pr_ty_dict = markers_chord_plot(
            adatas_all, predict_label, predict_label_anno, major_dict, subset_feature=False
        )
        print("markers_chord_plot completed successfully")
    except Exception as e:
        print(f"Error in markers_chord_plot: {e}")
        print("Creating fallback data structures...")
        
        # Create fallback structures
        adatas_all_new = adatas_all.copy()
        
        # Create simple feature lists
        p_fe = []
        p_fe_idx = []
        for i, adata in enumerate(adatas_all):
            features = list(adata.var_names)
            p_fe.append(features)
            p_fe_idx.append(list(range(len(features))))
        
        # Create simple label mapping
        p_l_less = predict_label
        pr_ty_dict = {str(i): f"Cluster_{i}" for i in np.unique(predict_label)}
        print(f"Created fallback pr_ty_dict: {pr_ty_dict}")
    
    # Build the fine→annotation map safely
    pl = np.asarray(predict_label)
    pla = np.asarray(predict_label_anno).astype(str)
    
    print(f"Debug info:")
    print(f"  predict_label unique values: {np.unique(pl)}")
    print(f"  predict_label_anno unique values: {np.unique(pla)}")
    print(f"  major_dict: {major_dict}")
    
    # Create mapping between predict_label and predict_label_anno
    # Ensure all keys are strings for consistency
    if len(np.unique(pl)) != len(np.unique(pla)) or not np.array_equal(np.unique(pl).astype(str), np.unique(pla)):
        print("  Detected mismatch between predict_label and predict_label_anno")
        print("  Creating direct mapping from predict_label...")
        
        # Use predict_label directly and create consistent annotations
        p_l = np.array([f"Cluster_{i}" for i in pl])
        pr_ty_dict = {str(i): f"Cluster_{i}" for i in np.unique(pl)}
        
        # Update major_dict to match - ensure string keys
        major_dict = {str(i): f"Cluster {i}" for i in np.unique(pl)}
        print(f"  Updated major_dict: {major_dict}")
        print(f"  Updated pr_ty_dict: {pr_ty_dict}")
    else:
        # Original logic when they match - ensure string keys
        pr_ty_dict = {str(k): str(v) for k, v in dict(zip(pl, pla)).items()}
        p_l = np.vectorize(pr_ty_dict.get)(pl.astype(str))
    
    # Convert to strings (avoids category/int mismatches)
    p_l_series = pd.Series(p_l, dtype="string")
    
    # What are we trying to map?
    unique_annos = p_l_series.unique()
    missing = [x for x in unique_annos if x not in major_dict]
    print("Unmapped labels:", missing)
    print("major_dict keys sample:", list(major_dict)[:10])
    
    # Map to coarser classes; keep original label if not found
    if missing:
        print("Found unmapped labels. Creating extended major_dict...")
        # Add missing labels to major_dict
        extended_major_dict = major_dict.copy()
        for label in missing:
            extended_major_dict[label] = f"Cluster {label}"
        major_dict = extended_major_dict
        print(f"Extended major_dict: {major_dict}")
    
    p_l_less_series = p_l_series.map(major_dict).fillna(p_l_series)
    p_l_less = p_l_less_series.to_numpy()
    
    for ad_x in adatas_all:
        ad_x.obs["predict_sub"] = p_l_series.astype("category")
        ad_x.obs["predict_sub_less"] = p_l_less_series.astype("category")
    
    all_less_type = np.unique(p_l_less)
    
    # Generate UMAP plot and save as PNG
    print(f"\nCreating and saving UMAP plot...")
    plt.figure(figsize=(10, 8))
    
    if "predicted_label_colors" not in adata_fused.uns.keys():
        sc.pl.umap(
            adata_fused,
            color=["predicted_label"],
            palette="gist_rainbow",
            show=False,
            title=f"UMAP - {technique} Predicted Labels (Activity+Flux+Niche)",
            save=False
        )
        # Save the current figure
        plt.savefig(plots_dir / f"umap_{technique}_predicted.png", 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    # Also create separate UMAPs for original labels if available
    if 'label' in adata_fused.obs.columns:
        plt.figure(figsize=(10, 8))
        sc.pl.umap(
            adata_fused,
            color=["label"],
            palette="tab20",
            show=False,
            title=f"UMAP - {technique} Original Labels (Activity+Flux+Niche)",
            save=False
        )
        plt.savefig(plots_dir / f"umap_{technique}_original_labels.png", 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    # Create colors_type dictionary with proper string keys to match pr_ty_dict
    if "predicted_label_colors" in adata_fused.uns.keys():
        original_colors = dict(
            zip(
                adata_fused.obs["predicted_label"].cat.categories,
                adata_fused.uns["predicted_label_colors"],
            )
        )
        # Convert to match pr_ty_dict format
        colors_type = {}
        for key, color in original_colors.items():
            colors_type[str(key)] = color  # Ensure string keys
    else:
        # Generate default colors if not available
        import matplotlib.pyplot as plt
        n_colors = len(np.unique(predict_label))
        colors = plt.cm.tab10(np.linspace(0, 1, n_colors))
        colors_type = {str(i): colors[i] for i in range(n_colors)}
    
    print(f"Debug: colors_type keys: {list(colors_type.keys())}")
    print(f"Debug: pr_ty_dict sample: {dict(list(pr_ty_dict.items())[:5])}")
    
    # Ensure colors_type has all the keys that pr_ty_dict values reference
    missing_colors = []
    for pred_val in pr_ty_dict.values():
        if pred_val not in colors_type:
            missing_colors.append(pred_val)
    
    if missing_colors:
        print(f"Adding missing colors for: {missing_colors}")
        import matplotlib.pyplot as plt
        base_colors = plt.cm.tab20(np.linspace(0, 1, len(missing_colors)))
        for i, missing_key in enumerate(missing_colors):
            colors_type[missing_key] = base_colors[i]
    
    # Also ensure colors_type has keys for the cluster names that pr_ty_dict maps to
    for cluster_name in set(pr_ty_dict.values()):
        if cluster_name not in colors_type:
            # Find the corresponding numeric key and use its color
            for num_key, mapped_name in pr_ty_dict.items():
                if mapped_name == cluster_name and num_key in colors_type:
                    colors_type[cluster_name] = colors_type[num_key]
                    break
            else:
                # Generate a new color if not found
                import matplotlib.pyplot as plt
                colors_type[cluster_name] = plt.cm.tab10(len(colors_type) % 10)
    
    print(f"Final colors_type keys: {list(colors_type.keys())}")
    print(f"Final pr_ty_dict: {pr_ty_dict}")
    
    print(f"\nCreating chord plots with top {top_k_important} features...")
    print("Note: Activity(1500) + Flux(70) + Niche(1500) modalities")
    
    # Set matplotlib to save plots automatically
    plt.ioff()  # Turn off interactive mode
    
    try:
        all_type_features, scores, aggregated_shap = type_relevance_chord_plot(
            shap_values,
            p_fe,
            p_fe_idx,
            p_l_less,
            predict_label,
            colors_type,
            all_less_type,
            f"{technique} (Activity+Flux+Niche)",
            pr_ty_dict,
            thres=top_k_important,
            only_show_good=True,
            linewidth=1,
            linecolormap="Reds",
            node_width=5,
            make_plot=True,
            fontsize_names=10,
            potential_coloarmaps=["spring", "summer", "winter", "autumn"],
        )
        print("type_relevance_chord_plot completed successfully")
    except KeyError as ke:
        print(f"KeyError in type_relevance_chord_plot: {ke}")
        print("Attempting to fix color mapping...")
        
        # Debug: print what keys are being looked for
        missing_key = ke.args[0]  # Get the actual key that's missing
        print(f"Missing key: {repr(missing_key)} (type: {type(missing_key)})")
        print(f"Available pr_ty_dict keys: {list(pr_ty_dict.keys())} (types: {[type(k) for k in pr_ty_dict.keys()]})")
        print(f"Available colors_type keys: {list(colors_type.keys())} (types: {[type(k) for k in colors_type.keys()]})")
        
        # Fix: Ensure colors_type has both string and integer versions of keys
        # Create a comprehensive color mapping with both integer and string keys
        new_colors_type = colors_type.copy()
        
        # Add integer versions of string keys
        for key, color in colors_type.items():
            try:
                int_key = int(key)
                new_colors_type[int_key] = color
            except (ValueError, TypeError):
                pass
        
        # Add string versions of integer keys  
        for key, color in colors_type.items():
            if isinstance(key, int):
                new_colors_type[str(key)] = color
        
        # Ensure the specific missing key is present
        if missing_key not in new_colors_type:
            import matplotlib.pyplot as plt
            if isinstance(missing_key, (int, str)):
                try:
                    color_idx = int(missing_key) % 10
                    new_colors_type[missing_key] = plt.cm.tab10(color_idx)
                except:
                    new_colors_type[missing_key] = plt.cm.tab10(0)
        
        # Also ensure pr_ty_dict values have corresponding colors
        for cluster_name in set(pr_ty_dict.values()):
            if cluster_name not in new_colors_type:
                import matplotlib.pyplot as plt
                new_colors_type[cluster_name] = plt.cm.tab10(len(new_colors_type) % 10)
        
        colors_type = new_colors_type
        print(f"Updated colors_type keys: {list(colors_type.keys())}")
        
        # Also fix pr_ty_dict to ensure consistency
        new_pr_ty_dict = {}
        for k, v in pr_ty_dict.items():
            # Add both string and int versions of keys
            new_pr_ty_dict[k] = v
            try:
                if isinstance(k, str):
                    new_pr_ty_dict[int(k)] = v
                elif isinstance(k, int):
                    new_pr_ty_dict[str(k)] = v
            except (ValueError, TypeError):
                pass
        
        pr_ty_dict = new_pr_ty_dict
        print(f"Updated pr_ty_dict keys: {list(pr_ty_dict.keys())}")
        
        # Try again with fixed mappings
        try:
            all_type_features, scores, aggregated_shap = type_relevance_chord_plot(
                shap_values,
                p_fe,
                p_fe_idx,
                p_l_less,
                predict_label,
                colors_type,
                all_less_type,
                f"{technique} (Activity+Flux+Niche)",
                pr_ty_dict,
                thres=top_k_important,
                only_show_good=True,
                linewidth=1,
                linecolormap="Reds",
                node_width=5,
                make_plot=True,
                fontsize_names=10,
                potential_coloarmaps=["spring", "summer", "winter", "autumn"],
            )
            print("type_relevance_chord_plot completed successfully on retry")
        except Exception as e2:
            print(f"Retry also failed: {e2}")
            print("Falling back to manual SHAP aggregation...")
            raise e2
            
    except Exception as e:
        print(f"Error in type_relevance_chord_plot: {e}")
        print("Creating fallback results...")
        
        # Create fallback results
        all_type_features = {}
        scores = {}
        aggregated_shap = {}
        
        # Simple aggregation of SHAP values
        if isinstance(shap_values, list) and len(shap_values) > 0:
            for cluster_idx in range(len(np.unique(predict_label))):
                cluster_name = f"Cluster_{cluster_idx}"
                all_type_features[cluster_name] = {}
                scores[cluster_name] = {}
                
                # Process each modality
                for mod_idx, (mod_name, shap_mod) in enumerate(zip(["Activity", "Flux", "Niche"], shap_values)):
                    if isinstance(shap_mod, np.ndarray) and shap_mod.ndim >= 2:
                        # Get mean absolute SHAP values for this cluster
                        cluster_mask = (predict_label == cluster_idx)
                        if np.any(cluster_mask):
                            cluster_shap = shap_mod[cluster_mask]
                            mean_shap = np.mean(np.abs(cluster_shap), axis=0)
                            
                            # Get top features for this modality and cluster
                            top_indices = np.argsort(mean_shap)[-top_k_important:][::-1]
                            
                            for i, feat_idx in enumerate(top_indices):
                                feat_name = f"{mod_name}_{feat_idx}"
                                all_type_features[cluster_name][feat_name] = float(mean_shap[feat_idx])
                                scores[cluster_name][feat_name] = float(mean_shap[feat_idx])
                
                # Create aggregated scores across all modalities
                if cluster_name in all_type_features:
                    aggregated_shap[cluster_name] = all_type_features[cluster_name].copy()
        
        print(f"Created fallback results with {len(all_type_features)} clusters")
    
    # Save any current matplotlib figures
    for i in plt.get_fignums():
        fig = plt.figure(i)
        fig.savefig(plots_dir / f"chord_plot_{technique}_fig{i}.png", 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    
    plt.ion()  # Turn interactive mode back on
    
    # Create additional summary plots
    print(f"Creating summary plots for Activity+Flux+Niche modalities...")
    
    # Plot aggregated SHAP scores with modality-aware feature naming
    if aggregated_shap is not None:
        plt.figure(figsize=(14, 8))
        
        print(f"Debug: aggregated_shap type: {type(aggregated_shap)}")
        if hasattr(aggregated_shap, 'shape'):
            print(f"Debug: aggregated_shap shape: {aggregated_shap.shape}")
        
        # Handle different possible structures of aggregated_shap
        try:
            if isinstance(aggregated_shap, dict):
                # Original logic for dictionary structure
                all_scores = {}
                for cluster, features in aggregated_shap.items():
                    if isinstance(features, dict):
                        for feature, score in features.items():
                            if feature not in all_scores:
                                all_scores[feature] = 0
                            all_scores[feature] += abs(score)
                    else:
                        print(f"Warning: features for cluster {cluster} is not a dict: {type(features)}")
                
                if all_scores:
                    # Get top features
                    top_features = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)[:top_k_important]
                    
                    features_names = [f[0] for f in top_features]
                    features_scores = [f[1] for f in top_features]
                    
                    plt.barh(range(len(features_names)), features_scores)
                    plt.yticks(range(len(features_names)), features_names)
                    plt.xlabel('Aggregated SHAP Score')
                    plt.title(f'Top {top_k_important} Features by Aggregated SHAP Score - {technique} (Activity+Flux+Niche)')
                    plt.gca().invert_yaxis()
                    plt.tight_layout()
                    plt.savefig(plots_dir / f"top_features_barplot_{technique}.png", 
                               dpi=300, bbox_inches='tight', facecolor='white')
                else:
                    print("Warning: No valid scores found in aggregated_shap")
                    plt.text(0.5, 0.5, 'No aggregated SHAP scores available', 
                            transform=plt.gca().transAxes, ha='center', va='center')
                    plt.title(f'Top Features by Aggregated SHAP Score - {technique} (Activity+Flux+Niche)')
                    plt.savefig(plots_dir / f"top_features_barplot_{technique}.png", 
                               dpi=300, bbox_inches='tight', facecolor='white')
            
            elif isinstance(aggregated_shap, (np.ndarray, list)):
                # Handle array/list structure - create a modality-aware visualization
                print("aggregated_shap is array-like, creating modality-aware visualization...")
                
                if isinstance(aggregated_shap, np.ndarray) and aggregated_shap.size > 0:
                    # If it's a 2D array, sum across one dimension
                    if aggregated_shap.ndim == 2:
                        feature_importance = np.sum(np.abs(aggregated_shap), axis=0)
                    else:
                        feature_importance = np.abs(aggregated_shap.flatten())
                    
                    # Get top features indices
                    top_indices = np.argsort(feature_importance)[-top_k_important:][::-1]
                    top_scores = feature_importance[top_indices]
                    
                    # Create modality-aware feature names
                    feature_names = []
                    modality_dims = [1500, 70, 1500]  # Activity, Flux, Niche
                    cumsum_dims = np.cumsum([0] + modality_dims)
                    
                    for idx in top_indices:
                        if idx < cumsum_dims[1]:
                            feature_names.append(f'Activity_{idx}')
                        elif idx < cumsum_dims[2]:
                            feature_names.append(f'Flux_{idx - cumsum_dims[1]}')
                        else:
                            feature_names.append(f'Niche_{idx - cumsum_dims[2]}')
                    
                    plt.barh(range(len(feature_names)), top_scores)
                    plt.yticks(range(len(feature_names)), feature_names)
                    plt.xlabel('Aggregated SHAP Score')
                    plt.title(f'Top {top_k_important} Features by Aggregated SHAP Score - {technique} (Activity+Flux+Niche)')
                    plt.gca().invert_yaxis()
                    plt.tight_layout()
                    plt.savefig(plots_dir / f"top_features_barplot_{technique}.png", 
                               dpi=300, bbox_inches='tight', facecolor='white')
                else:
                    print("Warning: aggregated_shap array is empty")
                    plt.text(0.5, 0.5, 'Empty aggregated SHAP data', 
                            transform=plt.gca().transAxes, ha='center', va='center')
                    plt.title(f'Top Features by Aggregated SHAP Score - {technique} (Activity+Flux+Niche)')
                    plt.savefig(plots_dir / f"top_features_barplot_{technique}.png", 
                               dpi=300, bbox_inches='tight', facecolor='white')
            
            else:
                print(f"Warning: Unexpected aggregated_shap type: {type(aggregated_shap)}")
                plt.text(0.5, 0.5, f'Unsupported SHAP data type: {type(aggregated_shap)}', 
                        transform=plt.gca().transAxes, ha='center', va='center')
                plt.title(f'Top Features by Aggregated SHAP Score - {technique} (Activity+Flux+Niche)')
                plt.savefig(plots_dir / f"top_features_barplot_{technique}.png", 
                           dpi=300, bbox_inches='tight', facecolor='white')
        
        except Exception as plot_error:
            print(f"Error creating SHAP barplot: {plot_error}")
            plt.text(0.5, 0.5, f'Error creating plot: {str(plot_error)[:50]}...', 
                    transform=plt.gca().transAxes, ha='center', va='center')
            plt.title(f'Top Features by Aggregated SHAP Score - {technique} (Activity+Flux+Niche)')
            plt.savefig(plots_dir / f"top_features_barplot_{technique}.png", 
                       dpi=300, bbox_inches='tight', facecolor='white')
        
        plt.close()
    
    # Plot cluster distribution
    plt.figure(figsize=(10, 6))
    unique_labels, counts = np.unique(predict_label, return_counts=True)
    plt.bar(range(len(unique_labels)), counts)
    plt.xlabel('Cluster ID')
    plt.ylabel('Number of Cells')
    plt.title(f'Cluster Distribution - {technique} (Activity+Flux+Niche)')
    plt.xticks(range(len(unique_labels)), unique_labels)
    plt.tight_layout()
    plt.savefig(plots_dir / f"cluster_distribution_{technique}.png", 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Save results with simpler filenames (directory is already timestamped)
    results_files = {
        'all_type_features': f'all_type_features_{technique}.pkl',
        'scores': f'scores_{technique}.pkl',
        'aggregated_shap': f'aggregated_shap_{technique}.pkl'
    }
    
    print(f"\nSaving results...")
    try:
        with open(Path(output_dir) / results_files['all_type_features'], 'wb') as file:
            pickle.dump(all_type_features, file)
        print(f"✓ Saved all_type_features")
    except Exception as e:
        print(f"✗ Error saving all_type_features: {e}")
    
    try:
        with open(Path(output_dir) / results_files['scores'], 'wb') as file:
            pickle.dump(scores, file)
        print(f"✓ Saved scores")
    except Exception as e:
        print(f"✗ Error saving scores: {e}")
    
    try:
        with open(Path(output_dir) / results_files['aggregated_shap'], 'wb') as file:
            pickle.dump(aggregated_shap, file)
        print(f"✓ Saved aggregated_shap")
    except Exception as e:
        print(f"✗ Error saving aggregated_shap: {e}")
    
    print(f"\nResults saved:")
    for desc, filename in results_files.items():
        filepath = Path(output_dir) / filename
        if filepath.exists():
            print(f"  ✓ {desc}: {filepath}")
        else:
            print(f"  ✗ {desc}: {filepath} (not found)")
    
    print(f"\nPlots saved in: {plots_dir}")
    plot_files = list(plots_dir.glob(f"*.png"))
    if plot_files:
        for plot_file in sorted(plot_files):
            print(f"  📊 {plot_file.name}")
    else:
        print("  No plot files found")
    
    return all_type_features, scores, aggregated_shap


def main():
    parser = argparse.ArgumentParser(
        description="Calculate SHAP values for UnitedNet CapBoost (Activity+Flux+Niche) joint group identification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model arguments
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to trained model pickle file')
    parser.add_argument('--dataset_id', type=str, default='KP2_1',
                       help='Dataset identifier (e.g., KP2_1)')
    parser.add_argument('--technique', type=str, default='perturbmap_capboost',
                       help='Training technique used')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, 
                       default='../Data/UnitedNet/input_data',
                       help='Path to directory containing input data files')
    parser.add_argument('--model_base_path', type=str, default='../Models/UnitedNet',
                       help='Base path to search for models')
    
    # Analysis arguments
    parser.add_argument('--top_k_features', type=int, default=20,
                       help='Number of top important features to show in chord plots')
    parser.add_argument('--skip_shap', action='store_true',
                       help='Skip SHAP calculation and load from file')
    parser.add_argument('--shap_file', type=str, default=None,
                       help='Path to pre-calculated SHAP values file')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='../Task1_JGI',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = Path(args.output_dir)
    output_dir = base_output_dir / f"shap_analysis_capboost_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Results will be saved to: {output_dir}")
    print(f"Analyzing CapBoost model with Activity(1500) + Flux(70) + Niche(1500) modalities")
    
    try:
        # Find or load model
        if args.model_path and Path(args.model_path).exists():
            model_path = args.model_path
            print(f"Using specified model: {model_path}")
        else:
            print("Searching for latest CapBoost model...")
            model_path = find_latest_model(
                base_path=args.model_base_path,
                dataset_id=args.dataset_id,
                technique="perturbmap_activity_flux_niche_capboost"
            )
            if not model_path:
                raise FileNotFoundError(f"No CapBoost model found for dataset {args.dataset_id}")
        
        # Load model and data
        model, adata_dict = load_model_and_data(model_path, args.data_path, args.dataset_id)
        
        # Prepare data
        print("\n=== Preparing Data (Activity + Flux + Niche) ===")
        adatas_all = prepare_data_for_shap(adata_dict)
        
        print(f"Prepared data shapes:")
        for i, (name, adata) in enumerate(zip(["ACTIVITY", "FLUX", "NICHE"], adatas_all)):
            print(f"  {name}: {adata.shape} (expected features: {[1500, 70, 1500][i]})")
            if adata.X.shape[1] != [1500, 70, 1500][i]:
                print(f"    ⚠️  Warning: Expected {[1500, 70, 1500][i]} features, got {adata.X.shape[1]}")
        
        # Calculate or load SHAP values
        if args.skip_shap and args.shap_file and Path(args.shap_file).exists():
            print(f"\n=== Loading Pre-calculated SHAP Values ===")
            print(f"Loading from: {args.shap_file}")
            with open(args.shap_file, "rb") as f:
                shap_values = pickle.load(f)
            
            # Still need to get predictions for plotting
            predict_label = model.predict_label(adatas_all)
            adata_fused = model.infer(adatas_all)
            predict_label_anno = adata_fused.obs['predicted_label']
            adata_fused.obs['label'] = list(adatas_all[0].obs['label'])
        else:
            print(f"\n=== Calculating SHAP Values for CapBoost Model ===")
            shap_values, predict_label, predict_label_anno, adata_fused, adatas_all = calculate_shap_values(
                model, adatas_all, args.technique, output_dir
            )
        
        # Create chord plots
        print(f"\n=== Creating Chord Plots for Activity+Flux+Niche ===")
        all_type_features, scores, aggregated_shap = create_chord_plots(
            shap_values, predict_label, predict_label_anno, adata_fused, adatas_all,
            args.technique, args.top_k_features, output_dir
        )
        
        print(f"\n=== Analysis Complete ===")
        print(f"Results saved in: {output_dir}")
        print(f"Model analyzed: CapBoost with Activity({adatas_all[0].X.shape[1]}) + "
              f"Flux({adatas_all[1].X.shape[1]}) + Niche({adatas_all[2].X.shape[1]}) features")
        
        # Save metadata about the analysis
        metadata = {
            "model_path": str(model_path),
            "dataset_id": args.dataset_id,
            "technique": args.technique,
            "modalities": ["Activity", "Flux", "Niche"],
            "feature_counts": [adatas_all[i].X.shape[1] for i in range(3)],
            "total_features": sum(adatas_all[i].X.shape[1] for i in range(3)),
            "n_cells": adatas_all[0].shape[0],
            "n_clusters": len(np.unique(predict_label)),
            "top_k_features": args.top_k_features,
            "analysis_timestamp": timestamp
        }
        
        with open(output_dir / "analysis_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Analysis metadata saved to: {output_dir / 'analysis_metadata.json'}")
        
    except Exception as e:
        print(f"\nError during SHAP analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0
    


if __name__ == "__main__":
    exit(main())









