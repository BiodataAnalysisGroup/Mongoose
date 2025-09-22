#!/usr/bin/env python3
"""
SHAP Analysis Script for UnitedNet PerturbMap

This script calculates SHAP values for joint group identification
and creates chord plots for feature-to-group relevance analysis.

Usage:
    python calculate_shap_values.py --dataset_id KP2_1
    python calculate_shap_values.py --model_path ../Model/perturbmap/model_perturbmap_KP2_1.pkl
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


def find_latest_model(base_path="../Model", dataset_id=None, technique="perturbmap"):
    """Find the most recent model file based on timestamp"""
    search_patterns = []
    
    if dataset_id:
        # Look for specific dataset models
        search_patterns.extend([
            f"{base_path}/{technique}_*/model_{technique}_{dataset_id}_*.pkl",
            f"{base_path}/{technique}/model_{technique}_{dataset_id}_*.pkl",
            f"{base_path}/*/model_{technique}_{dataset_id}_*.pkl",
        ])
    
    # Fallback patterns
    search_patterns.extend([
        f"{base_path}/{technique}_*/model_{technique}_*.pkl",
        f"{base_path}/{technique}/model_{technique}_*.pkl", 
        f"{base_path}/*/model_{technique}_*.pkl",
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
    
    # Load data files
    files_to_load = {
        'adata_rna_train': f'adata_rna_train_perturbmap_{dataset_id}.h5ad',
        'adata_niche_train': f'adata_niche_train_perturbmap_{dataset_id}.h5ad', 
        'adata_activity_train': f'adata_activity_train_perturbmap_{dataset_id}.h5ad',
        'adata_rna_test': f'adata_rna_test_perturbmap_{dataset_id}.h5ad',
        'adata_niche_test': f'adata_niche_test_perturbmap_{dataset_id}.h5ad',
        'adata_activity_test': f'adata_activity_test_perturbmap_{dataset_id}.h5ad'
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
    adata.obs['label'] = adata.obs['phenotypes']
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
    """Prepare data in the format needed for SHAP analysis"""
    
    # Extract individual datasets
    adata_rna_train = adata_dict['adata_rna_train']
    adata_niche_train = adata_dict['adata_niche_train'] 
    adata_activity_train = adata_dict['adata_activity_train']
    adata_rna_test = adata_dict['adata_rna_test']
    adata_niche_test = adata_dict['adata_niche_test']
    adata_activity_test = adata_dict['adata_activity_test']
    
    # Add split labels
    adata_rna_train = change_label(adata_rna_train, 'train')
    adata_niche_train = change_label(adata_niche_train, 'train')
    adata_activity_train = change_label(adata_activity_train, 'train')
    adata_rna_test = change_label(adata_rna_test, 'test')
    adata_niche_test = change_label(adata_niche_test, 'test')
    adata_activity_test = change_label(adata_activity_test, 'test')
    
    # Collect modalities into aligned lists
    adatas_train = [adata_rna_train, adata_niche_train, adata_activity_train]
    adatas_test = [adata_rna_test, adata_niche_test, adata_activity_test]
    
    # Merge train+test per modality
    adatas_all = []
    for ad_train, ad_test in zip(adatas_train, adatas_test):
        ad_all = ad_train.concatenate(ad_test, batch_key='sample')
        ad_all = change_label(ad_all, 'test')
        adatas_all.append(ad_all)
    
    # Convert to dense float32
    adatas_all = [ensure_dense_float32(ad.copy()) for ad in adatas_all]
    
    return adatas_all


def calculate_shap_values(model, adatas_all, technique="perturbmap", output_dir="./"):
    """Calculate SHAP values for joint group identification"""
    
    # Define the modality names for all 3 modalities
    modality_names = ["RNA", "NICHE", "ACTIVITY"]
    
    print("Setup complete:")
    print(f"Number of modalities: {len(modality_names)}")
    print(f"Modality names: {modality_names}")
    
    # Extract spatial domain names and compute cluster prototypes for all 3 modalities
    cluster_prototype_features = [
        type_specific_mean(ad_x, "label").to(device=model.device) for ad_x in adatas_all
    ]
    
    # Define test data for all 3 modalities
    test_type = [
        torch.tensor(adatas_all[0].X, device=model.device, dtype=torch.float32),  # RNA
        torch.tensor(adatas_all[1].X, device=model.device, dtype=torch.float32),  # NICHE 
        torch.tensor(adatas_all[2].X, device=model.device, dtype=torch.float32),  # ACTIVITY
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
        from unitednet.scripts import run_infer
        from unitednet.data import generate_dataloader
        
        dataloader = generate_dataloader(adatas_all, batch_size=32, train=False)
        with torch.no_grad():
            predictions = []
            for batch in dataloader:
                outputs = model.model(batch)
                pred_labels = torch.argmax(outputs['predicted'], dim=1)
                predictions.extend(pred_labels.cpu().numpy())
        predict_label = np.array(predictions)
        print(f"Alternative predicted labels: {np.unique(predict_label)} (shape: {predict_label.shape})")
    
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
        
        # Use the first modality as base and add UMAP if not present
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
                      technique="perturbmap", top_k_important=20, output_dir="./"):
    """Create chord plots for feature-to-group relevance analysis"""
    
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
    adatas_all_new, p_fe, p_fe_idx, p_l_less, pr_ty_dict = markers_chord_plot(
        adatas_all, predict_label, predict_label_anno, major_dict, subset_feature=False
    )
    
    # Build the fine→annotation map safely
    pl = np.asarray(predict_label)
    pla = np.asarray(predict_label_anno).astype(str)
    
    print(f"Debug info:")
    print(f"  predict_label unique values: {np.unique(pl)}")
    print(f"  predict_label_anno unique values: {np.unique(pla)}")
    print(f"  major_dict: {major_dict}")
    
    # Create mapping between predict_label and predict_label_anno
    # If they're different, we need to align them
    if len(np.unique(pl)) != len(np.unique(pla)) or not np.array_equal(np.unique(pl).astype(str), np.unique(pla)):
        print("  Detected mismatch between predict_label and predict_label_anno")
        print("  Creating direct mapping from predict_label...")
        
        # Use predict_label directly and create consistent annotations
        p_l = np.array([f"Cluster_{i}" for i in pl])
        pr_ty_dict = {i: f"Cluster_{i}" for i in np.unique(pl)}
        
        # Update major_dict to match
        major_dict = {i: f"Cluster {i}" for i in np.unique(pl)}
        print(f"  Updated major_dict: {major_dict}")
    else:
        # Original logic when they match
        pr_ty_dict = dict(zip(pl, pla))
        p_l = np.vectorize(pr_ty_dict.get)(pl)
    
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
            title=f"UMAP - {technique} Predicted Labels",
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
            title=f"UMAP - {technique} Original Labels",
            save=False
        )
        plt.savefig(plots_dir / f"umap_{technique}_original_labels.png", 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    colors_type = dict(
        zip(
            adata_fused.obs["predicted_label"].cat.categories,
            adata_fused.uns["predicted_label_colors"],
        )
    )
    
    print(f"\nCreating chord plots with top {top_k_important} features...")
    
    # Set matplotlib to save plots automatically
    plt.ioff()  # Turn off interactive mode
    
    all_type_features, scores, aggregated_shap = type_relevance_chord_plot(
        shap_values,
        p_fe,
        p_fe_idx,
        p_l_less,
        predict_label,
        colors_type,
        all_less_type,
        f"{technique}",
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
    
    # Save any current matplotlib figures
    for i in plt.get_fignums():
        fig = plt.figure(i)
        fig.savefig(plots_dir / f"chord_plot_{technique}_fig{i}.png", 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    
    plt.ion()  # Turn interactive mode back on
    
    # Create additional summary plots
    print(f"Creating summary plots...")
    
    # Plot aggregated SHAP scores
    if aggregated_shap is not None:
        plt.figure(figsize=(12, 8))
        
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
                    plt.title(f'Top {top_k_important} Features by Aggregated SHAP Score - {technique}')
                    plt.gca().invert_yaxis()
                    plt.tight_layout()
                    plt.savefig(plots_dir / f"top_features_barplot_{technique}.png", 
                               dpi=300, bbox_inches='tight', facecolor='white')
                else:
                    print("Warning: No valid scores found in aggregated_shap")
                    plt.text(0.5, 0.5, 'No aggregated SHAP scores available', 
                            transform=plt.gca().transAxes, ha='center', va='center')
                    plt.title(f'Top Features by Aggregated SHAP Score - {technique}')
                    plt.savefig(plots_dir / f"top_features_barplot_{technique}.png", 
                               dpi=300, bbox_inches='tight', facecolor='white')
            
            elif isinstance(aggregated_shap, (np.ndarray, list)):
                # Handle array/list structure - create a simple summary plot
                print("aggregated_shap is array-like, creating alternative visualization...")
                
                if isinstance(aggregated_shap, np.ndarray) and aggregated_shap.size > 0:
                    # If it's a 2D array, sum across one dimension
                    if aggregated_shap.ndim == 2:
                        feature_importance = np.sum(np.abs(aggregated_shap), axis=0)
                    else:
                        feature_importance = np.abs(aggregated_shap.flatten())
                    
                    # Get top features indices
                    top_indices = np.argsort(feature_importance)[-top_k_important:][::-1]
                    top_scores = feature_importance[top_indices]
                    
                    # Create feature names
                    feature_names = [f'Feature_{i}' for i in top_indices]
                    
                    plt.barh(range(len(feature_names)), top_scores)
                    plt.yticks(range(len(feature_names)), feature_names)
                    plt.xlabel('Aggregated SHAP Score')
                    plt.title(f'Top {top_k_important} Features by Aggregated SHAP Score - {technique}')
                    plt.gca().invert_yaxis()
                    plt.tight_layout()
                    plt.savefig(plots_dir / f"top_features_barplot_{technique}.png", 
                               dpi=300, bbox_inches='tight', facecolor='white')
                else:
                    print("Warning: aggregated_shap array is empty")
                    plt.text(0.5, 0.5, 'Empty aggregated SHAP data', 
                            transform=plt.gca().transAxes, ha='center', va='center')
                    plt.title(f'Top Features by Aggregated SHAP Score - {technique}')
                    plt.savefig(plots_dir / f"top_features_barplot_{technique}.png", 
                               dpi=300, bbox_inches='tight', facecolor='white')
            
            else:
                print(f"Warning: Unexpected aggregated_shap type: {type(aggregated_shap)}")
                plt.text(0.5, 0.5, f'Unsupported SHAP data type: {type(aggregated_shap)}', 
                        transform=plt.gca().transAxes, ha='center', va='center')
                plt.title(f'Top Features by Aggregated SHAP Score - {technique}')
                plt.savefig(plots_dir / f"top_features_barplot_{technique}.png", 
                           dpi=300, bbox_inches='tight', facecolor='white')
        
        except Exception as plot_error:
            print(f"Error creating SHAP barplot: {plot_error}")
            plt.text(0.5, 0.5, f'Error creating plot: {str(plot_error)[:50]}...', 
                    transform=plt.gca().transAxes, ha='center', va='center')
            plt.title(f'Top Features by Aggregated SHAP Score - {technique}')
            plt.savefig(plots_dir / f"top_features_barplot_{technique}.png", 
                       dpi=300, bbox_inches='tight', facecolor='white')
        
        plt.close()
    
    # Plot cluster distribution
    plt.figure(figsize=(10, 6))
    unique_labels, counts = np.unique(predict_label, return_counts=True)
    plt.bar(range(len(unique_labels)), counts)
    plt.xlabel('Cluster ID')
    plt.ylabel('Number of Cells')
    plt.title(f'Cluster Distribution - {technique}')
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
    
    with open(Path(output_dir) / results_files['all_type_features'], 'wb') as file:
        pickle.dump(all_type_features, file)
    with open(Path(output_dir) / results_files['scores'], 'wb') as file:
        pickle.dump(scores, file)
    with open(Path(output_dir) / results_files['aggregated_shap'], 'wb') as file:
        pickle.dump(aggregated_shap, file)
    
    print(f"\nResults saved:")
    for desc, filename in results_files.items():
        print(f"  {desc}: {Path(output_dir) / filename}")
    
    print(f"\nPlots saved in: {plots_dir}")
    plot_files = list(plots_dir.glob(f"*.png"))
    for plot_file in sorted(plot_files):
        print(f"  📊 {plot_file.name}")
    
    return all_type_features, scores, aggregated_shap


def main():
    parser = argparse.ArgumentParser(
        description="Calculate SHAP values for UnitedNet joint group identification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model arguments
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to trained model pickle file')
    parser.add_argument('--dataset_id', type=str, default='KP2_1',
                       help='Dataset identifier (e.g., KP2_1)')
    parser.add_argument('--technique', type=str, default='perturbmap',
                       help='Training technique used')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, 
                       default='../Data/UnitedNet/input_data',
                       help='Path to directory containing input data files')
    parser.add_argument('--model_base_path', type=str, default='../Model',
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
    output_dir = base_output_dir / f"shap_analysis_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Results will be saved to: {output_dir}")
    
    try:
        # Find or load model
        if args.model_path and Path(args.model_path).exists():
            model_path = args.model_path
            print(f"Using specified model: {model_path}")
        else:
            print("Searching for latest model...")
            model_path = find_latest_model(
                base_path=args.model_base_path,
                dataset_id=args.dataset_id,
                technique=args.technique
            )
            if not model_path:
                raise FileNotFoundError(f"No model found for dataset {args.dataset_id}")
        
        # Load model and data
        model, adata_dict = load_model_and_data(model_path, args.data_path, args.dataset_id)
        
        # Prepare data
        print("\n=== Preparing Data ===")
        adatas_all = prepare_data_for_shap(adata_dict)
        
        print(f"Prepared data shapes:")
        for i, (name, adata) in enumerate(zip(["RNA", "NICHE", "ACTIVITY"], adatas_all)):
            print(f"  {name}: {adata.shape}")
        
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
            print(f"\n=== Calculating SHAP Values ===")
            shap_values, predict_label, predict_label_anno, adata_fused, adatas_all = calculate_shap_values(
                model, adatas_all, args.technique, output_dir
            )
        
        # Create chord plots
        print(f"\n=== Creating Chord Plots ===")
        all_type_features, scores, aggregated_shap = create_chord_plots(
            shap_values, predict_label, predict_label_anno, adata_fused, adatas_all,
            args.technique, args.top_k_features, output_dir
        )
        
        print(f"\n=== Analysis Complete ===")
        print(f"Results saved in: {output_dir}")
        
    except Exception as e:
        print(f"\nError during SHAP analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())