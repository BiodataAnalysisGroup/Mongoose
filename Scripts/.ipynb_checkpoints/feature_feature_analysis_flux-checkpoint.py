#!/usr/bin/env python3
"""
Feature-to-Feature Cross-Modal Prediction Analysis for UnitedNet with Flux Modality

This script performs cross-modal prediction analysis between different modalities
and calculates feature-to-feature relevance using SHAP values.

Usage:
    python feature_feature_analysis.py --dataset_id KP2_1 --shap_dir ../Task1_JGI/shap_analysis_flux_20250911_143045
"""

import argparse
import os
import pickle
import pandas as pd
import numpy as np
import torch
from datetime import datetime
from pathlib import Path

# Import UnitedNet components
from unitednet.interface import UnitedNet
from unitednet.modules import submodel_trans
from unitednet.data import type_specific_mean
from unitednet.plots import feature_relevance_chord_plot
import shap


def find_latest_shap_dir(base_path="../Task1_JGI", pattern="shap_analysis_flux_*"):
    """Find the most recent SHAP analysis directory"""
    import glob
    import os
    
    base_path = Path(base_path)
    if not base_path.exists():
        print(f"Base SHAP path does not exist: {base_path}")
        return None
    
    # Search for SHAP directories with different patterns
    search_patterns = [
        str(base_path / pattern),
        str(base_path / "shap_analysis_*flux*"),
        str(base_path / "shap_analysis_*"),
    ]
    
    all_dirs = []
    for search_pattern in search_patterns:
        matching_dirs = glob.glob(search_pattern)
        all_dirs.extend([d for d in matching_dirs if os.path.isdir(d)])
    
    if not all_dirs:
        print(f"No SHAP directories found in {base_path}")
        print("Searched patterns:")
        for pattern in search_patterns:
            print(f"  {pattern}")
        return None
    
    # Remove duplicates and sort by modification time
    all_dirs = list(set(all_dirs))
    all_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    print(f"Found {len(all_dirs)} SHAP directories:")
    for i, shap_dir in enumerate(all_dirs[:5]):  # Show top 5
        mtime = datetime.fromtimestamp(os.path.getmtime(shap_dir))
        print(f"  {i+1}. {Path(shap_dir).name} (modified: {mtime})")
    
    return all_dirs[0]


def load_shap_results(shap_dir):
    """Load SHAP analysis results from previous step"""
    shap_dir = Path(shap_dir)
    
    print(f"Looking for SHAP results in: {shap_dir}")
    print(f"Absolute path: {shap_dir.absolute()}")
    print(f"Directory exists: {shap_dir.exists()}")
    
    if not shap_dir.exists():
        raise FileNotFoundError(f"SHAP directory not found: {shap_dir}")
    
    print("Files in directory:")
    all_files = list(shap_dir.iterdir())
    for file in all_files:
        print(f"  {file.name}")
    
    # Load the main SHAP results - CHANGED TO LOOK FOR FLUX FILES
    results = {}
    files_to_load = {
        'all_type_features': 'all_type_features_perturbmap_flux.pkl',  # CHANGED
        'scores': 'scores_perturbmap_flux.pkl',  # CHANGED
        'aggregated_shap': 'aggregated_shap_perturbmap_flux.pkl',  # CHANGED
        'shap_values': 'shap_values_perturbmap_flux.pkl'  # CHANGED
    }
    
    for key, filename in files_to_load.items():
        filepath = shap_dir / filename
        print(f"Checking for {key}: {filepath}")
        print(f"  File exists: {filepath.exists()}")
        
        if filepath.exists():
            try:
                with open(filepath, 'rb') as f:
                    results[key] = pickle.load(f)
                print(f"Successfully loaded {key} from {filepath}")
            except Exception as e:
                print(f"Error loading {key}: {e}")
        else:
            print(f"Warning: {filepath} not found")
            # Try alternative patterns - UPDATED FOR FLUX
            pattern_matches = list(shap_dir.glob(f"*{key}*flux*.pkl"))
            if not pattern_matches:
                pattern_matches = list(shap_dir.glob(f"*{key}*.pkl"))
            if pattern_matches:
                print(f"  Found alternative file: {pattern_matches[0]}")
                try:
                    with open(pattern_matches[0], 'rb') as f:
                        results[key] = pickle.load(f)
                    print(f"Successfully loaded {key} from {pattern_matches[0]}")
                except Exception as e:
                    print(f"Error loading {key} from alternative: {e}")
    
    print(f"Loaded results keys: {list(results.keys())}")
    return results


def find_latest_model(base_path="../Model", dataset_id=None, technique="perturbmap_flux"):  # CHANGED
    """Find the most recent model file"""
    import glob
    
    search_patterns = []
    if dataset_id:
        # UPDATED PATTERNS FOR FLUX
        search_patterns.extend([
            f"{base_path}/{technique}_*/model_{technique}_{dataset_id}_*.pkl",
            f"{base_path}/{technique}/model_{technique}_{dataset_id}_*.pkl",
            f"{base_path}/perturbmap_*/model_perturbmap_flux_{dataset_id}_*.pkl",
            f"{base_path}/perturbmap/model_perturbmap_flux_{dataset_id}_*.pkl",
        ])
    
    search_patterns.extend([
        f"{base_path}/{technique}_*/model_{technique}_*.pkl",
        f"{base_path}/{technique}/model_{technique}_*.pkl",
        f"{base_path}/*/model_perturbmap_flux_*.pkl",
    ])
    
    all_models = []
    for pattern in search_patterns:
        all_models.extend(glob.glob(pattern))
    
    if not all_models:
        return None
    
    all_models.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return all_models[0]


def prepare_data_for_cross_modal(data_path, dataset_id):
    """Prepare data for cross-modal analysis"""
    import scanpy as sc
    import scipy.sparse as sp
    
    def change_label(adata, batch):
        adata.obs['batch'] = batch
        adata.obs['imagecol'] = adata.obs['array_col']
        adata.obs['imagerow'] = adata.obs['array_row'] 
        adata.obs['label'] = adata.obs['phenotypes']
        return adata

    def ensure_dense_float32(adata):
        if sp.issparse(adata.X):
            adata.X = adata.X.tocsr()
            adata.X.sort_indices()
            adata.X = adata.X.astype(np.float32).toarray()
        else:
            adata.X = np.asarray(adata.X, dtype=np.float32)
        return adata
    
    # Load data files - CHANGED FROM ACTIVITY TO FLUX
    files_to_load = {
        'adata_rna_train': f'adata_rna_train_perturbmap_{dataset_id}.h5ad',
        'adata_niche_train': f'adata_niche_train_perturbmap_{dataset_id}.h5ad', 
        'adata_flux_train': f'adata_flux_train_perturbmap_{dataset_id}.h5ad',  # CHANGED
        'adata_rna_test': f'adata_rna_test_perturbmap_{dataset_id}.h5ad',
        'adata_niche_test': f'adata_niche_test_perturbmap_{dataset_id}.h5ad',
        'adata_flux_test': f'adata_flux_test_perturbmap_{dataset_id}.h5ad'  # CHANGED
    }
    
    adata_dict = {}
    for key, filename in files_to_load.items():
        filepath = Path(data_path) / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Required file not found: {filepath}")
        adata_dict[key] = sc.read_h5ad(filepath)
    
    # Process data - CHANGED FROM ACTIVITY TO FLUX
    for key in ['adata_rna_train', 'adata_niche_train', 'adata_flux_train']:
        adata_dict[key] = change_label(adata_dict[key], 'train')
    for key in ['adata_rna_test', 'adata_niche_test', 'adata_flux_test']:
        adata_dict[key] = change_label(adata_dict[key], 'test')
    
    # Merge train+test per modality - CHANGED FROM ACTIVITY TO FLUX
    adatas_all = []
    for train_key, test_key in [('adata_rna_train', 'adata_rna_test'),
                                ('adata_niche_train', 'adata_niche_test'),
                                ('adata_flux_train', 'adata_flux_test')]:  # CHANGED
        ad_all = adata_dict[train_key].concatenate(adata_dict[test_key], batch_key='sample')
        ad_all = change_label(ad_all, 'test')
        adatas_all.append(ensure_dense_float32(ad_all))
    
    return adatas_all


def compute_cross_modal_shap(model, adatas_all, source_modality, target_modality, 
                           output_dir, calculate_shap=True):
    """Compute SHAP values for cross-modal prediction"""
    
    modality_names = ["RNA", "NICHE", "FLUX"]  # CHANGED FROM ACTIVITY TO FLUX
    
    print(f"Computing SHAP: {modality_names[source_modality]} -> {modality_names[target_modality]}")
    
    # Show flux-specific info when relevant
    if source_modality == 2 or target_modality == 2:
        flux_data = adatas_all[2]
        print(f"Flux modality info:")
        print(f"  Metabolites: {flux_data.n_vars}")
        if 'metabolite_category' in flux_data.var.columns:
            print(f"  Categories: {flux_data.var['metabolite_category'].value_counts().to_dict()}")
        if 'has_flux_data' in flux_data.obs.columns:
            print(f"  Cells with actual data: {flux_data.obs['has_flux_data'].sum()}")
    
    shap_file = Path(output_dir) / f'shap_values_{source_modality}_{target_modality}.pkl'
    
    if calculate_shap:
        # Create submodel for translation
        sub = submodel_trans(model.model, [source_modality, target_modality]).to(model.device)
        
        # Use cluster prototypes as background
        cluster_prototype_features = [
            type_specific_mean(ad_x, 'label').to(device=model.device) for ad_x in adatas_all
        ]
        background = cluster_prototype_features[source_modality]
        
        # Create SHAP explainer
        e = shap.DeepExplainer(sub, background)
        
        # Prepare test data
        test_type = torch.tensor(adatas_all[source_modality].X, device=model.device, dtype=torch.float32)
        
        print(f"Background shape: {background.shape}")
        print(f"Test data shape: {test_type.shape}")
        
        # Compute SHAP values
        try:
            shap_values = e.shap_values(test_type, check_additivity=True)
        except Exception as e:
            print(f"Error with check_additivity=True: {e}")
            print("Retrying without additivity check...")
            shap_values = e.shap_values(test_type, check_additivity=False)
        
        # Save SHAP values
        with open(shap_file, 'wb') as f:
            pickle.dump(shap_values, f)
        print(f"SHAP values saved to: {shap_file}")
        
    else:
        # Load existing SHAP values
        if not shap_file.exists():
            raise FileNotFoundError(f"SHAP file not found: {shap_file}")
        
        with open(shap_file, 'rb') as f:
            shap_values = pickle.load(f)
        print(f"SHAP values loaded from: {shap_file}")
    
    return shap_values


def str_to_mod_id(x):
    """Convert modality string/int to standardized ID"""
    s = str(x).lower()
    if s in ("0", "rna", "gene", "g"): return 0
    if s in ("1", "niche", "n"): return 1
    if s in ("2", "flux", "f", "metabolite", "met"): return 2  # CHANGED FROM ACTIVITY
    try: return int(s)
    except: raise KeyError(f"Unknown modality key: {x}")


def normalize_all_type_features(all_type_features):
    """Normalize cluster keys and modality subkeys in all_type_features"""
    print(f"Debug: Original all_type_features keys: {list(all_type_features.keys())}")
    
    atf_norm = {}
    for k, sub in all_type_features.items():
        # Handle cluster keys - convert 'Cluster_X' to integer X
        if isinstance(k, str) and k.startswith('Cluster_'):
            try:
                cluster_id = int(k.split('_')[1])
            except (IndexError, ValueError):
                print(f"Warning: Could not parse cluster ID from '{k}', keeping as-is")
                cluster_id = k
        elif str(k).isdigit():
            cluster_id = int(k)
        else:
            cluster_id = k
        
        # Handle modality subkeys
        normalized_sub = {}
        for sk, sv in sub.items():
            try:
                mod_id = str_to_mod_id(sk)
                normalized_sub[mod_id] = sv
            except KeyError as e:
                print(f"Warning: {e}, keeping original key '{sk}'")
                normalized_sub[sk] = sv
        
        atf_norm[cluster_id] = normalized_sub
    
    print(f"Debug: Normalized all_type_features keys: {list(atf_norm.keys())}")
    return atf_norm


def feature_feature_relevance(shap_values_dict, all_type_features, var_names_all, 
                            source_id, target_id, technique="perturbmap_flux", make_plot=False):  # CHANGED
    """Compute feature-to-feature relevance between modalities"""
    
    # Prepare variable names
    var_names = [np.array(var_names_all[source_id]), np.array(var_names_all[target_id])]
    
    # Normalize all_type_features
    atf_norm = normalize_all_type_features(all_type_features)
    
    # Get clusters from pickles
    clusters_from_pickles = np.array(sorted([k for k in atf_norm.keys() if isinstance(k, int)]))
    
    # Get SHAP values for this modality pair
    shap_key = f"{source_id}_{target_id}"
    if shap_key not in shap_values_dict:
        raise KeyError(f"SHAP values not found for {shap_key}")
    
    return feature_relevance_chord_plot(
        shap_values_dict[shap_key],
        clusters_from_pickles,
        var_names,
        atf_norm,
        f"{technique}_average",
        direction=f"{source_id}to{target_id}",
        in_mod=0,
        thres=None,
        make_plot=make_plot,
        potential_coloarmaps=["spring", "summer", "winter", "autumn"],
    )


def extract_feature_feature_values(shap_values_dict, all_type_features, var_names_all, 
                                 combinations, technique="perturbmap_flux"):  # CHANGED
    """Extract feature-feature values for all combinations"""
    
    id2modality = {0: "RNA", 1: "NICHE", 2: "FLUX"}  # CHANGED FROM ACTIVITY TO FLUX
    rows = []
    
    def feature_feature_values(source_index, target_index):
        """Extract feature-feature values for a given source-target pair"""
        print(f"Processing {id2modality[source_index]} -> {id2modality[target_index]}")
        
        # Check if SHAP values exist for this combination
        shap_key = f"{source_index}_{target_index}"
        if shap_key not in shap_values_dict:
            print(f"Warning: SHAP values not found for {id2modality[source_index]} -> {id2modality[target_index]}")
            return
        
        # Debug information
        print(f"  SHAP values type: {type(shap_values_dict[shap_key])}")
        if hasattr(shap_values_dict[shap_key], 'shape'):
            print(f"  SHAP values shape: {shap_values_dict[shap_key].shape}")
        elif isinstance(shap_values_dict[shap_key], list):
            print(f"  SHAP values list length: {len(shap_values_dict[shap_key])}")
            if len(shap_values_dict[shap_key]) > 0:
                print(f"  First element type: {type(shap_values_dict[shap_key][0])}")
                if hasattr(shap_values_dict[shap_key][0], 'shape'):
                    print(f"  First element shape: {shap_values_dict[shap_key][0].shape}")
        
        print(f"  all_type_features keys: {list(all_type_features.keys())}")
        print(f"  var_names_all lengths: source={len(var_names_all[source_index])}, target={len(var_names_all[target_index])}")
        
        # Show flux-specific info when relevant
        if source_index == 2:
            print(f"  Source (FLUX) metabolite sample: {var_names_all[source_index][:5]}")
        if target_index == 2:
            print(f"  Target (FLUX) metabolite sample: {var_names_all[target_index][:5]}")
        
        try:
            values, labels = feature_feature_relevance(
                shap_values_dict, all_type_features, var_names_all,
                source_index, target_index, technique, make_plot=False
            )
            
            print(f"  Returned values type: {type(values)}")
            print(f"  Returned labels type: {type(labels)}")
            
            if isinstance(values, dict):
                print(f"  Values keys: {list(values.keys())}")
                print(f"  Labels keys: {list(labels.keys())}")
            
            if not values:
                print(f"  Warning: Empty values returned for {id2modality[source_index]} -> {id2modality[target_index]}")
                return
            
            for cluster_id, connections in values.items():
                print(f"  Processing cluster {cluster_id}")
                print(f"    Connections type: {type(connections)}")
                if hasattr(connections, 'shape'):
                    print(f"    Connections shape: {connections.shape}")
                
                cluster_labels = labels[cluster_id]
                source_labels = cluster_labels[0]
                target_labels = cluster_labels[1]
                
                print(f"    Source labels length: {len(source_labels)}")
                print(f"    Target labels length: {len(target_labels)}")
                
                target2source = np.transpose(connections)
                
                for target_label_id, target_label in enumerate(target_labels):
                    for source_label_id, source_label in enumerate(source_labels):
                        # Enhanced feature identification for flux
                        source_feature = source_label
                        target_feature = target_label
                        
                        if source_index == 2:  # Source is flux
                            source_feature = f"FLUX:{source_label}"
                        if target_index == 2:  # Target is flux  
                            target_feature = f"FLUX:{target_label}"
                        
                        rows.append({
                            "Cluster": cluster_id,
                            "Direction": f"{id2modality[source_index]} -> {id2modality[target_index]}",
                            "Target": target_feature,
                            "Source": source_feature,
                            "Value": target2source[target_label_id][source_label_id],
                        })
                
                print(f"    Added {len(target_labels) * len(source_labels)} rows for cluster {cluster_id}")
        
        except Exception as e:
            print(f"Error processing {id2modality[source_index]} -> {id2modality[target_index]}: {e}")
            import traceback
            traceback.print_exc()
    
    # Process all combinations
    print("Processing feature-feature relationships...")
    for source_idx, target_idx in combinations:
        feature_feature_values(source_idx, target_idx)
    
    print(f"Total rows collected: {len(rows)}")
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Feature-to-feature cross-modal prediction analysis for UnitedNet with Flux modality",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input arguments
    parser.add_argument('--dataset_id', type=str, default='KP2_1',
                       help='Dataset identifier')
    parser.add_argument('--shap_dir', type=str, default=None,
                       help='Directory containing SHAP analysis results from previous step (auto-detected if not provided)')
    parser.add_argument('--shap_base_path', type=str, default='../Task1_JGI',
                       help='Base path to search for SHAP directories')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to trained model (auto-detected if not provided)')
    parser.add_argument('--data_path', type=str, default='../Data/UnitedNet/input_data',
                       help='Path to input data directory')
    
    # Analysis arguments
    parser.add_argument('--calculate_shap', action='store_true', default=True,
                       help='Calculate new SHAP values for cross-modal prediction')
    parser.add_argument('--no_shap_calc', dest='calculate_shap', action='store_false',
                       help='Use existing SHAP values only')
    parser.add_argument('--combinations', type=str, nargs='+', 
                       default=['0,1', '0,2', '1,2'],  # ADDED 1,2 FOR NICHE->FLUX
                       help='Modality combinations to analyze (format: "source,target")')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='../Task2_CMP',
                       help='Base output directory')
    parser.add_argument('--timestamp', type=str, default=None,
                       help='Custom timestamp for output directory (format: YYYYMMDD_HHMMSS)')
    parser.add_argument('--auto_timestamp', action='store_true', default=True,
                       help='Auto-generate timestamp (default)')
    parser.add_argument('--no_timestamp', dest='auto_timestamp', action='store_false',
                       help='Use technique name only without timestamp')
    
    args = parser.parse_args()
    
    # Handle timestamp options
    if args.timestamp:
        # Use custom timestamp
        timestamp = args.timestamp
        output_dir = Path(args.output_dir) / f"cross_modal_flux_analysis_{timestamp}"  # CHANGED
    elif args.auto_timestamp:
        # Auto-generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(args.output_dir) / f"cross_modal_flux_analysis_{timestamp}"  # CHANGED
    else:
        # No timestamp
        timestamp = "manual"
        output_dir = Path(args.output_dir) / "cross_modal_flux_analysis"  # CHANGED
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")
    
    try:
        # Find or use SHAP directory
        if args.shap_dir and Path(args.shap_dir).exists():
            shap_dir = args.shap_dir
            print(f"Using specified SHAP directory: {shap_dir}")
        else:
            if args.shap_dir:
                print(f"Specified SHAP directory not found: {args.shap_dir}")
            print("Searching for latest SHAP directory...")
            shap_dir = find_latest_shap_dir(base_path=args.shap_base_path)
            if not shap_dir:
                raise FileNotFoundError(f"No SHAP directories found in {args.shap_base_path}")
            print(f"Using auto-detected SHAP directory: {shap_dir}")
        
        # Load SHAP results from previous analysis
        print("\n=== Loading SHAP Results ===")
        shap_results = load_shap_results(shap_dir)
        all_type_features = shap_results.get('all_type_features')
        
        if all_type_features is None:
            raise ValueError("all_type_features not found in SHAP results")
        
        # Find and load model
        if args.model_path and Path(args.model_path).exists():
            model_path = args.model_path
        else:
            print("Searching for latest flux model...")
            model_path = find_latest_model(dataset_id=args.dataset_id)
            if not model_path:
                raise FileNotFoundError(f"No flux model found for dataset {args.dataset_id}")
        
        print(f"Loading model from: {model_path}")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Prepare data
        print("\n=== Preparing Data ===")
        adatas_all = prepare_data_for_cross_modal(args.data_path, args.dataset_id)
        
        # Get variable names for all modalities
        var_names_all = [list(adata.var_names) for adata in adatas_all]
        
        print(f"Variable counts:")
        modality_names = ["RNA", "NICHE", "FLUX"]
        for i, (var_names, name) in enumerate(zip(var_names_all, modality_names)):
            print(f"  {name}: {len(var_names)} features")
            if name == "FLUX":
                if 'metabolite_category' in adatas_all[i].var.columns:
                    print(f"    Metabolite categories: {adatas_all[i].var['metabolite_category'].value_counts().to_dict()}")
                print(f"    Sample metabolites: {var_names[:5]}")
        
        # Parse combinations
        combinations = []
        for comb_str in args.combinations:
            source, target = map(int, comb_str.split(','))
            combinations.append([source, target])
        
        print(f"Analyzing {len(combinations)} combinations: {combinations}")
        for source_idx, target_idx in combinations:
            print(f"  {modality_names[source_idx]} -> {modality_names[target_idx]}")
        
        # Compute cross-modal SHAP values
        print("\n=== Computing Cross-Modal SHAP Values ===")
        shap_values_dict = {}
        
        for source_idx, target_idx in combinations:
            print(f"\nComputing {modality_names[source_idx]} -> {modality_names[target_idx]}...")
            
            shap_values = compute_cross_modal_shap(
                model, adatas_all, source_idx, target_idx, 
                output_dir, args.calculate_shap
            )
            shap_values_dict[f"{source_idx}_{target_idx}"] = shap_values
        
        # Extract feature-feature relationships
        print("\n=== Extracting Feature-Feature Relationships ===")
        df = extract_feature_feature_values(
            shap_values_dict, all_type_features, var_names_all, 
            combinations, technique="perturbmap_flux"  # CHANGED
        )
        
        # Save results
        feature_feature_file = output_dir / 'feature_feature_importance_flux_3modalities.csv'  # CHANGED
        df.to_csv(feature_feature_file, index=False)
        
        print(f"\n=== Analysis Complete ===")
        print(f"Total relationships extracted: {len(df)}")
        print(f"Combinations analyzed: {len(combinations)}")
        print(f"Results saved to: {feature_feature_file}")
        print(f"Modalities analyzed: RNA + Niche + Flux")
        
        if len(df) > 0:
            print("\nSample of results:")
            print(df.head(10))
            
            print("\nSummary by direction:")
            print(df.groupby('Direction').size())
            
            # Show flux-specific summary
            flux_rows = df[df['Direction'].str.contains('FLUX')]
            if len(flux_rows) > 0:
                print(f"\nFlux-related relationships: {len(flux_rows)}")
                print("Flux directions:")
                print(flux_rows.groupby('Direction').size())
        
        # Save analysis summary
        summary = {
            'timestamp': timestamp,
            'dataset_id': args.dataset_id,
            'modalities': ['RNA', 'NICHE', 'FLUX'],
            'combinations_analyzed': combinations,
            'total_relationships': len(df),
            'flux_relationships': len(df[df['Direction'].str.contains('FLUX')]) if len(df) > 0 else 0,
            'shap_dir_used': shap_dir,  # CHANGED FROM args.shap_dir
            'model_path_used': model_path,
            'output_files': {
                'feature_relationships': str(feature_feature_file),
                'cross_modal_shap': [str(output_dir / f'shap_values_{s}_{t}.pkl') 
                                   for s, t in combinations]
            }
        }
        
        summary_file = output_dir / 'analysis_summary.json'
        import json
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Analysis summary saved to: {summary_file}")
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())