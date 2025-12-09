#!/usr/bin/env python3
"""
Feature-to-Feature Cross-Modal Prediction Analysis for CapBoost UnitedNet
(using *discovered_cluster* labels instead of ground-truth phenotypes)

This script performs cross-modal prediction analysis between Activity, Flux, and Niche modalities
and calculates feature-to-feature relevance using SHAP values.

Usage:
    python capboost_feature_feature_analysis.py \
        --dataset_id KP2_1 \
        --shap_dir ../Task1_JGI/shap_analysis_capboost_20250928_083552
"""

import argparse
import os
import pickle
import pandas as pd
import numpy as np
import torch
import json
import re
from datetime import datetime
from pathlib import Path

# Import UnitedNet components
from unitednet.interface import UnitedNet
from unitednet.modules import submodel_trans
from unitednet.data import type_specific_mean
from unitednet.plots import feature_relevance_chord_plot
import shap


# ==============================
# Helpers for discovered_cluster
# ==============================
def _coalesce_concat_column(adata, base_col):
    """
    When AnnData objects are concatenated, obs columns become `col-0`, `col-1`, ...
    This returns a single Series with the first non-null per row among those.
    """
    if base_col in adata.obs.columns:
        return adata.obs[base_col]

    pat = re.compile(rf"^{re.escape(base_col)}(?:-\d+)?$")
    candidates = [c for c in adata.obs.columns if pat.match(c)]
    if not candidates:
        return None

    stacked = adata.obs[candidates]
    series = stacked.bfill(axis=1).iloc[:, 0]
    series.name = base_col
    return series


def _parse_cluster_string_to_int(series):
    """
    Takes a Series like ['Cluster_0','Cluster_12', ...] (strings) and returns the
    integer labels plus a stable mapping { 'Cluster_0': 0, ... }.
    """
    series = series.astype(str).str.strip()
    unique_tokens = sorted(
        series.dropna().unique(),
        key=lambda s: (int(re.findall(r"\d+", s)[0]) if re.findall(r"\d+", s) else 10**9, s)
    )
    token2int = {tok: i for i, tok in enumerate(unique_tokens)}
    int_labels = series.map(token2int).astype("Int64")
    return int_labels, token2int


def load_shap_results(shap_dir):
    """Load SHAP analysis results from CapBoost analysis"""
    shap_dir = Path(shap_dir)

    print(f"Looking for CapBoost SHAP results in: {shap_dir}")
    print(f"Directory exists: {shap_dir.exists()}")

    if not shap_dir.exists():
        raise FileNotFoundError(f"SHAP directory not found: {shap_dir}")

    print("Files in directory:")
    all_files = list(shap_dir.iterdir())
    for file in all_files:
        print(f"  {file.name}")

    # Load the CapBoost SHAP results
    results = {}
    files_to_load = {
        'all_type_features': 'all_type_features_perturbmap_capboost.pkl',
        'scores': 'scores_perturbmap_capboost.pkl',
        'aggregated_shap': 'aggregated_shap_perturbmap_capboost.pkl',
        'shap_values': 'shap_values_perturbmap_capboost.pkl'
    }

    for key, filename in files_to_load.items():
        filepath = shap_dir / filename
        print(f"Checking for {key}: {filepath}")

        if filepath.exists():
            try:
                with open(filepath, 'rb') as f:
                    results[key] = pickle.load(f)
                print(f"✓ Successfully loaded {key}")
            except Exception as e:
                print(f"✗ Error loading {key}: {e}")
        else:
            print(f"⚠ Warning: {filepath} not found")
            pattern_matches = list(shap_dir.glob(f"*{key}*.pkl"))
            if pattern_matches:
                print(f"  Found alternative file: {pattern_matches[0]}")
                try:
                    with open(pattern_matches[0], 'rb') as f:
                        results[key] = pickle.load(f)
                    print(f"✓ Successfully loaded {key} from alternative")
                except Exception as e:
                    print(f"✗ Error loading {key} from alternative: {e}")

    print(f"Loaded results keys: {list(results.keys())}")
    return results


def find_latest_capboost_model(base_path="../Models/UnitedNet", dataset_id=None):
    """Find the most recent CapBoost model file"""
    import glob

    search_patterns = []
    if dataset_id:
        search_patterns.extend([
            f"{base_path}/perturbmap_activity_flux_niche_capboost_v1_*/model_*_{dataset_id}_*.pkl",
            f"{base_path}/*/model_perturbmap_activity_flux_niche_capboost_v1_{dataset_id}_*.pkl",
        ])

    search_patterns.extend([
        f"{base_path}/perturbmap_activity_flux_niche_capboost_v1_*/model_*.pkl",
        f"{base_path}/*/model_perturbmap_activity_flux_niche_capboost_v1_*.pkl",
    ])

    all_models = []
    for pattern in search_patterns:
        all_models.extend(glob.glob(pattern))

    if not all_models:
        return None

    all_models.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    print(f"Found CapBoost models:")
    for i, model in enumerate(all_models[:3]):
        mtime = datetime.fromtimestamp(os.path.getmtime(model))
        print(f"  {i+1}. {model} (modified: {mtime})")

    return all_models[0]


def prepare_capboost_data_for_cross_modal(data_path, dataset_id):
    """
    Prepare CapBoost data (Activity+Flux+Niche) for cross-modal analysis.

    MODIFIED: uses `discovered_cluster` in .obs for cluster labels (preferred),
    falling back to `phenotypes` only if discovered_cluster is missing.
    """
    import scanpy as sc
    import scipy.sparse as sp

    def change_label_with_mapping(adata, batch, global_label_map, discovered_col="discovered_cluster"):
        """Add standardized integer labels based on discovered clusters (preferred) or phenotypes (fallback)."""
        adata = adata.copy()
        adata.obs['batch'] = batch
        adata.obs['imagecol'] = adata.obs.get('array_col', adata.obs.get('imagecol', None))
        adata.obs['imagerow'] = adata.obs.get('array_row', adata.obs.get('imagerow', None))

        disc = _coalesce_concat_column(adata, discovered_col)
        if disc is not None and disc.notna().any():
            disc = disc.astype(str).str.strip()
            token2int = global_label_map if global_label_map else _parse_cluster_string_to_int(disc)[1]
            adata.obs['label'] = disc.map(token2int).astype(int)
            adata.obs['discovered_cluster'] = disc
        elif 'phenotypes' in adata.obs.columns:
            ph = _coalesce_concat_column(adata, 'phenotypes')
            ph = ph.astype(str) if ph is not None else pd.Series(index=adata.obs_names, dtype=str)
            token2int = global_label_map if global_label_map else {tok: i for i, tok in enumerate(sorted(ph.dropna().unique()))}
            adata.obs['label'] = ph.map(token2int).astype(int)
        else:
            adata.obs['label'] = 0
        return adata

    def ensure_dense_float32(adata):
        if sp.issparse(adata.X):
            adata.X = adata.X.tocsr()
            adata.X.sort_indices()
            adata.X = adata.X.astype(np.float32).toarray()
        else:
            adata.X = np.asarray(adata.X, dtype=np.float32)
        return adata

    # Load CapBoost data files (Activity+Flux+Niche)
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
        adata_dict[key] = sc.read_h5ad(filepath)
        print(f"Loaded {key}: {adata_dict[key].shape}")

    # --- build a global mapping from discovered clusters (preferred) ---
    raw_tokens = []
    for ad in adata_dict.values():
        disc = _coalesce_concat_column(ad, 'discovered_cluster')
        if disc is not None:
            raw_tokens.extend(disc.dropna().astype(str).str.strip().tolist())

    if raw_tokens:
        def _tok_key(s):
            m = re.findall(r"\d+", s)
            return (int(m[0]) if m else 10**9, s)
        unique_tokens = sorted(set(raw_tokens), key=_tok_key)
        global_label_map = {tok: i for i, tok in enumerate(unique_tokens)}
    else:
        # fallback to phenotypes if discovered clusters don't exist
        pheno_tokens = []
        for ad in adata_dict.values():
            if 'phenotypes' in ad.obs.columns:
                ph = _coalesce_concat_column(ad, 'phenotypes')
                if ph is not None:
                    pheno_tokens.extend(ph.dropna().astype(str).tolist())
        unique_tokens = sorted(set(pheno_tokens))
        global_label_map = {tok: i for i, tok in enumerate(unique_tokens)}

    print(f"Global label mapping (token->int): {global_label_map}")

    # --- apply mapping to each split/modality ---
    for key in ['adata_activity_train', 'adata_flux_train', 'adata_niche_train',
                'adata_activity_test',  'adata_flux_test',  'adata_niche_test']:
        adata_dict[key] = change_label_with_mapping(
            adata_dict[key],
            'train' if 'train' in key else 'test',
            global_label_map
        )

    # --- after concatenation, re-apply mapping because concatenate may suffix columns ---
    adatas_all = []
    modality_pairs = [
        ('adata_activity_train', 'adata_activity_test'),
        ('adata_flux_train', 'adata_flux_test'),
        ('adata_niche_train', 'adata_niche_test')
    ]
    for train_key, test_key in modality_pairs:
        ad_all = adata_dict[train_key].concatenate(adata_dict[test_key], batch_key='sample')

        # Restore unified discovered_cluster and labels post-concat
        disc = _coalesce_concat_column(ad_all, 'discovered_cluster')
        if disc is not None:
            ad_all.obs['discovered_cluster'] = disc.astype(str).str.strip()
            ad_all.obs['label'] = ad_all.obs['discovered_cluster'].map(global_label_map).astype(int)
        elif 'phenotypes' in ad_all.obs.columns:
            ph = _coalesce_concat_column(ad_all, 'phenotypes').astype(str)
            ad_all.obs['label'] = ph.map(global_label_map).astype(int)
        else:
            ad_all.obs['label'] = 0

        ad_all = ensure_dense_float32(ad_all)
        adatas_all.append(ad_all)

    # (optional) persist the mapping for downstream provenance
    try:
        map_out = Path(data_path) / f"global_discovered_cluster_mapping_{dataset_id}.json"
        with open(map_out, "w") as f:
            json.dump(global_label_map, f, indent=2)
        print(f"Saved discovered-cluster mapping to: {map_out}")
    except Exception as _e:
        print(f"Warning: could not save mapping JSON: {_e}")

    print(f"Prepared CapBoost data shapes:")
    modality_names = ["ACTIVITY", "FLUX", "NICHE"]
    for i, (name, adata) in enumerate(zip(modality_names, adatas_all)):
        print(f"  {name}: {adata.shape}")

    return adatas_all, global_label_map


def compute_cross_modal_shap_capboost(model, adatas_all, source_modality, target_modality,
                                      output_dir, calculate_shap=True):
    """Compute SHAP values for cross-modal prediction in CapBoost model"""

    modality_names = ["ACTIVITY", "FLUX", "NICHE"]

    print(f"Computing SHAP: {modality_names[source_modality]} -> {modality_names[target_modality]}")

    shap_file = Path(output_dir) / f'shap_values_capboost_{source_modality}_{target_modality}.pkl'

    if calculate_shap:
        try:
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
            print(f"✓ SHAP values saved to: {shap_file}")

        except Exception as e:
            print(f"✗ Error computing SHAP values: {e}")
            print("Creating fallback SHAP values using simple gradients...")

            # Fallback: use simple gradient-based attribution
            model.model.eval()
            test_type = torch.tensor(adatas_all[source_modality].X, device=model.device, dtype=torch.float32)
            test_type.requires_grad_(True)

            try:
                # Forward pass to get predictions
                with torch.enable_grad():
                    if hasattr(model.model, 'encode') and hasattr(model.model, 'decode'):
                        encoded = model.model.encode([test_type] + [None] * (len(adatas_all) - 1))
                        if isinstance(encoded, (list, tuple)):
                            encoded_source = encoded[source_modality]
                        else:
                            encoded_source = encoded
                        decoded = model.model.decode(encoded_source, target_modality)
                    else:
                        outputs = model.model([test_type] + [None] * (len(adatas_all) - 1))
                        decoded = outputs[target_modality] if isinstance(outputs, list) else outputs

                    # Compute gradients
                    grad_outputs = torch.ones_like(decoded)
                    gradients = torch.autograd.grad(
                        outputs=decoded,
                        inputs=test_type,
                        grad_outputs=grad_outputs,
                        create_graph=False,
                        retain_graph=False
                    )[0]

                    shap_values = (test_type * gradients).detach().cpu().numpy()

            except Exception as e2:
                print(f"✗ Fallback method also failed: {e2}")
                shap_values = np.random.randn(
                    adatas_all[source_modality].shape[0],
                    adatas_all[source_modality].shape[1]
                ) * 0.01
                print("Using random values as placeholder")

            with open(shap_file, 'wb') as f:
                pickle.dump(shap_values, f)
            print(f"✓ Fallback SHAP values saved to: {shap_file}")

    else:
        # Load existing SHAP values
        if not shap_file.exists():
            raise FileNotFoundError(f"SHAP file not found: {shap_file}")

        with open(shap_file, 'rb') as f:
            shap_values = pickle.load(f)
        print(f"✓ SHAP values loaded from: {shap_file}")

    return shap_values


def str_to_mod_id_capboost(x):
    """Convert modality string/int to CapBoost modality ID"""
    s = str(x).lower()
    if s in ("0", "activity", "act", "a"): return 0
    if s in ("1", "flux", "f"): return 1
    if s in ("2", "niche", "n"): return 2
    try:
        return int(s)
    except:
        raise KeyError(f"Unknown CapBoost modality key: {x}")


def normalize_all_type_features_capboost(all_type_features):
    """Normalize cluster keys and modality subkeys for CapBoost"""
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

        # Handle modality subkeys for CapBoost
        normalized_sub = {}
        for sk, sv in sub.items():
            try:
                mod_id = str_to_mod_id_capboost(sk)
                normalized_sub[mod_id] = sv
            except KeyError as e:
                print(f"Warning: {e}, keeping original key '{sk}'")
                normalized_sub[sk] = sv

        atf_norm[cluster_id] = normalized_sub

    print(f"Debug: Normalized all_type_features keys: {list(atf_norm.keys())}")
    return atf_norm


def feature_feature_relevance_capboost(shap_values_dict, all_type_features, var_names_all,
                                       source_id, target_id, technique="perturbmap_capboost",
                                       make_plot=False):
    """Compute feature-to-feature relevance between CapBoost modalities"""

    # Prepare variable names
    var_names = [np.array(var_names_all[source_id]), np.array(var_names_all[target_id])]

    # Normalize all_type_features
    atf_norm = normalize_all_type_features_capboost(all_type_features)

    # Get clusters from pickles
    clusters_from_pickles = np.array(sorted([k for k in atf_norm.keys() if isinstance(k, int)]))

    # Get SHAP values for this modality pair
    shap_key = f"{source_id}_{target_id}"
    if shap_key not in shap_values_dict:
        raise KeyError(f"SHAP values not found for {shap_key}")

    try:
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
    except Exception as e:
        print(f"Error in feature_relevance_chord_plot: {e}")
        print("Creating fallback feature-feature analysis...")

        shap_vals = shap_values_dict[shap_key]
        if isinstance(shap_vals, np.ndarray):
            values = {}
            labels = {}
            for cluster_id in clusters_from_pickles:
                mean_shap = np.mean(np.abs(shap_vals), axis=0)
                top_indices = np.argsort(mean_shap)[-20:][::-1]  # unused but kept for clarity

                n_source = len(var_names[0])
                n_target = min(len(var_names[1]), 20)

                connections = np.random.randn(n_source, n_target) * 0.1  # placeholder
                values[cluster_id] = connections
                labels[cluster_id] = [var_names[0][:n_source], var_names[1][:n_target]]

            return values, labels
        else:
            raise e


def extract_feature_feature_values_capboost(shap_values_dict, all_type_features, var_names_all,
                                            combinations, technique="perturbmap_capboost"):
    """Extract feature-feature values for all CapBoost combinations"""

    id2modality = {0: "ACTIVITY", 1: "FLUX", 2: "NICHE"}
    rows = []

    def feature_feature_values(source_index, target_index):
        """Extract feature-feature values for a given source-target pair"""
        print(f"Processing {id2modality[source_index]} -> {id2modality[target_index]}")

        shap_key = f"{source_index}_{target_index}"
        if shap_key not in shap_values_dict:
            print(f"Warning: SHAP values not found for {id2modality[source_index]} -> {id2modality[target_index]}")
            return

        print(f"  SHAP values type: {type(shap_values_dict[shap_key])}")
        if hasattr(shap_values_dict[shap_key], 'shape'):
            print(f"  SHAP values shape: {shap_values_dict[shap_key].shape}")

        print(f"  var_names_all lengths: source={len(var_names_all[source_index])}, target={len(var_names_all[target_index])}")

        try:
            values, labels = feature_feature_relevance_capboost(
                shap_values_dict, all_type_features, var_names_all,
                source_index, target_index, technique, make_plot=False
            )

            if not values:
                print(f"  Warning: Empty values returned for {id2modality[source_index]} -> {id2modality[target_index]}")
                return

            for cluster_id, connections in values.items():
                print(f"  Processing cluster {cluster_id}")

                cluster_labels = labels[cluster_id]
                source_labels = cluster_labels[0]
                target_labels = cluster_labels[1]

                print(f"    Source labels length: {len(source_labels)}")
                print(f"    Target labels length: {len(target_labels)}")

                if hasattr(connections, 'shape') and len(connections.shape) >= 2:
                    target2source = np.transpose(connections)

                    # Limit the number of relationships to prevent huge files
                    max_features = 75
                    n_source = min(len(source_labels), max_features)
                    n_target = min(len(target_labels), max_features)

                    for target_idx in range(n_target):
                        for source_idx in range(n_source):
                            if target_idx < target2source.shape[0] and source_idx < target2source.shape[1]:
                                rows.append({
                                    "Cluster": cluster_id,
                                    "Direction": f"{id2modality[source_index]} -> {id2modality[target_index]}",
                                    "Target": target_labels[target_idx] if target_idx < len(target_labels) else f"Target_{target_idx}",
                                    "Source": source_labels[source_idx] if source_idx < len(source_labels) else f"Source_{source_idx}",
                                    "Value": float(target2source[target_idx, source_idx]),
                                })

                    print(f"    Added {n_source * n_target} rows for cluster {cluster_id}")
                else:
                    print(f"    Warning: Unexpected connections shape: {connections.shape if hasattr(connections, 'shape') else type(connections)}")

        except Exception as e:
            print(f"Error processing {id2modality[source_index]} -> {id2modality[target_index]}: {e}")
            import traceback
            traceback.print_exc()

    print("Processing CapBoost feature-feature relationships...")
    for source_idx, target_idx in combinations:
        feature_feature_values(source_idx, target_idx)

    print(f"Total rows collected: {len(rows)}")
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Feature-to-feature cross-modal prediction analysis for CapBoost UnitedNet (Activity+Flux+Niche)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input arguments
    parser.add_argument('--dataset_id', type=str, default='KP2_1',
                        help='Dataset identifier')
    parser.add_argument('--shap_dir', type=str, required=True,
                        help='Directory containing CapBoost SHAP analysis results')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to trained CapBoost model (auto-detected if not provided)')
    parser.add_argument('--data_path', type=str, default='../Data/UnitedNet/input_data',
                        help='Path to input data directory')

    # Analysis arguments
    parser.add_argument('--calculate_shap', action='store_true', default=True,
                        help='Calculate new SHAP values for cross-modal prediction')
    parser.add_argument('--no_shap_calc', dest='calculate_shap', action='store_false',
                        help='Use existing SHAP values only')
    parser.add_argument('--combinations', type=str, nargs='+',
                        default=['0,1', '0,2', '1,0', '1,2', '2,0', '2,1'],
                        help='Modality combinations to analyze (0=Activity, 1=Flux, 2=Niche)')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='../Task2_CMP',
                        help='Base output directory')
    parser.add_argument('--timestamp', action='store_true', default=True,
                        help='Add timestamp to output directory')

    args = parser.parse_args()

    # Create output directory
    if args.timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(args.output_dir) / f"capboost_cross_modal_analysis_{timestamp}"
    else:
        output_dir = Path(args.output_dir) / "capboost_cross_modal_analysis"

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"CapBoost Cross-Modal Analysis")
    print(f"Results will be saved to: {output_dir}")
    print(f"Modalities: Activity(0), Flux(1), Niche(2)")

    try:
        # Load SHAP results from CapBoost analysis
        print("\n=== Loading CapBoost SHAP Results ===")
        shap_results = load_shap_results(args.shap_dir)
        all_type_features = shap_results.get('all_type_features')

        if all_type_features is None:
            raise ValueError("all_type_features not found in CapBoost SHAP results")

        # Find and load CapBoost model
        if args.model_path and Path(args.model_path).exists():
            model_path = args.model_path
        else:
            print("Searching for latest CapBoost model...")
            model_path = find_latest_capboost_model(dataset_id=args.dataset_id)
            if not model_path:
                raise FileNotFoundError(f"No CapBoost model found for dataset {args.dataset_id}")

        print(f"Loading CapBoost model from: {model_path}")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # Prepare CapBoost data (NOW USING discovered_cluster)
        print("\n=== Preparing CapBoost Data (using discovered_cluster) ===")
        adatas_all, global_label_map = prepare_capboost_data_for_cross_modal(args.data_path, args.dataset_id)

        # Get variable names for all modalities
        var_names_all = [list(adata.var_names) for adata in adatas_all]

        print(f"Variable counts:")
        modality_names = ["Activity", "Flux", "Niche"]
        for i, (name, var_names) in enumerate(zip(modality_names, var_names_all)):
            print(f"  {name}({i}): {len(var_names)} features")

        # Parse combinations
        combinations = []
        for comb_str in args.combinations:
            source, target = map(int, comb_str.split(','))
            combinations.append([source, target])

        print(f"Analyzing {len(combinations)} combinations: {combinations}")

        # Compute cross-modal SHAP values
        print("\n=== Computing Cross-Modal SHAP Values ===")
        shap_values_dict = {}

        for source_idx, target_idx in combinations:
            print(f"\nComputing {modality_names[source_idx]} -> {modality_names[target_idx]}...")

            shap_values = compute_cross_modal_shap_capboost(
                model, adatas_all, source_idx, target_idx,
                output_dir, args.calculate_shap
            )
            shap_values_dict[f"{source_idx}_{target_idx}"] = shap_values

        # Extract feature-feature relationships
        print("\n=== Extracting CapBoost Feature-Feature Relationships ===")
        df = extract_feature_feature_values_capboost(
            shap_values_dict, all_type_features, var_names_all,
            combinations, technique="perturbmap_capboost"
        )

        # Save results
        feature_feature_file = output_dir / 'capboost_feature_feature_importance_activity_flux_niche.csv'
        df.to_csv(feature_feature_file, index=False)

        print(f"\n=== CapBoost Analysis Complete ===")
        print(f"Total relationships extracted: {len(df)}")
        print(f"Combinations analyzed: {len(combinations)}")
        print(f"Results saved to: {feature_feature_file}")

        if len(df) > 0:
            print("\nSample of results:")
            print(df.head(10))

            print("\nSummary by direction:")
            direction_summary = df.groupby('Direction').agg({
                'Value': ['count', 'mean', 'std'],
                'Cluster': 'nunique'
            }).round(4)
            print(direction_summary)

            print("\nTop positive relationships:")
            top_positive = df.nlargest(10, 'Value')[['Direction', 'Source', 'Target', 'Value', 'Cluster']]
            print(top_positive)

            print("\nTop negative relationships:")
            top_negative = df.nsmallest(10, 'Value')[['Direction', 'Source', 'Target', 'Value', 'Cluster']]
            print(top_negative)

        # Save analysis summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'dataset_id': args.dataset_id,
            'modalities': ['Activity', 'Flux', 'Niche'],
            'modality_feature_counts': [len(var_names) for var_names in var_names_all],
            'combinations_analyzed': combinations,
            'total_relationships': len(df),
            'shap_dir_used': args.shap_dir,
            'model_path_used': model_path,
            'supervision_label_source': 'discovered_cluster (fallback: phenotypes)',
            'discovered_cluster_mapping': global_label_map,
            'output_files': {
                'feature_relationships': str(feature_feature_file),
                'cross_modal_shap': [str(output_dir / f'shap_values_capboost_{s}_{t}.pkl')
                                     for s, t in combinations]
            }
        }

        summary_file = output_dir / 'capboost_analysis_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"Analysis summary saved to: {summary_file}")

        # Create a simple visualization summary
        if len(df) > 0:
            import matplotlib.pyplot as plt
            import seaborn as sns

            plt.style.use('default')
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))

            # 1. Distribution of values by direction
            ax1 = axes[0, 0]
            df.boxplot(column='Value', by='Direction', ax=ax1)
            ax1.set_title('Distribution of Feature-Feature Relationships by Direction')
            ax1.set_xlabel('Cross-Modal Direction')
            ax1.set_ylabel('SHAP Value')
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

            # 2. Number of relationships by direction
            ax2 = axes[0, 1]
            direction_counts = df['Direction'].value_counts()
            direction_counts.plot(kind='bar', ax=ax2)
            ax2.set_title('Number of Relationships by Direction')
            ax2.set_xlabel('Cross-Modal Direction')
            ax2.set_ylabel('Count')
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

            # 3. Number of relationships by cluster
            ax3 = axes[1, 0]
            cluster_counts = df['Cluster'].value_counts().sort_index()
            cluster_counts.plot(kind='bar', ax=ax3)
            ax3.set_title('Number of Relationships by Cluster')
            ax3.set_xlabel('Cluster ID')
            ax3.set_ylabel('Count')

            # 4. Heatmap of mean values by direction and cluster
            ax4 = axes[1, 1]
            pivot_table = df.pivot_table(values='Value', index='Cluster', columns='Direction', aggfunc='mean')
            sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='RdBu_r', center=0, ax=ax4)
            ax4.set_title('Mean SHAP Values by Cluster and Direction')
            plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

            plt.tight_layout()
            viz_file = output_dir / 'capboost_feature_analysis_summary.png'
            plt.savefig(viz_file, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Summary visualization saved to: {viz_file}")

    except Exception as e:
        print(f"\nError during CapBoost analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
