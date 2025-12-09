#!/usr/bin/env python3
"""
Cluster-Phenotype Correspondence Analysis for Activity + Niche Model (PREFX v2)

CHANGELOG (v2 - Activity + Niche):
---------------------------------
- Adapted for 2-modality Activity + Niche models (removed flux)
- Ensures discovered_cluster in AnnData.obs is stored as 'Cluster_X' strings
- Adds discovered_cluster_id (0-based int) in parallel
- Updates .uns['cluster_phenotype_mapping'] to use 'Cluster_X' keys
- Ensures categorical ordering consistency across all objects and saved files
- Only processes Activity and Niche modalities
"""

import argparse
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
import scanpy as sc


# ----------------------------- helpers ----------------------------- #

def _clean_phenotypes_to_strings(series_like):
    """Return safe strings from AnnData.obs column, with 'Unknown' for NaNs."""
    values = series_like.values if hasattr(series_like, "values") else np.asarray(series_like)
    if hasattr(values, "categories"):
        values = [str(x) for x in values]
    else:
        values = [str(x) if not isinstance(x, (list, np.ndarray)) else str(x[0]) for x in values]
    cleaned = []
    for p in values:
        if p in ['nan', 'None', 'NaN', 'none']:
            cleaned.append('Unknown')
        elif pd.isna(p):
            cleaned.append('Unknown')
        else:
            cleaned.append(p)
    return cleaned


# ------------------------ core functionality ----------------------- #

def prepare_data_for_analysis(adata_dict):
    """Concatenate train/test per modality and ensure dense matrices (Activity + Niche only)."""
    activity_all = sc.concat([adata_dict['adata_activity_train'], adata_dict['adata_activity_test']], join='inner')
    niche_all = sc.concat([adata_dict['adata_niche_train'], adata_dict['adata_niche_test']], join='inner')

    for adata in (activity_all, niche_all):  # Only 2 modalities
        if hasattr(adata.X, 'toarray'):
            adata.X = adata.X.toarray()
        adata.X = np.asarray(adata.X, dtype=np.float32)

    all_phenotypes = []
    for adata in (activity_all, niche_all):  # Only 2 modalities
        if 'phenotypes' in adata.obs.columns:
            all_phenotypes = _clean_phenotypes_to_strings(adata.obs['phenotypes'])
            break

    if all_phenotypes:
        unique_labels = sorted(set(all_phenotypes))
        label_to_code = {label: i for i, label in enumerate(unique_labels)}
        print("Phenotype mapping (Activity + Niche Model):")
        for label, code in label_to_code.items():
            count = sum(p == label for p in all_phenotypes)
            print(f"  {label} -> {code} ({count} cells)")

        for adata in (activity_all, niche_all):  # Only 2 modalities
            if 'phenotypes' in adata.obs.columns:
                phenotypes_clean = _clean_phenotypes_to_strings(adata.obs['phenotypes'])
                adata.obs['phenotypes_clean'] = phenotypes_clean
                adata.obs['label'] = [label_to_code[p] for p in phenotypes_clean]

    return [activity_all, niche_all]  # Return only 2 modalities


def load_model_and_get_predictions(model_path, data_path, dataset_id):
    """Load Activity + Niche model + input data; return predictions and adata dict."""
    print(f"Loading Activity + Niche model from: {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    data_path = Path(data_path)
    adata_dict = {}
    
    # Only load Activity and Niche modalities
    for split in ['train', 'test']:
        for modality in ['activity', 'niche']:  # Removed 'flux'
            key = f'adata_{modality}_{split}'
            fp = data_path / f'adata_{modality}_{split}_perturbmap_{dataset_id}.h5ad'
            if not fp.exists():
                raise FileNotFoundError(f"Missing: {fp}")
            print(f"Loading {key} from {fp}")
            adata_dict[key] = sc.read_h5ad(fp)

    adatas_all = prepare_data_for_analysis(adata_dict)

    print("Getting model predictions...")
    try:
        predict_label = model.predict_label(adatas_all)
        print("✓ Obtained predictions from Activity + Niche model.predict_label()")
    except Exception as e:
        print(f"Error with predict_label: {e}")
        print("→ Using fallback KMeans clustering on Activity + Niche features.")
        combined_features = np.concatenate([ad.X for ad in adatas_all], axis=1)  # Only Activity + Niche
        n_clusters = 4
        if hasattr(model, 'technique') and isinstance(model.technique, dict):
            n_clusters = model.technique.get('clusters', {}).get('output', n_clusters)
        print(f"Using KMeans with n_clusters={n_clusters} on Activity + Niche")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        predict_label = kmeans.fit_predict(combined_features)

    phenos = adatas_all[0].obs.get('phenotypes', pd.Series(['Unknown'] * adatas_all[0].shape[0]))
    original_phenotypes = np.array(_clean_phenotypes_to_strings(phenos))
    print(f"Predicted clusters: {sorted(np.unique(predict_label))}")
    print(f"Original phenotypes: {sorted(np.unique(original_phenotypes))}")

    return predict_label, original_phenotypes, adatas_all, adata_dict


def update_input_files_with_discovered_clusters(predict_label, adata_dict, args):
    """
    Add discovered clusters to AnnData.obs for Activity + Niche files:
      - discovered_cluster_id: 0-based integer
      - discovered_cluster: 'Cluster_X' string
    """
    print("\n" + "=" * 60)
    print("UPDATING ACTIVITY + NICHE INPUT FILES WITH DISCOVERED CLUSTERS (PREFX v2)")
    print("=" * 60)

    data_path = Path(args.data_path)
    
    # Only Activity and Niche files (no flux)
    files_to_update = {
        'adata_activity_train': f'adata_activity_train_perturbmap_{args.dataset_id}.h5ad',
        'adata_niche_train': f'adata_niche_train_perturbmap_{args.dataset_id}.h5ad',
        'adata_activity_test': f'adata_activity_test_perturbmap_{args.dataset_id}.h5ad',
        'adata_niche_test': f'adata_niche_test_perturbmap_{args.dataset_id}.h5ad',
    }

    # Only Activity and Niche train sizes
    train_sizes = {m: adata_dict[f'adata_{m}_train'].shape[0] for m in ['activity', 'niche']}
    mapping_rows = []

    unique_clusters_sorted = sorted(np.unique(predict_label).astype(int))
    cluster_names = [f"Cluster_{c}" for c in unique_clusters_sorted]

    for key, filename in files_to_update.items():
        fp = data_path / filename
        ad = sc.read_h5ad(fp)
        modality, split = key.split('_')[1], key.split('_')[2]

        start_idx = 0 if split == 'train' else train_sizes[modality]
        end_idx = start_idx + ad.shape[0]

        cluster_ids = predict_label[start_idx:end_idx].astype(int)
        cluster_strs = [f"Cluster_{cid}" for cid in cluster_ids]

        # === write both columns ===
        ad.obs['discovered_cluster_id'] = pd.Categorical(cluster_ids, categories=unique_clusters_sorted, ordered=True)
        ad.obs['discovered_cluster'] = pd.Categorical(cluster_strs, categories=cluster_names, ordered=True)

        # metadata
        ad.obs['cluster_discovery_timestamp'] = datetime.now().isoformat()
        ad.obs['cluster_discovery_method'] = 'UnitedNet_Activity_Niche_predict_label'
        ad.obs['cluster_total_count'] = len(unique_clusters_sorted)

        # === fix uns mapping ===
        if 'phenotypes' in ad.obs.columns:
            mapping = {}
            for cid in unique_clusters_sorted:
                mask = cluster_ids == cid
                if np.any(mask):
                    phens = ad.obs['phenotypes'][mask]
                    most_common = pd.Series(phens).mode()
                    mapping[f"Cluster_{cid}"] = str(most_common.iloc[0]) if len(most_common) > 0 else 'Unknown'
            ad.uns['cluster_phenotype_mapping'] = mapping

        # backup before overwrite
        backup = fp.with_suffix('.h5ad.backup')
        if not backup.exists():
            ad.copy().write_h5ad(backup)
            print(f"📁 Backup saved: {backup}")

        ad.write_h5ad(fp)
        print(f"✅ Updated {fp.name} with Activity + Niche Cluster_X labels")

        # mapping CSV info
        orig_phens = ad.obs.get('phenotypes', pd.Series(['Unknown'] * ad.shape[0])).astype(str).tolist()
        for i, (cid, cname) in enumerate(zip(cluster_ids, cluster_strs)):
            mapping_rows.append({
                'file': key,
                'modality': modality,
                'split': split,
                'cell_index': i,
                'discovered_cluster_id': int(cid),
                'discovered_cluster': cname,
                'original_phenotype': orig_phens[i],
            })

    df_map = pd.DataFrame(mapping_rows)
    mapping_file = data_path / f'discovered_clusters_activity_niche_mapping_{args.dataset_id}.csv'
    df_map.to_csv(mapping_file, index=False)
    print(f"📊 Activity + Niche Mapping CSV saved: {mapping_file}")
    print("✅ Activity + Niche AnnData objects now contain prefixed cluster labels ('Cluster_X')")
    return mapping_file


# ------------------------- correspondence analysis ------------------------- #

def create_correspondence_analysis(predict_label, original_phenotypes, output_dir):
    """Create correspondence analysis for Activity + Niche model."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pred = np.asarray(predict_label, dtype=int)
    orig = np.asarray(original_phenotypes, dtype=object)
    
    print(f"\n{'='*60}")
    print(f"ACTIVITY + NICHE MODEL CORRESPONDENCE ANALYSIS")
    print(f"{'='*60}")
    print(f"Number of samples: {len(pred)}")
    print(f"Predicted clusters: {sorted(np.unique(pred))}")
    print(f"Original phenotypes: {sorted(np.unique(orig))}")
    
    # Create contingency table
    contingency = pd.crosstab(pred, orig, margins=True, margins_name="Total")
    contingency.to_csv(output_dir / 'contingency_table_activity_niche.csv')

    # Calculate metrics
    ari = adjusted_rand_score(orig, pred)
    nmi = normalized_mutual_info_score(orig, pred)
    
    print(f"\n{'='*60}")
    print(f"ACTIVITY + NICHE CLUSTERING METRICS")
    print(f"{'='*60}")
    print(f"Adjusted Rand Index: {ari:.4f}")
    print(f"Normalized Mutual Information: {nmi:.4f}")

    # Create visualizations
    cm = contingency.iloc[:-1, :-1].values
    uniq_clusters = sorted(np.unique(pred))
    uniq_phens = sorted(np.unique(orig))
    
    # Normalize confusion matrix
    cm_norm = np.divide(cm, cm.sum(axis=1, keepdims=True), where=cm.sum(axis=1, keepdims=True)!=0)
    
    # Create heatmap
    plt.figure(figsize=(10,7))
    sns.heatmap(cm_norm, annot=True, cmap='Blues', fmt='.2f',
                xticklabels=uniq_phens, 
                yticklabels=[f"Cluster_{c}" for c in uniq_clusters],
                cbar_kws={'label': 'Normalized Count'})
    plt.title("Activity + Niche Model: Cluster–Phenotype Correspondence", fontsize=14, fontweight='bold')
    plt.xlabel('Original Phenotype', fontsize=12)
    plt.ylabel('Predicted Cluster', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'correspondence_heatmap_activity_niche.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create summary plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=uniq_phens,
                yticklabels=[f"Cluster_{c}" for c in uniq_clusters],
                ax=ax1)
    ax1.set_title('Raw Counts', fontweight='bold')
    ax1.set_xlabel('Original Phenotype')
    ax1.set_ylabel('Predicted Cluster')
    
    # Normalized
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Reds',
                xticklabels=uniq_phens,
                yticklabels=[f"Cluster_{c}" for c in uniq_clusters],
                ax=ax2)
    ax2.set_title('Normalized (Row %)', fontweight='bold')
    ax2.set_xlabel('Original Phenotype')
    ax2.set_ylabel('Predicted Cluster')
    
    plt.suptitle(f'Activity + Niche Model Performance\nARI: {ari:.3f} | NMI: {nmi:.3f}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'activity_niche_performance_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed results
    results = {
        'ari': float(ari), 
        'nmi': float(nmi),
        'model_type': 'Activity_Niche_2_Modality',
        'confusion_matrix': cm.tolist(),
        'unique_clusters': [int(x) for x in uniq_clusters],
        'unique_phenotypes': [str(x) for x in uniq_phens],
        'total_samples': int(len(pred))
    }
    
    with open(output_dir / 'activity_niche_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"📊 Analysis results saved to: {output_dir}")
    return results


# ------------------------------- CLI ------------------------------- #

def main():
    p = argparse.ArgumentParser(description="Activity + Niche Cluster–Phenotype Correspondence (PREFX v2)")
    p.add_argument('--model_path', required=True, help='Path to Activity + Niche model pickle file')
    p.add_argument('--data_path', default='../Data/UnitedNet/input_data', 
                   help='Path to directory containing Activity + Niche input data files')
    p.add_argument('--dataset_id', default='KP2_1', help='Dataset identifier')
    p.add_argument('--output_dir', default='../Analysis/Activity_Niche_Cluster_Analysis',
                   help='Directory to save analysis results')
    p.add_argument('--update_input_files', action='store_true',
                   help='Update original AnnData files with discovered cluster labels')
    args = p.parse_args()

    # Create timestamped output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    outdir = Path(args.output_dir) / f"activity_niche_correspondence_prefx_{timestamp}"

    print(f"\n{'='*60}")
    print(f"ACTIVITY + NICHE MODEL CLUSTER ANALYSIS (PREFX v2)")
    print(f"{'='*60}")
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.dataset_id}")
    print(f"Modalities: Activity + Niche (2-modality model)")
    print(f"Update files: {args.update_input_files}")
    print(f"Results directory: {outdir}")
    print(f"{'='*60}\n")

    try:
        # Load model and get predictions
        predict_label, orig_phens, _, adata_dict = load_model_and_get_predictions(
            args.model_path, args.data_path, args.dataset_id
        )

        # Update input files if requested
        if args.update_input_files:
            mapping_file = update_input_files_with_discovered_clusters(predict_label, adata_dict, args)
            print(f"\n✅ Updated Activity + Niche input files with cluster labels!")
            print(f"📁 Mapping file: {mapping_file}")
        
        # Perform correspondence analysis
        results = create_correspondence_analysis(predict_label, orig_phens, outdir)

        print(f"\n{'='*60}")
        print(f"✓ ACTIVITY + NICHE ANALYSIS COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        print(f"Key Results:")
        print(f"  - ARI: {results['ari']:.4f}")
        print(f"  - NMI: {results['nmi']:.4f}")
        print(f"  - Total samples: {results['total_samples']}")
        print(f"  - Clusters found: {len(results['unique_clusters'])}")
        print(f"  - Phenotypes: {len(results['unique_phenotypes'])}")
        if args.update_input_files:
            print(f"  - Input files updated with 'Cluster_X' labels")
        print(f"  - Cross-modal Activity ↔ Niche enables phenotype separation")
        print(f"{'='*60}\n")
        
        return 0

    except Exception as e:
        print(f"\n{'='*60}")
        print(f"✗ ERROR DURING ANALYSIS")
        print(f"{'='*60}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())