#!/usr/bin/env python3
"""
Cluster-Phenotype Correspondence Analysis for RNA+Activity+Niche Model (PREFX v2)

CHANGELOG (v2):
---------------
- Uses RNA modality instead of Flux
- Ensures discovered_cluster in AnnData.obs is stored as 'Cluster_X' strings.
- Adds discovered_cluster_id (0-based int) in parallel.
- Updates .uns['cluster_phenotype_mapping'] to use 'Cluster_X' keys.
- Ensures categorical ordering consistency across all objects and saved files.
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

def prepare_data_for_shap(adata_dict):
    """Concatenate train/test per modality and ensure dense matrices. Uses RNA+Activity+Niche."""
    rna_all = sc.concat([adata_dict['adata_rna_train'], adata_dict['adata_rna_test']], join='inner')
    activity_all = sc.concat([adata_dict['adata_activity_train'], adata_dict['adata_activity_test']], join='inner')
    niche_all = sc.concat([adata_dict['adata_niche_train'], adata_dict['adata_niche_test']], join='inner')

    for adata in (rna_all, activity_all, niche_all):
        if hasattr(adata.X, 'toarray'):
            adata.X = adata.X.toarray()
        adata.X = np.asarray(adata.X, dtype=np.float32)

    all_phenotypes = []
    for adata in (rna_all, activity_all, niche_all):
        if 'phenotypes' in adata.obs.columns:
            all_phenotypes = _clean_phenotypes_to_strings(adata.obs['phenotypes'])
            break

    if all_phenotypes:
        unique_labels = sorted(set(all_phenotypes))
        label_to_code = {label: i for i, label in enumerate(unique_labels)}
        print("Phenotype mapping:")
        for label, code in label_to_code.items():
            count = sum(p == label for p in all_phenotypes)
            print(f"  {label} -> {code} ({count} cells)")

        for adata in (rna_all, activity_all, niche_all):
            if 'phenotypes' in adata.obs.columns:
                phenotypes_clean = _clean_phenotypes_to_strings(adata.obs['phenotypes'])
                adata.obs['phenotypes_clean'] = phenotypes_clean
                adata.obs['label'] = [label_to_code[p] for p in phenotypes_clean]

    return [rna_all, activity_all, niche_all]


def load_model_and_get_predictions(model_path, data_path, dataset_id):
    """Load model + input data; return predictions and adata dict. Uses RNA+Activity+Niche."""
    print(f"Loading model from: {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    data_path = Path(data_path)
    adata_dict = {}
    for split in ['train', 'test']:
        for modality in ['rna', 'activity', 'niche']:
            key = f'adata_{modality}_{split}'
            fp = data_path / f'adata_{modality}_{split}_perturbmap_{dataset_id}.h5ad'
            if not fp.exists():
                raise FileNotFoundError(f"Missing: {fp}")
            print(f"Loading {key} from {fp}")
            adata_dict[key] = sc.read_h5ad(fp)

    adatas_all = prepare_data_for_shap(adata_dict)

    print("Getting model predictions...")
    try:
        predict_label = model.predict_label(adatas_all)
        print("✓ Obtained predictions from model.predict_label()")
    except Exception as e:
        print(f"Error with predict_label: {e}")
        print("→ Using fallback KMeans clustering.")
        combined_features = np.concatenate([ad.X for ad in adatas_all], axis=1)
        n_clusters = 6  # Default for RNA model
        if hasattr(model, 'technique') and isinstance(model.technique, dict):
            n_clusters = model.technique.get('clusters', {}).get('output', n_clusters)
        print(f"Using KMeans with n_clusters={n_clusters}")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        predict_label = kmeans.fit_predict(combined_features)

    phenos = adatas_all[0].obs.get('phenotypes', pd.Series(['Unknown'] * adatas_all[0].shape[0]))
    original_phenotypes = np.array(_clean_phenotypes_to_strings(phenos))
    print(f"Predicted clusters: {sorted(np.unique(predict_label))}")
    print(f"Original phenotypes: {sorted(np.unique(original_phenotypes))}")

    return predict_label, original_phenotypes, adatas_all, adata_dict


def update_input_files_with_discovered_clusters(predict_label, adata_dict, args):
    """
    Add discovered clusters to AnnData.obs:
      - discovered_cluster_id: 0-based integer
      - discovered_cluster: 'Cluster_X' string
    Uses RNA+Activity+Niche modalities.
    """
    print("\n" + "=" * 60)
    print("UPDATING INPUT FILES WITH DISCOVERED CLUSTERS (RNA PREFX v2)")
    print("=" * 60)

    data_path = Path(args.data_path)
    files_to_update = {
        'adata_rna_train': f'adata_rna_train_perturbmap_{args.dataset_id}.h5ad',
        'adata_activity_train': f'adata_activity_train_perturbmap_{args.dataset_id}.h5ad',
        'adata_niche_train': f'adata_niche_train_perturbmap_{args.dataset_id}.h5ad',
        'adata_rna_test': f'adata_rna_test_perturbmap_{args.dataset_id}.h5ad',
        'adata_activity_test': f'adata_activity_test_perturbmap_{args.dataset_id}.h5ad',
        'adata_niche_test': f'adata_niche_test_perturbmap_{args.dataset_id}.h5ad',
    }

    train_sizes = {m: adata_dict[f'adata_{m}_train'].shape[0] for m in ['rna', 'activity', 'niche']}
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
        ad.obs['cluster_discovery_method'] = 'UnitedNet_RNA_predict_label'
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
        print(f"✅ Updated {fp.name} with Cluster_X labels")

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
    mapping_file = data_path / f'discovered_clusters_mapping_{args.dataset_id}.csv'
    df_map.to_csv(mapping_file, index=False)
    print(f"📊 Mapping CSV saved: {mapping_file}")
    print("✅ AnnData objects now contain prefixed cluster labels ('Cluster_X')")
    return mapping_file


# ------------------------- correspondence analysis ------------------------- #

def create_correspondence_analysis(predict_label, original_phenotypes, output_dir):
    """Create correspondence analysis with metrics, heatmaps, and barplots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred = np.asarray(predict_label, dtype=int)
    orig = np.asarray(original_phenotypes, dtype=object)
    
    # Contingency table
    contingency = pd.crosstab(pred, orig, margins=True, margins_name="Total")
    contingency.to_csv(output_dir / 'contingency_table.csv')
    print(f"\nContingency Table:")
    print(contingency)

    # Calculate metrics
    ari = adjusted_rand_score(orig, pred)
    nmi = normalized_mutual_info_score(orig, pred)
    print(f"\nARI={ari:.4f}, NMI={nmi:.4f}")

    # Confusion matrix (without margins for heatmap)
    cm = contingency.iloc[:-1, :-1].values
    uniq_clusters = sorted(np.unique(pred))
    uniq_phens = sorted(np.unique(orig))
    
    # Normalized confusion matrix
    cm_norm = np.divide(cm, cm.sum(axis=1, keepdims=True), where=cm.sum(axis=1, keepdims=True)!=0)
    
    # Plot normalized heatmap
    plt.figure(figsize=(10,7))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=uniq_phens, 
                yticklabels=[f"Cluster_{c}" for c in uniq_clusters])
    plt.title("Cluster–Phenotype Correspondence (Normalized)")
    plt.xlabel("Original Phenotype")
    plt.ylabel("Predicted Cluster")
    plt.tight_layout()
    plt.savefig(output_dir / 'correspondence_heatmap_normalized.png', dpi=300)
    plt.close()
    
    # Plot raw counts heatmap
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=uniq_phens,
                yticklabels=[f"Cluster_{c}" for c in uniq_clusters],
                cbar_kws={'label': 'Number of cells'})
    plt.title("Cluster–Phenotype Correspondence (Raw Counts)")
    plt.xlabel("Original Phenotype")
    plt.ylabel("Predicted Cluster")
    plt.tight_layout()
    plt.savefig(output_dir / 'correspondence_heatmap_counts.png', dpi=300)
    plt.close()
    
    # Best mapping using Hungarian algorithm
    cost_matrix = -cm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    best_accuracy = cm[row_ind, col_ind].sum() / cm.sum()
    
    best_mapping = {}
    for pred_idx, orig_idx in zip(row_ind, col_ind):
        best_mapping[f'Cluster_{uniq_clusters[pred_idx]}'] = uniq_phens[orig_idx]
    
    print(f"\nBest Mapping Accuracy: {best_accuracy:.4f}")
    print("Best Cluster → Phenotype Mapping:")
    for cluster, phenotype in sorted(best_mapping.items()):
        print(f"  {cluster:12s} → {phenotype}")
    
    # Save results
    results = {
        'ari': float(ari),
        'nmi': float(nmi),
        'best_mapping_accuracy': float(best_accuracy),
        'best_cluster_to_phenotype_mapping': best_mapping,
        'unique_clusters': [int(c) for c in uniq_clusters],
        'unique_phenotypes': list(uniq_phens),
        'confusion_matrix': cm.tolist()
    }
    
    with open(output_dir / 'correspondence_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_dir}")
    
    return results


# ------------------------------- CLI ------------------------------- #

def main():
    p = argparse.ArgumentParser(description="Cluster–Phenotype Correspondence (RNA PREFX v2)")
    p.add_argument('--model_path', required=True, help='Path to trained RNA+Activity+Niche model')
    p.add_argument('--data_path', default='../Data/UnitedNet/input_data')
    p.add_argument('--dataset_id', default='KP2_1')
    p.add_argument('--output_dir', default='../Analysis/Cluster_Phenotype_Correspondence')
    p.add_argument('--update_input_files', action='store_true',
                   help='Update input h5ad files with discovered cluster labels')
    args = p.parse_args()

    outdir = Path(args.output_dir) / f"correspondence_analysis_rna_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print("\n" + "=" * 60)
    print("CLUSTER-PHENOTYPE CORRESPONDENCE ANALYSIS (RNA VERSION)")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.dataset_id}")
    print(f"Modalities: RNA + Activity + Niche")
    print(f"Output: {outdir}")
    print("=" * 60 + "\n")

    predict_label, orig_phens, _, adata_dict = load_model_and_get_predictions(
        args.model_path, args.data_path, args.dataset_id
    )

    if args.update_input_files:
        mapping_file = update_input_files_with_discovered_clusters(predict_label, adata_dict, args)
    
    results = create_correspondence_analysis(predict_label, orig_phens, outdir)

    print(f"\n{'='*60}")
    print(f"✓ ANALYSIS COMPLETED")
    print(f"{'='*60}")
    print(f"ARI={results['ari']:.3f}, NMI={results['nmi']:.3f}, Best Mapping={results['best_mapping_accuracy']:.3f}")
    if args.update_input_files:
        print(f"📁 Updated AnnData files and saved mapping CSV: {mapping_file}")
    print(f"{'='*60}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())