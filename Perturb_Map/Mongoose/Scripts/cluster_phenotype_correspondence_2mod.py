#!/usr/bin/env python3
"""
Cluster-Phenotype Correspondence Analysis for 2-Modality UnitedNet Model
(Activity + Niche only)

CHANGELOG from original:
-----------------------
- Adapted for 2-modality models (activity + niche only, no flux)
- Removes all flux-related processing
- Maintains same cluster mapping and analysis logic
- Ensures discovered_cluster in AnnData.obs is stored as 'Cluster_X' strings
- Adds discovered_cluster_id (0-based int) in parallel

USAGE EXAMPLES:
--------------
# Command line usage:
python cluster_phenotype_correspondence_2mod.py --model_path /path/to/model.pkl --update_input_files

# Programmatic usage (for notebooks):
import sys
sys.argv = [
    "cluster_phenotype_correspondence_2mod.py",
    "--model_path", "../Models/UnitedNet/activity_niche_2mod_20251115_092842/model_perturbmap_activity_niche_KP2_1_activity_niche_20251115_092939.pkl",
    "--data_path", "../Data/UnitedNet/input_data", 
    "--dataset_id", "KP2_1",
    "--output_dir", "../Analysis/Activity_Niche_Cluster_Analysis",
    "--update_input_files"
]
from cluster_phenotype_correspondence_2mod import main
main()
"""

import argparse
import sys
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

def prepare_data_for_shap_2mod(adata_dict):
    """Concatenate train/test per modality and ensure dense matrices (2-modality version)."""
    print("Preparing 2-modality data (activity + niche)...")
    
    activity_all = sc.concat([adata_dict['adata_activity_train'], adata_dict['adata_activity_test']], join='inner')
    niche_all = sc.concat([adata_dict['adata_niche_train'], adata_dict['adata_niche_test']], join='inner')

    for adata in (activity_all, niche_all):
        if hasattr(adata.X, 'toarray'):
            adata.X = adata.X.toarray()
        adata.X = np.asarray(adata.X, dtype=np.float32)

    print(f"Activity data shape: {activity_all.shape}")
    print(f"Niche data shape: {niche_all.shape}")

    all_phenotypes = []
    for adata in (activity_all, niche_all):
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

        for adata in (activity_all, niche_all):
            if 'phenotypes' in adata.obs.columns:
                phenotypes_clean = _clean_phenotypes_to_strings(adata.obs['phenotypes'])
                adata.obs['phenotypes_clean'] = phenotypes_clean
                adata.obs['label'] = [label_to_code[p] for p in phenotypes_clean]

    return [activity_all, niche_all]


def load_model_and_get_predictions_2mod(model_path, data_path, dataset_id):
    """Load model + input data; return predictions and adata dict (2-modality version)."""
    print(f"Loading 2-modality model from: {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    data_path = Path(data_path)
    adata_dict = {}
    
    # Only load activity and niche modalities
    for split in ['train', 'test']:
        for modality in ['activity', 'niche']:
            key = f'adata_{modality}_{split}'
            fp = data_path / f'adata_{modality}_{split}_perturbmap_{dataset_id}.h5ad'
            if not fp.exists():
                raise FileNotFoundError(f"Missing: {fp}")
            print(f"Loading {key} from {fp}")
            adata_dict[key] = sc.read_h5ad(fp)

    adatas_all = prepare_data_for_shap_2mod(adata_dict)

    print("Getting model predictions...")
    try:
        predict_label = model.predict_label(adatas_all)
        print("✓ Obtained predictions from model.predict_label()")
    except Exception as e:
        print(f"Error with predict_label: {e}")
        print("→ Using fallback KMeans clustering.")
        combined_features = np.concatenate([ad.X for ad in adatas_all], axis=1)
        n_clusters = 4
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


def update_input_files_with_discovered_clusters_2mod(predict_label, adata_dict, args):
    """
    Add discovered clusters to AnnData.obs for 2-modality model:
      - discovered_cluster_id: 0-based integer
      - discovered_cluster: 'Cluster_X' string
    """
    print("\n" + "=" * 60)
    print("UPDATING INPUT FILES WITH DISCOVERED CLUSTERS (2-MODALITY)")
    print("=" * 60)

    data_path = Path(args.data_path)
    
    # Only update activity and niche files
    files_to_update = {
        'adata_activity_train': f'adata_activity_train_perturbmap_{args.dataset_id}.h5ad',
        'adata_niche_train': f'adata_niche_train_perturbmap_{args.dataset_id}.h5ad',
        'adata_activity_test': f'adata_activity_test_perturbmap_{args.dataset_id}.h5ad',
        'adata_niche_test': f'adata_niche_test_perturbmap_{args.dataset_id}.h5ad',
    }

    train_sizes = {m: adata_dict[f'adata_{m}_train'].shape[0] for m in ['activity', 'niche']}
    mapping_rows = []

    unique_clusters_sorted = sorted(np.unique(predict_label).astype(int))
    cluster_names = [f"Cluster_{c}" for c in unique_clusters_sorted]
    
    print(f"Discovered {len(unique_clusters_sorted)} clusters: {cluster_names}")

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
        ad.obs['cluster_discovery_method'] = 'UnitedNet_2mod_predict_label'
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
    mapping_file = data_path / f'discovered_clusters_mapping_2mod_{args.dataset_id}.csv'
    df_map.to_csv(mapping_file, index=False)
    print(f"📊 Mapping CSV saved: {mapping_file}")
    print("✅ AnnData objects now contain prefixed cluster labels ('Cluster_X')")
    return mapping_file


# ------------------------- correspondence analysis ------------------------- #

def create_correspondence_analysis_2mod(predict_label, original_phenotypes, output_dir):
    """Create correspondence analysis with enhanced visualizations for 2-modality model."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pred = np.asarray(predict_label, dtype=int)
    orig = np.asarray(original_phenotypes, dtype=object)
    
    # Create contingency table
    contingency = pd.crosstab(pred, orig, margins=True, margins_name="Total")
    contingency.to_csv(output_dir / 'contingency_table_2mod.csv')
    
    print("\nContingency Table (Clusters vs Phenotypes):")
    print(contingency)

    # Calculate metrics
    ari = adjusted_rand_score(orig, pred)
    nmi = normalized_mutual_info_score(orig, pred)
    print(f"\n📊 CLUSTERING METRICS:")
    print(f"   ARI (Adjusted Rand Index): {ari:.4f}")
    print(f"   NMI (Normalized Mutual Info): {nmi:.4f}")

    # Create heatmap
    cm = contingency.iloc[:-1, :-1].values
    uniq_clusters = sorted(np.unique(pred))
    uniq_phens = sorted(np.unique(orig))
    
    # Normalize by row (cluster)
    cm_norm = np.divide(cm, cm.sum(axis=1, keepdims=True), where=cm.sum(axis=1, keepdims=True)!=0)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues', 
                xticklabels=uniq_phens, 
                yticklabels=[f"Cluster_{c}" for c in uniq_clusters])
    plt.title(f"Cluster–Phenotype Correspondence (2-Modality Model)\nARI: {ari:.3f}, NMI: {nmi:.3f}")
    plt.xlabel("Ground Truth Phenotypes")
    plt.ylabel("Discovered Clusters")
    plt.tight_layout()
    plt.savefig(output_dir / 'correspondence_heatmap_2mod.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Create detailed cluster composition bar plot
    plt.figure(figsize=(14, 6))
    cluster_composition = []
    cluster_labels = []
    
    for cluster in uniq_clusters:
        cluster_mask = pred == cluster
        pheno_counts = pd.Series(orig[cluster_mask]).value_counts()
        total_in_cluster = len(orig[cluster_mask])
        
        for pheno in uniq_phens:
            count = pheno_counts.get(pheno, 0)
            percentage = (count / total_in_cluster) * 100
            cluster_composition.append(percentage)
            cluster_labels.append(f"Cluster_{cluster}")
    
    # Reshape for plotting
    n_clusters = len(uniq_clusters)
    n_phenos = len(uniq_phens)
    comp_matrix = np.array(cluster_composition).reshape(n_clusters, n_phenos)
    
    # Create stacked bar plot
    x = np.arange(n_clusters)
    bottom = np.zeros(n_clusters)
    colors = plt.cm.Set3(np.linspace(0, 1, n_phenos))
    
    for i, pheno in enumerate(uniq_phens):
        plt.bar(x, comp_matrix[:, i], bottom=bottom, label=pheno, color=colors[i])
        bottom += comp_matrix[:, i]
    
    plt.xlabel("Discovered Clusters")
    plt.ylabel("Percentage Composition")
    plt.title("Cluster Composition by Ground Truth Phenotypes")
    plt.xticks(x, [f"Cluster_{c}" for c in uniq_clusters])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_dir / 'cluster_composition_2mod.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Save detailed statistics
    stats = {
        'ari': ari,
        'nmi': nmi,
        'n_discovered_clusters': len(uniq_clusters),
        'n_ground_truth_phenotypes': len(uniq_phens),
        'discovered_clusters': [f"Cluster_{c}" for c in uniq_clusters],
        'ground_truth_phenotypes': list(uniq_phens),
        'cluster_sizes': {f"Cluster_{c}": int(np.sum(pred == c)) for c in uniq_clusters},
        'phenotype_sizes': {pheno: int(np.sum(orig == pheno)) for pheno in uniq_phens}
    }
    
    with open(output_dir / 'correspondence_stats_2mod.json', 'w') as f:
        json.dump(stats, f, indent=2)

    return stats


# ------------------------------- CLI ------------------------------- #

def main():
    p = argparse.ArgumentParser(description="Cluster–Phenotype Correspondence (2-Modality: Activity + Niche)")
    p.add_argument('--model_path', required=True, help='Path to the trained 2-modality model pickle file')
    p.add_argument('--data_path', default='../Data/UnitedNet/input_data', help='Path to input data directory')
    p.add_argument('--dataset_id', default='KP2_1', help='Dataset identifier')
    p.add_argument('--output_dir', default='../Analysis/Cluster_Phenotype_Correspondence', help='Output directory')
    p.add_argument('--update_input_files', action='store_true', help='Update input AnnData files with discovered clusters')
    args = p.parse_args()

    print("=" * 60)
    print("CLUSTER-PHENOTYPE CORRESPONDENCE ANALYSIS")
    print("2-Modality Model (Activity + Niche)")
    print("=" * 60)

    outdir = Path(args.output_dir) / f"correspondence_2mod_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    try:
        predict_label, orig_phens, _, adata_dict = load_model_and_get_predictions_2mod(
            args.model_path, args.data_path, args.dataset_id
        )

        if args.update_input_files:
            mapping_file = update_input_files_with_discovered_clusters_2mod(predict_label, adata_dict, args)
            
        results = create_correspondence_analysis_2mod(predict_label, orig_phens, outdir)

        print(f"\n🎯 FINAL RESULTS:")
        print(f"   📊 ARI: {results['ari']:.3f}")
        print(f"   📊 NMI: {results['nmi']:.3f}")
        print(f"   🔍 Discovered {results['n_discovered_clusters']} clusters")
        print(f"   🏷️  Ground truth has {results['n_ground_truth_phenotypes']} phenotypes")
        print(f"   📁 Results saved to: {outdir}")
        
        if args.update_input_files:
            print(f"   📝 Updated AnnData files and saved mapping: {mapping_file}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())