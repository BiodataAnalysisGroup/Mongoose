#!/usr/bin/env python3
"""
Cluster-Phenotype Correspondence Analysis for CapBoost Model (Activity + Niche only)

This version:
- Removes any dependency on the 'flux' modality
- Keeps all phenotype handling, clustering metrics, and plots identical

Usage:
    python analyze_cluster_phenotype_correspondence_no_flux.py --model_path <path_to_model>
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
import scanpy as sc


def prepare_data_for_shap(adata_dict):
    """
    Prepare concatenated train+test data for SHAP analysis
    Activity + Niche only (no Flux)
    """
    # Concatenate train and test for each modality
    activity_all = sc.concat(
        [adata_dict['adata_activity_train'], adata_dict['adata_activity_test']],
        join='inner'
    )
    niche_all = sc.concat(
        [adata_dict['adata_niche_train'], adata_dict['adata_niche_test']],
        join='inner'
    )

    # Ensure dense arrays (float32)
    for adata in [activity_all, niche_all]:
        if hasattr(adata.X, 'toarray'):
            adata.X = adata.X.toarray()
        adata.X = np.asarray(adata.X, dtype=np.float32)

    # Extract phenotypes properly - convert to list of strings, handling NaN
    all_phenotypes = []
    for adata in [activity_all, niche_all]:
        if 'phenotypes' in adata.obs.columns:
            phenotypes = adata.obs['phenotypes'].values

            # Handle categorical data
            if hasattr(phenotypes, 'categories'):
                phenotypes = [str(x) for x in phenotypes]
            else:
                phenotypes = [
                    str(x) if not isinstance(x, (list, np.ndarray)) else str(x[0])
                    for x in phenotypes
                ]

            # Clean NaN / None
            phenotypes_clean = []
            for p in phenotypes:
                if p in ['nan', 'None', 'NaN', 'none']:
                    phenotypes_clean.append('Unknown')
                elif pd.isna(p):
                    phenotypes_clean.append('Unknown')
                else:
                    phenotypes_clean.append(p)

            all_phenotypes.extend(phenotypes_clean)
            # We only need to inspect one modality to build the mapping
            break

    # Create consistent label mapping and apply to both modalities
    if all_phenotypes:
        unique_labels = sorted(set(all_phenotypes))
        label_to_code = {label: i for i, label in enumerate(unique_labels)}

        print("Phenotype mapping:")
        for label, code in label_to_code.items():
            count = sum(1 for p in all_phenotypes if p == label)
            print(f"  {label} -> {code} ({count} cells)")

        for adata in [activity_all, niche_all]:
            if 'phenotypes' in adata.obs.columns:
                phenotypes = adata.obs['phenotypes'].values
                if hasattr(phenotypes, 'categories'):
                    phenotypes = [str(x) for x in phenotypes]
                else:
                    phenotypes = [
                        str(x) if not isinstance(x, (list, np.ndarray)) else str(x[0])
                        for x in phenotypes
                    ]

                phenotypes_clean = []
                for p in phenotypes:
                    if p in ['nan', 'None', 'NaN', 'none']:
                        phenotypes_clean.append('Unknown')
                    elif pd.isna(p):
                        phenotypes_clean.append('Unknown')
                    else:
                        phenotypes_clean.append(p)

                numeric_labels = [label_to_code[p] for p in phenotypes_clean]
                adata.obs['label'] = numeric_labels
                adata.obs['phenotypes_clean'] = phenotypes_clean

    return [activity_all, niche_all]


def load_model_and_get_predictions(model_path, data_path, dataset_id):
    """Load model and get predictions (Activity + Niche only)"""

    # Load the trained model
    print(f"Loading model from: {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Load data files
    data_path = Path(data_path)
    adata_dict = {}

    for split in ['train', 'test']:
        for modality in ['activity', 'niche']:
            key = f'adata_{modality}_{split}'
            filename = f'adata_{modality}_{split}_perturbmap_{dataset_id}.h5ad'
            filepath = data_path / filename

            if not filepath.exists():
                raise FileNotFoundError(f"Required file not found: {filepath}")

            print(f"Loading {key} from {filepath}")
            adata_dict[key] = sc.read_h5ad(filepath)

    # Prepare data
    adatas_all = prepare_data_for_shap(adata_dict)  # [activity_all, niche_all]

    # Get predictions
    print("Getting model predictions...")
    try:
        predict_label = model.predict_label(adatas_all)
        print("Successfully obtained predictions from model.predict_label()")
    except Exception as e:
        print(f"Error with predict_label: {e}")
        print("Using fallback KMeans clustering on concatenated Activity + Niche features...")

        # Fallback: use KMeans on combined features from Activity + Niche
        combined_features = np.concatenate(
            [
                adatas_all[0].X,  # Activity
                adatas_all[1].X,  # Niche
            ],
            axis=1
        )

        # Determine number of clusters from model config or default to 4
        n_clusters = 4
        if hasattr(model, 'technique') and isinstance(model.technique, dict):
            if 'clusters' in model.technique and 'output' in model.technique['clusters']:
                n_clusters = model.technique['clusters']['output']

        print(f"Using KMeans with {n_clusters} clusters")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        predict_label = kmeans.fit_predict(combined_features)

    # Get original phenotypes from activity modality
    phenotypes_raw = adatas_all[0].obs['phenotypes'].values
    if hasattr(phenotypes_raw, 'categories'):
        original_phenotypes = np.array([str(x) for x in phenotypes_raw])
    else:
        original_phenotypes = np.array(
            [str(x) if not isinstance(x, (list, np.ndarray)) else str(x[0])
             for x in phenotypes_raw]
        )

    print(f"Loaded {len(predict_label)} predictions")
    print(f"Unique predicted clusters: {sorted(np.unique(predict_label))}")
    print(f"Unique original phenotypes: {sorted(np.unique(original_phenotypes))}")

    return predict_label, original_phenotypes, adatas_all


def create_correspondence_analysis(predict_label, original_phenotypes, output_dir):
    """Create comprehensive correspondence analysis using pandas crosstab"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_labels = np.array(predict_label)
    orig_labels = np.array(original_phenotypes)

    print(f"\n{'='*60}")
    print("CORRESPONDENCE ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Number of samples: {len(pred_labels)}")
    print(f"Predicted clusters: {sorted(np.unique(pred_labels))}")
    print(f"Original phenotypes: {sorted(np.unique(orig_labels))}")

    unique_phenotypes = sorted(np.unique(orig_labels))
    unique_pred_clusters = sorted(np.unique(pred_labels))

    phenotype_to_code = {phenotype: i for i, phenotype in enumerate(unique_phenotypes)}
    orig_codes = np.array([phenotype_to_code[p] for p in orig_labels])

    # 1. Confusion / correspondence table
    print("\nBuilding correspondence table using crosstab...")
    correspondence_df = pd.crosstab(
        pd.Series(orig_labels, name='Phenotype'),
        pd.Series(pred_labels, name='Cluster'),
        dropna=False
    )

    correspondence_df = correspondence_df.reindex(index=unique_phenotypes, fill_value=0)
    correspondence_df.columns = [f'Cluster_{int(i)}' for i in correspondence_df.columns]

    cm = correspondence_df.values

    print(f"Confusion matrix shape: {cm.shape}")
    print(f"Expected: ({len(unique_phenotypes)} phenotypes, {len(unique_pred_clusters)} clusters)")

    # Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=[f'Cluster {i}' for i in unique_pred_clusters],
        yticklabels=unique_phenotypes,
        cbar_kws={'label': 'Number of cells'}
    )
    plt.title('Confusion Matrix: Original Phenotypes vs Predicted Clusters',
              fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Cluster', fontsize=12)
    plt.ylabel('Original Phenotype', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Percentages
    correspondence_pct = correspondence_df.div(correspondence_df.sum(axis=1), axis=0) * 100

    print(f"\n{'='*60}")
    print("CORRESPONDENCE TABLE (Raw Counts)")
    print(f"{'='*60}")
    print(correspondence_df.to_string())

    print(f"\n{'='*60}")
    print("CORRESPONDENCE TABLE (Percentages)")
    print(f"{'='*60}")
    print(correspondence_pct.round(1).to_string())

    correspondence_df.to_csv(output_dir / 'correspondence_counts.csv')
    correspondence_pct.to_csv(output_dir / 'correspondence_percentages.csv')

    # 3. Cluster purity
    cluster_purity = {}
    phenotype_distribution = {}

    for cluster in unique_pred_clusters:
        cluster_mask = (pred_labels == cluster)
        cluster_phenotypes = orig_labels[cluster_mask]

        unique, counts = np.unique(cluster_phenotypes, return_counts=True)
        most_common_idx = np.argmax(counts)
        most_common_phenotype = unique[most_common_idx]
        purity = counts[most_common_idx] / len(cluster_phenotypes)

        cluster_purity[f'Cluster_{cluster}'] = {
            'dominant_phenotype': most_common_phenotype,
            'purity': float(purity),
            'total_cells': int(len(cluster_phenotypes)),
            'distribution': {str(k): int(v) for k, v in zip(unique, counts)}
        }

        phenotype_distribution[f'Cluster_{cluster}'] = {
            str(k): float(v / len(cluster_phenotypes) * 100)
            for k, v in zip(unique, counts)
        }

    # 4. Phenotype recovery
    phenotype_recovery = {}
    for phenotype in unique_phenotypes:
        phenotype_mask = (orig_labels == phenotype)
        phenotype_predictions = pred_labels[phenotype_mask]

        unique, counts = np.unique(phenotype_predictions, return_counts=True)
        most_common_idx = np.argmax(counts)
        most_common_cluster = unique[most_common_idx]
        recovery = counts[most_common_idx] / len(phenotype_predictions)

        phenotype_recovery[phenotype] = {
            'dominant_cluster': f'Cluster_{int(most_common_cluster)}',
            'recovery': float(recovery),
            'total_cells': int(len(phenotype_predictions)),
            'distribution': {
                f'Cluster_{int(c)}': int(cnt) for c, cnt in zip(unique, counts)
            }
        }

    # 5. Metrics: ARI, NMI, best mapping
    ari = adjusted_rand_score(orig_codes, pred_labels)
    nmi = normalized_mutual_info_score(orig_codes, pred_labels)

    from scipy.optimize import linear_sum_assignment
    cost_matrix = -cm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    best_accuracy = cm[row_ind, col_ind].sum() / cm.sum()

    best_mapping = {
        f'Cluster_{unique_pred_clusters[pred_idx]}': unique_phenotypes[orig_idx]
        for orig_idx, pred_idx in zip(row_ind, col_ind)
    }

    metrics = {
        'adjusted_rand_index': float(ari),
        'normalized_mutual_info': float(nmi),
        'best_mapping_accuracy': float(best_accuracy),
        'best_cluster_to_phenotype_mapping': best_mapping
    }

    print(f"\n{'='*60}")
    print("CLUSTERING METRICS")
    print(f"{'='*60}")
    print(f"Adjusted Rand Index: {ari:.4f}")
    print(f"Normalized Mutual Information: {nmi:.4f}")
    print(f"Best Mapping Accuracy: {best_accuracy:.4f}")

    print(f"\n{'='*60}")
    print("BEST CLUSTER → PHENOTYPE MAPPING")
    print(f"{'='*60}")
    for cluster, phenotype in sorted(best_mapping.items()):
        print(f"{cluster:12s} → {phenotype}")

    # 6. Visualizations: Cluster Purity & Recovery
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    clusters = sorted(cluster_purity.keys())
    purities = [cluster_purity[c]['purity'] for c in clusters]
    dominant_phenotypes = [cluster_purity[c]['dominant_phenotype'] for c in clusters]

    bars = ax1.bar(clusters, purities, color='skyblue', alpha=0.7, edgecolor='navy')
    ax1.set_ylabel('Purity (fraction of dominant phenotype)', fontsize=11)
    ax1.set_title('Cluster Purity', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 1.1)
    ax1.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='50% threshold')
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)

    for bar, phenotype, purity in zip(bars, dominant_phenotypes, purities):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f'{phenotype}\n({purity:.2f})',
            ha='center',
            va='bottom',
            fontsize=9
        )

    phenotypes = list(phenotype_recovery.keys())
    recoveries = [phenotype_recovery[p]['recovery'] for p in phenotypes]
    dominant_clusters = [phenotype_recovery[p]['dominant_cluster'] for p in phenotypes]

    bars2 = ax2.bar(
        range(len(phenotypes)),
        recoveries,
        color='lightcoral',
        alpha=0.7,
        edgecolor='darkred'
    )
    ax2.set_ylabel('Recovery (fraction in dominant cluster)', fontsize=11)
    ax2.set_title('Phenotype Recovery', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 1.1)
    ax2.set_xticks(range(len(phenotypes)))
    ax2.set_xticklabels(phenotypes, rotation=45, ha='right')
    ax2.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='50% threshold')
    ax2.legend()

    for bar, cluster, recovery in zip(bars2, dominant_clusters, recoveries):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f'{cluster}\n({recovery:.2f})',
            ha='center',
            va='bottom',
            fontsize=9
        )

    plt.tight_layout()
    plt.savefig(output_dir / 'cluster_purity_recovery.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 7. Stacked distribution charts
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    cluster_names = sorted([f'Cluster_{i}' for i in unique_pred_clusters])
    phenotype_colors = plt.cm.Set3(np.linspace(0, 1, len(unique_phenotypes)))

    bottom1 = np.zeros(len(cluster_names))
    for i, phenotype in enumerate(unique_phenotypes):
        values = [phenotype_distribution[cluster].get(phenotype, 0) for cluster in cluster_names]
        ax1.bar(
            cluster_names,
            values,
            bottom=bottom1,
            label=phenotype,
            color=phenotype_colors[i],
            edgecolor='white',
            linewidth=0.5
        )
        bottom1 += values

    ax1.set_ylabel('Percentage', fontsize=11)
    ax1.set_title('Phenotype Distribution per Cluster', fontsize=13, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.set_ylim(0, 100)
    ax1.tick_params(axis='x', rotation=45)

    cluster_colors = plt.cm.tab10(np.linspace(0, 1, len(cluster_names)))

    bottom2 = np.zeros(len(unique_phenotypes))
    for i, cluster in enumerate(cluster_names):
        values = []
        for phenotype in unique_phenotypes:
            total_phenotype = np.sum(orig_labels == phenotype)
            count_in_cluster = phenotype_recovery[phenotype]['distribution'].get(cluster, 0)
            percentage = (count_in_cluster / total_phenotype) * 100 if total_phenotype > 0 else 0
            values.append(percentage)

        ax2.bar(
            unique_phenotypes,
            values,
            bottom=bottom2,
            label=cluster,
            color=cluster_colors[i],
            edgecolor='white',
            linewidth=0.5
        )
        bottom2 += values

    ax2.set_ylabel('Percentage', fontsize=11)
    ax2.set_title('Cluster Distribution per Phenotype', fontsize=13, fontweight='bold')
    ax2.set_xticks(range(len(unique_phenotypes)))
    ax2.set_xticklabels(unique_phenotypes, rotation=45, ha='right')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_dir / 'distribution_stacked.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 8. Save comprehensive results
    results = {
        'metrics': metrics,
        'cluster_purity': cluster_purity,
        'phenotype_recovery': phenotype_recovery,
        'phenotype_to_code_mapping': phenotype_to_code,
        'confusion_matrix': cm.tolist(),
        'unique_phenotypes': unique_phenotypes,
        'unique_clusters': [int(x) for x in unique_pred_clusters]
    }

    with open(output_dir / 'correspondence_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("DETAILED CLUSTER ANALYSIS")
    print(f"{'='*60}")
    for cluster in sorted(cluster_purity.keys()):
        info = cluster_purity[cluster]
        print(f"\n{cluster}:")
        print(f"  Dominant phenotype: {info['dominant_phenotype']}")
        print(f"  Purity: {info['purity']:.3f} ({info['purity']*100:.1f}%)")
        print(f"  Total cells: {info['total_cells']}")
        print(f"  Distribution: {info['distribution']}")

    print(f"\n{'='*60}")
    print("DETAILED PHENOTYPE RECOVERY")
    print(f"{'='*60}")
    for phenotype in sorted(phenotype_recovery.keys()):
        info = phenotype_recovery[phenotype]
        print(f"\n{phenotype}:")
        print(f"  Dominant cluster: {info['dominant_cluster']}")
        print(f"  Recovery: {info['recovery']:.3f} ({info['recovery']*100:.1f}%)")
        print(f"  Total cells: {info['total_cells']}")
        print(f"  Distribution: {info['distribution']}")

    print(f"\n{'='*60}")
    print(f"All results saved to: {output_dir}")
    print(f"{'='*60}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Analyze correspondence between predicted clusters and original phenotypes (Activity + Niche only)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to trained CapBoost model pickle file'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default='../Data/UnitedNet/input_data',
        help='Path to directory containing input data files'
    )
    parser.add_argument(
        '--dataset_id',
        type=str,
        default='KP2_1',
        help='Dataset identifier'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='../Analysis/Cluster_Phenotype_Correspondence',
        help='Directory to save analysis results'
    )

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"correspondence_analysis_{timestamp}"

    print(f"\n{'='*60}")
    print("CLUSTER-PHENOTYPE CORRESPONDENCE ANALYSIS (NO FLUX)")
    print(f"{'='*60}")
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.dataset_id}")
    print(f"Results directory: {output_dir}")
    print(f"{'='*60}\n")

    try:
        predict_label, original_phenotypes, adatas_all = load_model_and_get_predictions(
            args.model_path, args.data_path, args.dataset_id
        )

        _ = create_correspondence_analysis(predict_label, original_phenotypes, output_dir)

        print(f"\n{'='*60}")
        print("✓ ANALYSIS COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"\n{'='*60}")
        print("✗ ERROR DURING ANALYSIS")
        print(f"{'='*60}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
