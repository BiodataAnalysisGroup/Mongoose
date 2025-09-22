#!/usr/bin/env python3
"""
PerturbMap: Niche vs RNA Modality Comparison
Compares niche expression modality with original RNA expression
Adapted from DBiT-seq analysis approach
"""

import pandas as pd
import anndata as ad
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import scanpy as sc
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

def load_original_rna_data():
    """
    Load original RNA data with annotations (same as niche calculation script).
    """
    print("Loading original GSM5808054_10x_Visium_processed.h5ad...")
    adata = sc.read_h5ad('../Data/perturbmap_stomicsdb/GSM5808054_10x_Visium_processed.h5ad')
    print(f"Loaded original data: {adata.shape}")
    
    # Load phenotype annotations
    print("Loading spot annotations...")
    pheno = pd.read_csv("../Data/PertMap_metadata/spot_annotation_KP1.csv")
    
    # Ensure barcodes are the index for merging
    pheno = pheno.set_index("barcode")
    
    # Add phenotypes to adata.obs (matching on index)
    adata.obs["phenotypes"] = adata.obs.index.map(pheno["phenotypes"])
    adata.obs['kmeans'] = adata.obs.index.map(pheno["kmeans"])
    
    # Remove spots with missing annotations
    before_filter = adata.shape[0]
    adata = adata[~adata.obs["phenotypes"].isna()].copy()
    after_filter = adata.shape[0]
    print(f"Filtered spots: {before_filter} -> {after_filter} ({before_filter - after_filter} removed)")
    
    return adata

def load_data_for_comparison():
    """
    Load RNA and niche modality data for comparison.
    """
    print("=== PerturbMap RNA vs Niche Modality Analysis ===")
    
    # Load original RNA data
    print("Loading original RNA data...")
    adata_rna = load_original_rna_data()
    
    # Load niche data  
    print("Loading niche data...")
    adata_niche = sc.read_h5ad('../Data/processed_data/GSM5808054_niche_full.h5ad')
    print(f"Niche data: {adata_niche.shape}")
    
    # Check gene overlap
    rna_genes = set(adata_rna.var_names)
    niche_genes = set(adata_niche.var_names)
    common_genes = rna_genes.intersection(niche_genes)
    
    print(f"\nGene overlap:")
    print(f"RNA genes: {len(rna_genes)}")
    print(f"Niche genes: {len(niche_genes)}")
    print(f"Common genes: {len(common_genes)}")
    
    if len(common_genes) == 0:
        raise ValueError("No common genes found between RNA and niche datasets!")
    
    # Subset to common genes
    print("Subsetting to common genes...")
    common_genes_list = list(common_genes)
    adata_rna_subset = adata_rna[:, common_genes_list].copy()
    adata_niche_subset = adata_niche[:, common_genes_list].copy()
    
    # Ensure same gene order
    adata_niche_subset = adata_niche_subset[:, adata_rna_subset.var_names]
    
    print(f"Subsetted RNA data: {adata_rna_subset.shape}")
    print(f"Subsetted Niche data: {adata_niche_subset.shape}")
    
    # Verify same number of cells
    if adata_rna_subset.shape[0] != adata_niche_subset.shape[0]:
        print(f"WARNING: Different number of cells - RNA: {adata_rna_subset.shape[0]}, Niche: {adata_niche_subset.shape[0]}")
        # Find common cells
        common_cells = list(set(adata_rna_subset.obs_names) & set(adata_niche_subset.obs_names))
        print(f"Common cells: {len(common_cells)}")
        adata_rna_subset = adata_rna_subset[common_cells, :].copy()
        adata_niche_subset = adata_niche_subset[common_cells, :].copy()
        print(f"Trimmed to {len(common_cells)} common cells")
    
    return adata_rna_subset, adata_niche_subset

def calculate_gene_wise_differences(adata_rna, adata_niche):
    """
    Calculate comprehensive gene-wise differences between RNA and niche modalities.
    """
    print("Converting to dense arrays...")
    if hasattr(adata_rna.X, 'toarray'):
        rna_expr = adata_rna.X.toarray()
    else:
        rna_expr = adata_rna.X

    if hasattr(adata_niche.X, 'toarray'):
        niche_expr = adata_niche.X.toarray()
    else:
        niche_expr = adata_niche.X

    print("Data conversion complete!")
    print(f"RNA expression matrix shape: {rna_expr.shape}")
    print(f"Niche expression matrix shape: {niche_expr.shape}")

    # Calculate gene-wise differences
    print("Calculating gene-wise differences...")

    gene_stats = []
    n_genes = adata_rna.shape[1]
    print(f"Processing {n_genes} genes...")

    for i, gene in enumerate(adata_rna.var_names):
        if i % 100 == 0:
            print(f"  Processing gene {i+1}/{n_genes}")
            
        rna_vals = rna_expr[:, i]
        niche_vals = niche_expr[:, i]
        
        # Skip genes with no expression
        if np.all(rna_vals == 0) and np.all(niche_vals == 0):
            continue
        
        # Calculate various metrics
        mean_abs_diff = np.mean(np.abs(rna_vals - niche_vals))
        mse = mean_squared_error(rna_vals, niche_vals)
        
        # Correlation (handle cases where one modality has no variance)
        if np.std(rna_vals) > 0 and np.std(niche_vals) > 0:
            corr, _ = pearsonr(rna_vals, niche_vals)
        else:
            corr = 0.0
        
        # Mean expression levels
        rna_mean = np.mean(rna_vals)
        niche_mean = np.mean(niche_vals)
        mean_diff = niche_mean - rna_mean
        
        # Variance differences (measure of smoothing effect)
        rna_var = np.var(rna_vals)
        niche_var = np.var(niche_vals)
        var_ratio = niche_var / (rna_var + 1e-10)  # Avoid division by zero
        
        # Expression level (for filtering low-expressed genes)
        max_expr = max(rna_mean, niche_mean)
        
        gene_stats.append({
            'gene': gene,
            'mean_abs_diff': mean_abs_diff,
            'mse': mse,
            'correlation': corr,
            'rna_mean': rna_mean,
            'niche_mean': niche_mean,
            'mean_diff': mean_diff,
            'rna_var': rna_var,
            'niche_var': niche_var,
            'var_ratio': var_ratio,
            'max_expr': max_expr,
            'smoothing_effect': 1 - var_ratio  # Higher = more smoothing
        })

    df_stats = pd.DataFrame(gene_stats)
    print(f"Calculated statistics for {len(df_stats)} genes")
    
    return df_stats

def create_comprehensive_plots(df_stats, output_dir):
    """
    Create comprehensive visualization of gene expression differences.
    """
    # Filter for reasonably expressed genes
    min_expr = 0.1
    df_filtered = df_stats[df_stats['max_expr'] > min_expr].copy()
    print(f"Analyzing {len(df_filtered)} genes with expression > {min_expr}")

    # Summary statistics
    print(f"\nSummary Statistics:")
    print(f"  Mean correlation: {df_filtered['correlation'].mean():.3f}")
    print(f"  Median correlation: {df_filtered['correlation'].median():.3f}")
    print(f"  Mean absolute difference: {df_filtered['mean_abs_diff'].mean():.3f}")
    print(f"  Mean smoothing effect: {df_filtered['smoothing_effect'].mean():.3f}")

    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('PerturbMap: RNA vs Niche Modality Gene Expression Differences', fontsize=16, fontweight='bold')

    top_n = 15

    # Plot 1: Top genes by mean absolute difference
    top_diff = df_filtered.nlargest(top_n, 'mean_abs_diff')
    axes[0,0].barh(range(len(top_diff)), top_diff['mean_abs_diff'], color='lightcoral')
    axes[0,0].set_yticks(range(len(top_diff)))
    axes[0,0].set_yticklabels(top_diff['gene'], fontsize=10)
    axes[0,0].set_xlabel('Mean Absolute Difference')
    axes[0,0].set_title(f'Top {top_n} Genes by Mean Absolute Difference')
    axes[0,0].grid(axis='x', alpha=0.3)

    # Plot 2: Lowest correlation
    low_corr = df_filtered.nsmallest(top_n, 'correlation')
    axes[0,1].barh(range(len(low_corr)), low_corr['correlation'], color='lightblue')
    axes[0,1].set_yticks(range(len(low_corr)))
    axes[0,1].set_yticklabels(low_corr['gene'], fontsize=10)
    axes[0,1].set_xlabel('Correlation (RNA vs Niche)')
    axes[0,1].set_title(f'Top {top_n} Genes with Lowest Correlation')
    axes[0,1].grid(axis='x', alpha=0.3)

    # Plot 3: Highest MSE
    top_mse = df_filtered.nlargest(top_n, 'mse')
    axes[0,2].barh(range(len(top_mse)), top_mse['mse'], color='lightgreen')
    axes[0,2].set_yticks(range(len(top_mse)))
    axes[0,2].set_yticklabels(top_mse['gene'], fontsize=10)
    axes[0,2].set_xlabel('Mean Squared Error')
    axes[0,2].set_title(f'Top {top_n} Genes by MSE')
    axes[0,2].grid(axis='x', alpha=0.3)

    # Plot 4: Distribution of correlations
    axes[1,0].hist(df_filtered['correlation'], bins=50, alpha=0.7, color='purple', edgecolor='black')
    axes[1,0].axvline(df_filtered['correlation'].mean(), color='red', linestyle='--', 
                     label=f'Mean: {df_filtered["correlation"].mean():.3f}')
    axes[1,0].set_xlabel('Correlation (RNA vs Niche)')
    axes[1,0].set_ylabel('Number of Genes')
    axes[1,0].set_title('Distribution of Gene Correlations')
    axes[1,0].legend()
    axes[1,0].grid(alpha=0.3)

    # Plot 5: Smoothing effect (variance ratio)
    most_smoothed = df_filtered.nlargest(top_n, 'smoothing_effect')
    axes[1,1].barh(range(len(most_smoothed)), most_smoothed['smoothing_effect'], color='orange')
    axes[1,1].set_yticks(range(len(most_smoothed)))
    axes[1,1].set_yticklabels(most_smoothed['gene'], fontsize=10)
    axes[1,1].set_xlabel('Smoothing Effect (1 - Var_niche/Var_rna)')
    axes[1,1].set_title(f'Top {top_n} Most Smoothed Genes')
    axes[1,1].grid(axis='x', alpha=0.3)

    # Plot 6: Mean difference (niche - rna)
    mean_diffs = df_filtered.copy()
    mean_diffs['abs_mean_diff'] = np.abs(mean_diffs['mean_diff'])
    top_mean_diff = mean_diffs.nlargest(top_n, 'abs_mean_diff')

    colors = ['red' if x > 0 else 'blue' for x in top_mean_diff['mean_diff']]
    axes[1,2].barh(range(len(top_mean_diff)), top_mean_diff['mean_diff'], color=colors)
    axes[1,2].set_yticks(range(len(top_mean_diff)))
    axes[1,2].set_yticklabels(top_mean_diff['gene'], fontsize=10)
    axes[1,2].set_xlabel('Mean Expression Difference (Niche - RNA)')
    axes[1,2].set_title(f'Top {top_n} Genes by Mean Expression Change')
    axes[1,2].axvline(0, color='black', linestyle='-', alpha=0.5)
    axes[1,2].grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/perturbmap_rna_niche_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df_filtered

def create_correlation_plots(df_filtered, output_dir):
    """
    Create overall correlation analysis plots.
    """
    print("\n=== OVERALL CORRELATION ANALYSIS ===")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter plot of mean expressions
    axes[0].scatter(df_filtered['rna_mean'], df_filtered['niche_mean'], alpha=0.6, s=20)
    axes[0].plot([0, df_filtered[['rna_mean', 'niche_mean']].max().max()], 
                 [0, df_filtered[['rna_mean', 'niche_mean']].max().max()], 'r--', alpha=0.8)
    axes[0].set_xlabel('RNA Mean Expression')
    axes[0].set_ylabel('Niche Mean Expression')
    axes[0].set_title('Mean Expression: RNA vs Niche')

    overall_corr, _ = pearsonr(df_filtered['rna_mean'], df_filtered['niche_mean'])
    axes[0].text(0.05, 0.95, f'r = {overall_corr:.3f}', transform=axes[0].transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # Variance comparison
    axes[1].scatter(df_filtered['rna_var'], df_filtered['niche_var'], alpha=0.6, s=20)
    axes[1].plot([0, df_filtered[['rna_var', 'niche_var']].max().max()], 
                 [0, df_filtered[['rna_var', 'niche_var']].max().max()], 'r--', alpha=0.8)
    axes[1].set_xlabel('RNA Variance')
    axes[1].set_ylabel('Niche Variance')
    axes[1].set_title('Expression Variance: RNA vs Niche')

    var_corr, _ = pearsonr(df_filtered['rna_var'], df_filtered['niche_var'])
    axes[1].text(0.05, 0.95, f'r = {var_corr:.3f}', transform=axes[1].transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()
    plt.savefig(f'{output_dir}/perturbmap_rna_niche_correlation_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return overall_corr, var_corr

def main():
    """
    Main function to run comprehensive PerturbMap niche vs RNA comparison.
    """
    # Create output directory
    import os
    output_dir = '../Data/processed_data'
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load data
        adata_rna, adata_niche = load_data_for_comparison()
        
        # Calculate gene-wise differences
        df_stats = calculate_gene_wise_differences(adata_rna, adata_niche)
        
        # Create comprehensive plots
        df_filtered = create_comprehensive_plots(df_stats, output_dir)
        
        # Create correlation plots
        overall_corr, var_corr = create_correlation_plots(df_filtered, output_dir)
        
        # Save results
        print("\nSaving results...")
        df_stats.to_csv('../Data/processed_data/perturbmap_rna_vs_niche_gene_comparison.csv', index=False)
        print("Gene comparison results saved to: ../Data/processed_data/perturbmap_rna_vs_niche_gene_comparison.csv")
        
        print("\n=== ANALYSIS COMPLETE ===")
        print(f"Key findings:")
        print(f"- Overall correlation between RNA and niche means: {overall_corr:.3f}")
        print(f"- Variance correlation: {var_corr:.3f}")
        print(f"- {len(df_filtered[df_filtered['correlation'] < 0.5])} genes have correlation < 0.5")
        print(f"- {len(df_filtered[df_filtered['smoothing_effect'] > 0.5])} genes show strong smoothing (>50% variance reduction)")
        print(f"- Mean smoothing effect across all genes: {df_filtered['smoothing_effect'].mean():.3f}")
        
        return df_stats, df_filtered
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    df_stats, df_filtered = main()