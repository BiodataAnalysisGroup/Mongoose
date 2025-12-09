#!/usr/bin/env python3
"""
GenKI Discovery Parameter Sweep - No Ground Truth Required
Systematic exploration to maximize significant gene detection

Usage:
    # Using preset config
    python genki_discovery_sweep.py \
        --adata ../Data/processedData3/kp13.h5ad \
        --gene Tgfbr2 \
        --config lr_sweep \
        --output_dir ../GenKI_Discovery/sweep_$(date +%Y%m%d_%H%M%S)
    
    # Using custom parameter ranges
    python genki_discovery_sweep.py \
        --adata ../Data/processedData3/kp13.h5ad \
        --gene Tgfbr2 \
        --learning_rate 1e-3 5e-3 1e-2 \
        --beta 1e-4 5e-4 \
        --epochs 300 \
        --output_dir ../GenKI_Discovery/custom
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from itertools import product

# ============================================================================
# PARAMETER CONFIGURATIONS
# ============================================================================

PRESET_CONFIGS = {
    "test": {
        "learning_rate": [1e-2],
        "beta": [5e-4],
        "epochs": [200],
        "edge_threshold_percentile": [85],
        "n_permutations": [100],
        "alpha": [0.05]
    },
    
    "lr_sweep": {
        # Only vary learning rate, keep everything else fixed
        "learning_rate": [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1],
        "beta": [5e-4],
        "epochs": [300],
        "edge_threshold_percentile": [85],
        "n_permutations": [200],
        "alpha": [0.05]
    },
    
    "beta_sweep": {
        "learning_rate": [1e-2],
        "beta": [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
        "epochs": [300],
        "edge_threshold_percentile": [85],
        "n_permutations": [200],
        "alpha": [0.05]
    },
    
    "rational": {
        "learning_rate": [1e-3, 5e-3, 1e-2, 5e-2],
        "beta": [1e-5, 1e-4, 5e-4, 1e-3],
        "epochs": [200, 300, 500],
        "edge_threshold_percentile": [80, 85, 90],
        "n_permutations": [200, 500],
        "alpha": [0.01, 0.05, 0.10]
    },
    
    "sensitivity_focused": {
        "learning_rate": [5e-3, 1e-2],
        "beta": [1e-5, 5e-5, 1e-4],
        "epochs": [300, 500],
        "edge_threshold_percentile": [80, 85],
        "n_permutations": [200, 500],
        "alpha": [0.05, 0.10]
    },
    
    "specificity_focused": {
        "learning_rate": [1e-3, 5e-3],
        "beta": [5e-4, 1e-3, 5e-3],
        "epochs": [300, 500],
        "edge_threshold_percentile": [85, 90, 95],
        "n_permutations": [500, 1000],
        "alpha": [0.01, 0.05]
    },
    
    "comprehensive": {
        "learning_rate": [5e-4, 1e-3, 5e-3, 1e-2, 5e-2],
        "beta": [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3],
        "epochs": [200, 300, 500],
        "edge_threshold_percentile": [80, 85, 90, 95],
        "n_permutations": [200, 500],
        "alpha": [0.01, 0.05, 0.10]
    }
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def setup_output_dir(base_dir: str) -> Path:
    """Create and return output directory"""
    output_path = Path(base_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path

def print_progress_bar(iteration: int, total: int, prefix: str = '', 
                       suffix: str = '', length: int = 50):
    """Print a progress bar"""
    percent = 100 * (iteration / float(total))
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '░' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent:.1f}% {suffix}', end='', flush=True)
    if iteration == total:
        print()

def calculate_discovery_metrics(results_df: pd.DataFrame, kl_divergences: np.ndarray) -> Dict:
    """Calculate discovery-oriented metrics without ground truth"""
    
    metrics = {}
    
    # Significance at different thresholds
    if 'padj' in results_df.columns:
        metrics['n_sig_0.001'] = (results_df['padj'] < 0.001).sum()
        metrics['n_sig_0.01'] = (results_df['padj'] < 0.01).sum()
        metrics['n_sig_0.05'] = (results_df['padj'] < 0.05).sum()
        metrics['n_sig_0.10'] = (results_df['padj'] < 0.10).sum()
        
        # P-value distribution metrics
        if len(results_df) > 0:
            metrics['median_padj'] = results_df['padj'].median()
            metrics['min_padj'] = results_df['padj'].min()
            metrics['pct_padj_lt_0.05'] = (results_df['padj'] < 0.05).mean() * 100
    
    elif 'pvalue' in results_df.columns:
        metrics['n_sig_0.001'] = (results_df['pvalue'] < 0.001).sum()
        metrics['n_sig_0.01'] = (results_df['pvalue'] < 0.01).sum()
        metrics['n_sig_0.05'] = (results_df['pvalue'] < 0.05).sum()
        metrics['n_sig_0.10'] = (results_df['pvalue'] < 0.10).sum()
        
        if len(results_df) > 0:
            metrics['median_pvalue'] = results_df['pvalue'].median()
            metrics['min_pvalue'] = results_df['pvalue'].min()
            metrics['pct_pvalue_lt_0.05'] = (results_df['pvalue'] < 0.05).mean() * 100
    
    # KL divergence distribution
    metrics['kl_mean'] = np.mean(kl_divergences)
    metrics['kl_std'] = np.std(kl_divergences)
    metrics['kl_max'] = np.max(kl_divergences)
    metrics['kl_median'] = np.median(kl_divergences)
    
    # Dynamic range (how spread out are the KL values)
    metrics['kl_q95'] = np.percentile(kl_divergences, 95)
    metrics['kl_q99'] = np.percentile(kl_divergences, 99)
    metrics['kl_dynamic_range'] = metrics['kl_max'] / (metrics['kl_median'] + 1e-10)
    
    # Total genes analyzed
    metrics['n_total_genes'] = len(results_df)
    
    return metrics

# ============================================================================
# MAIN BENCHMARK EXECUTION
# ============================================================================

def run_single_genki(
    adata_path: str,
    ko_gene: str,
    params: Dict,
    run_id: int,
    grn_rebuild: bool = False,
    n_cpus: int = 20,
    verbose: bool = False
) -> Tuple[pd.DataFrame, np.ndarray, Dict]:
    """
    Run a single GenKI analysis with specified parameters
    Returns: (full_results_df, kl_divergences, metrics_dict)
    """
    try:
        # Import GenKI components
        from GenKI.preprocesing import build_adata
        from GenKI.dataLoader import DataLoader
        from GenKI.train import VGAE_trainer
        from GenKI import utils as gk_utils
        
        # Load data
        adata = build_adata(adata_path, uppercase=False)
        
        # Setup DataLoader
        data_wrapper = DataLoader(
            adata=adata,
            target_gene=[ko_gene],
            target_cell=None,
            obs_label="ident",
            GRN_file_dir="GRNs",
            rebuild_GRN=grn_rebuild,
            pcNet_name="PertMap_Auto",
            verbose=verbose,
            n_cpus=n_cpus,
        )
        
        data_wt = data_wrapper.load_data()
        data_ko = data_wrapper.load_kodata()
        
        # Train VGAE
        trainer = VGAE_trainer(
            data_wt,
            epochs=params["epochs"],
            lr=params["learning_rate"],
            log_dir=None,
            beta=params["beta"],
            seed=8096,
            verbose=verbose,
        )
        trainer.train()
        
        # Get latent representations
        z_mu_wt, z_std_wt = trainer.get_latent_vars(data_wt)
        z_mu_ko, z_std_ko = trainer.get_latent_vars(data_ko)
        
        # Calculate KL divergence for all genes
        dis = gk_utils.get_distance(z_mu_ko, z_std_ko, z_mu_wt, z_std_wt, by="KL")
        
        # Handle both numpy array and pandas Series
        if hasattr(dis, 'values'):
            kl_divergences = dis.values
        else:
            kl_divergences = np.array(dis) if not isinstance(dis, np.ndarray) else dis
        
        # Permutation test
        if params["n_permutations"] > 0:
            null_dist = trainer.pmt(data_ko, n=params["n_permutations"], by="KL")
            res_full = gk_utils.get_generank(data_wt, dis, null_dist)
        else:
            # Just rank by KL divergence
            res_full = gk_utils.get_generank(data_wt, dis, rank=True)
        
        # Calculate discovery metrics
        metrics = calculate_discovery_metrics(res_full, kl_divergences)
        
        # Get top significant genes
        if 'padj' in res_full.columns:
            sig_genes = res_full[res_full['padj'] < params['alpha']]
        elif 'pvalue' in res_full.columns:
            sig_genes = res_full[res_full['pvalue'] < params['alpha']]
        else:
            sig_genes = res_full.head(100)  # Fallback
        
        metrics['significant_genes'] = list(sig_genes.index.astype(str))
        
        return res_full, kl_divergences, metrics
        
    except Exception as e:
        if verbose:
            print(f"\n  ERROR in run {run_id}: {str(e)}")
        raise e

def run_parameter_sweep(
    adata_path: str,
    ko_gene: str,
    param_grid: Dict,
    output_dir: Path,
    n_cpus: int = 20,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Execute parameter sweep across all combinations
    """
    print("\n" + "="*80)
    print(f"  GenKI Discovery Parameter Sweep: {ko_gene}")
    print("="*80)
    
    # Generate parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))
    
    print(f"\nParameter sweep configuration:")
    print(f"  Total combinations: {len(combinations)}")
    print(f"  CPUs: {n_cpus}")
    print(f"  Output: {output_dir}")
    
    print("\nParameter ranges:")
    for name, values in param_grid.items():
        print(f"  {name}: {values}")
    
    # Confirm if many runs
    if len(combinations) > 20 and not verbose:
        print(f"\n⚠️  This will run {len(combinations)} GenKI analyses")
        response = input("Continue? [y/N]: ")
        if response.lower() != 'y':
            print("Cancelled.")
            sys.exit(0)
    
    print("\n" + "="*80)
    print("Running parameter sweep...")
    print("="*80 + "\n")
    
    # Run sweep
    results = []
    all_significant_genes = {}
    all_kl_distributions = {}
    
    for idx, combo in enumerate(combinations, 1):
        param_dict = dict(zip(param_names, combo))
        
        if verbose:
            print(f"\n[{idx}/{len(combinations)}] Parameters:")
            for k, v in param_dict.items():
                print(f"  {k}: {v}")
        else:
            print_progress_bar(
                idx, len(combinations), 
                prefix='Progress:', 
                suffix=f'Run {idx}/{len(combinations)}'
            )
        
        try:
            # Run GenKI
            full_res, kl_div, metrics = run_single_genki(
                adata_path=adata_path,
                ko_gene=ko_gene,
                params=param_dict,
                run_id=idx,
                grn_rebuild=(idx == 1),  # Only rebuild GRN once
                n_cpus=n_cpus,
                verbose=verbose
            )
            
            # Store results
            result_entry = {
                "run_id": idx,
                "ko_gene": ko_gene,
                **param_dict,
                **{k: v for k, v in metrics.items() if k != "significant_genes"},
                "status": "success"
            }
            
            all_significant_genes[idx] = metrics["significant_genes"]
            all_kl_distributions[idx] = kl_div.tolist()
            
            if verbose:
                print(f"  → Significant genes (α={param_dict['alpha']}): {result_entry['n_sig_0.05']}")
                print(f"  → KL range: {metrics['kl_median']:.3f} (median) to {metrics['kl_max']:.3f} (max)")
        
        except Exception as e:
            result_entry = {
                "run_id": idx,
                "ko_gene": ko_gene,
                **param_dict,
                "status": "failed",
                "error": str(e)
            }
            if verbose:
                print(f"  ✗ Failed: {str(e)}")
        
        results.append(result_entry)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    print("\n\n" + "="*80)
    print("Saving results...")
    print("="*80)
    
    results_path = output_dir / "discovery_sweep_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"✓ Results: {results_path}")
    
    sig_genes_path = output_dir / "significant_genes.json"
    with open(sig_genes_path, 'w') as f:
        json.dump(all_significant_genes, f, indent=2)
    print(f"✓ Significant genes: {sig_genes_path}")
    
    kl_dist_path = output_dir / "kl_distributions.json"
    with open(kl_dist_path, 'w') as f:
        json.dump(all_kl_distributions, f, indent=2)
    print(f"✓ KL distributions: {kl_dist_path}")
    
    # Save configuration
    config_path = output_dir / "sweep_config.json"
    with open(config_path, 'w') as f:
        json.dump({
            "ko_gene": ko_gene,
            "adata_path": adata_path,
            "n_combinations": len(combinations),
            "parameter_grid": param_grid,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)
    print(f"✓ Configuration: {config_path}")
    
    # Analyze and save best parameters
    analyze_discovery_results(results_df, output_dir)
    
    return results_df

def analyze_discovery_results(results_df: pd.DataFrame, output_dir: Path):
    """Analyze results and identify optimal parameter configurations"""
    
    print("\n" + "="*80)
    print("Discovery Analysis Summary")
    print("="*80)
    
    successful = results_df[results_df["status"] == "success"]
    
    if len(successful) == 0:
        print("\n⚠️  No successful runs to analyze")
        return
    
    print(f"\nRuns: {len(results_df)} total | {len(successful)} successful")
    
    # Helper function for type conversion
    def convert_to_native(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        else:
            return obj
    
    # Find configurations with most significant genes at different thresholds
    rankings = {}
    
    for threshold_col in ['n_sig_0.001', 'n_sig_0.01', 'n_sig_0.05', 'n_sig_0.10']:
        if threshold_col in successful.columns:
            best_idx = successful[threshold_col].idxmax()
            rankings[threshold_col] = successful.loc[best_idx]
    
    # Configuration with best dynamic range (most separation)
    if 'kl_dynamic_range' in successful.columns:
        best_range_idx = successful['kl_dynamic_range'].idxmax()
        rankings['best_dynamic_range'] = successful.loc[best_range_idx]
    
    # Print top findings
    param_cols = ["learning_rate", "beta", "epochs", "edge_threshold_percentile", 
                  "n_permutations", "alpha"]
    
    print("\n" + "-"*80)
    print("🏆 OPTIMAL CONFIGURATIONS")
    print("-"*80)
    
    for metric_name, config in rankings.items():
        print(f"\n{metric_name.upper().replace('_', ' ')}:")
        
        if 'n_sig' in metric_name:
            print(f"  Significant genes: {int(config[metric_name])}")
        elif 'dynamic_range' in metric_name:
            print(f"  KL dynamic range: {config['kl_dynamic_range']:.2f}")
        
        print("  Parameters:")
        for col in param_cols:
            if col in config.index:
                print(f"    {col}: {config[col]}")
    
    # Summary statistics
    print("\n" + "-"*80)
    print("Discovery Statistics Across All Runs")
    print("-"*80)
    
    for col in ['n_sig_0.05', 'n_sig_0.01', 'kl_dynamic_range', 'kl_mean']:
        if col in successful.columns:
            print(f"{col:25s} → mean: {successful[col].mean():.2f}, "
                  f"std: {successful[col].std():.2f}, "
                  f"max: {successful[col].max():.2f}")
    
    # Save optimal configs
    optimal_configs = {}
    for metric_name, config in rankings.items():
        metrics_dict = {
            k: convert_to_native(config[k]) 
            for k in config.index 
            if k not in param_cols and k != 'status'
        }
        params_dict = {
            col: convert_to_native(config[col]) 
            for col in param_cols 
            if col in config.index
        }
        
        optimal_configs[metric_name] = {
            "metrics": metrics_dict,
            "parameters": params_dict
        }
    
    optimal_path = output_dir / "optimal_parameters.json"
    with open(optimal_path, 'w') as f:
        json.dump(optimal_configs, f, indent=2)
    print(f"\n✓ Optimal parameters saved: {optimal_path}")
    
    # Top 10 by significance at α=0.05
    if 'n_sig_0.05' in successful.columns:
        print("\n" + "-"*80)
        print("Top 10 Configurations (by significant genes at α=0.05)")
        print("-"*80)
        top10 = successful.nlargest(10, "n_sig_0.05")
        display_cols = ["run_id"] + param_cols + ["n_sig_0.05", "n_sig_0.01", "kl_dynamic_range"]
        display_cols = [c for c in display_cols if c in top10.columns]
        print(top10[display_cols].to_string(index=False))
        
        top10_path = output_dir / "top10_configurations.csv"
        top10[display_cols].to_csv(top10_path, index=False)
        print(f"\n✓ Top 10 saved: {top10_path}")
    
    # Parameter sensitivity analysis
    print("\n" + "-"*80)
    print("Parameter Impact Analysis")
    print("-"*80)
    
    for param in param_cols:
        if param in successful.columns and 'n_sig_0.05' in successful.columns:
            # Check if parameter was actually varied
            unique_vals = successful[param].nunique()
            if unique_vals > 1:
                grouped = successful.groupby(param)['n_sig_0.05'].agg(['mean', 'std', 'max'])
                print(f"\n{param}:")
                print(grouped.to_string())

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="GenKI Discovery Parameter Sweep (No Ground Truth Required)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example Usage:

    # Using preset config
    python genki_discovery_sweep.py \\
        --adata ../Data/processedData3/kp13.h5ad \\
        --gene Tgfbr2 \\
        --config lr_sweep \\
        --output_dir ../GenKI_Discovery/lr_sweep
    
    # Custom parameter ranges (any combination)
    python genki_discovery_sweep.py \\
        --adata ../Data/processedData3/kp13.h5ad \\
        --gene Tgfbr2 \\
        --learning_rate 1e-3 5e-3 1e-2 \\
        --beta 1e-4 5e-4 \\
        --epochs 300 500 \\
        --output_dir ../GenKI_Discovery/custom
    
    # Single parameter sweep (learning rate only)
    python genki_discovery_sweep.py \\
        --adata ../Data/processedData3/kp13.h5ad \\
        --gene Tgfbr2 \\
        --learning_rate 1e-4 5e-4 1e-3 5e-3 1e-2 5e-2 1e-1 \\
        --beta 5e-4 \\
        --epochs 300 \\
        --edge_threshold_percentile 85 \\
        --n_permutations 200 \\
        --alpha 0.05 \\
        --output_dir ../GenKI_Discovery/lr_only
    
    # Two parameter sweep (learning rate × beta)
    python genki_discovery_sweep.py \\
        --adata ../Data/processedData3/kp13.h5ad \\
        --gene Tgfbr2 \\
        --learning_rate 1e-3 1e-2 5e-2 \\
        --beta 1e-5 1e-4 5e-4 1e-3 \\
        --epochs 300 \\
        --output_dir ../GenKI_Discovery/lr_beta_grid
        """
    )
    
    # Required arguments
    parser.add_argument("--adata", required=True,
                        help="Path to AnnData h5ad file")
    parser.add_argument("--gene", required=True,
                        help="Gene to knockout (e.g., Tgfbr2)")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for results")
    
    # Configuration: either preset or custom parameters
    parser.add_argument("--config", 
                       choices=["test", "lr_sweep", "beta_sweep", "rational", 
                               "sensitivity_focused", "specificity_focused", "comprehensive"],
                       help="Preset parameter configuration (optional if custom params provided)")
    
    # Custom parameter ranges (optional - override preset)
    parser.add_argument("--learning_rate", nargs='+', type=float,
                       help="Learning rate values to test (e.g., 1e-3 5e-3 1e-2)")
    parser.add_argument("--beta", nargs='+', type=float,
                       help="Beta (KL regularization) values to test (e.g., 1e-4 5e-4 1e-3)")
    parser.add_argument("--epochs", nargs='+', type=int,
                       help="Epoch values to test (e.g., 200 300 500)")
    parser.add_argument("--edge_threshold_percentile", nargs='+', type=int,
                       help="Edge threshold percentile values (e.g., 80 85 90)")
    parser.add_argument("--n_permutations", nargs='+', type=int,
                       help="Number of permutations to test (e.g., 100 200 500)")
    parser.add_argument("--alpha", nargs='+', type=float,
                       help="Significance threshold values (e.g., 0.01 0.05 0.10)")
    
    # Optional arguments
    parser.add_argument("--cpus", type=int, default=20,
                        help="Number of CPUs to use (default: 20)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.adata).exists():
        print(f"ERROR: AnnData file not found: {args.adata}")
        sys.exit(1)
    
    # Determine parameter grid
    custom_params_provided = any([
        args.learning_rate, args.beta, args.epochs,
        args.edge_threshold_percentile, args.n_permutations, args.alpha
    ])
    
    if custom_params_provided:
        # Build custom parameter grid
        print("Using custom parameter ranges...")
        
        # Start with defaults
        param_grid = {
            "learning_rate": [1e-2],
            "beta": [5e-4],
            "epochs": [300],
            "edge_threshold_percentile": [85],
            "n_permutations": [200],
            "alpha": [0.05]
        }
        
        # Override with provided values
        if args.learning_rate:
            param_grid["learning_rate"] = args.learning_rate
        if args.beta:
            param_grid["beta"] = args.beta
        if args.epochs:
            param_grid["epochs"] = args.epochs
        if args.edge_threshold_percentile:
            param_grid["edge_threshold_percentile"] = args.edge_threshold_percentile
        if args.n_permutations:
            param_grid["n_permutations"] = args.n_permutations
        if args.alpha:
            param_grid["alpha"] = args.alpha
            
    elif args.config:
        # Use preset configuration
        print(f"Using preset configuration: {args.config}")
        param_grid = PRESET_CONFIGS[args.config]
    else:
        print("ERROR: Must provide either --config or custom parameter ranges")
        print("See --help for examples")
        sys.exit(1)
    
    # Setup output directory
    output_dir = setup_output_dir(args.output_dir)
    
    # Run parameter sweep
    results_df = run_parameter_sweep(
        adata_path=args.adata,
        ko_gene=args.gene,
        param_grid=param_grid,
        output_dir=output_dir,
        n_cpus=args.cpus,
        verbose=args.verbose
    )
    
    print("\n" + "="*80)
    print("✓ Discovery sweep complete!")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print("\nNext steps:")
    print(f"  1. Review results: {output_dir}/discovery_sweep_results.csv")
    print(f"  2. Check optimal params: {output_dir}/optimal_parameters.json")
    print(f"  3. See significant genes: {output_dir}/significant_genes.json")
    print(f"  4. Analyze KL distributions: {output_dir}/kl_distributions.json")
    print()

if __name__ == "__main__":
    main()