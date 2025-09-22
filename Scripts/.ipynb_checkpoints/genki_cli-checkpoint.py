#!/usr/bin/env python3
"""
GenKI Command Line Interface
Run GenKI analysis with command line arguments and get all significantly perturbed genes
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
from scipy.sparse import csr_matrix
from scipy.sparse import issparse
import GenKI as gk
from GenKI.preprocesing import build_adata
from GenKI.dataLoader import DataLoader
from GenKI.train import VGAE_trainer
from GenKI import utils
from datetime import datetime

sc.settings.verbosity = 0

def genki_cli_analysis(
    adata_path,
    genes_of_interest,
    npz_filename="PertMap_Auto",
    extract_adjacency=True,
    # VGAE parameters
    epochs=300,
    learning_rate=5e-2,
    beta=5e-4,
    seed=8096,
    # DataLoader parameters
    grn_rebuild=True,
    n_cpus=20,
    # Significance test parameters
    run_significance_test=True,
    n_permutations=100,
    alpha=0.05,  # significance threshold
    # Output parameters
    base_output_dir="../GenKI",
    num_top_genes_for_summary=50  # Just for summary display, all significant genes are saved
):
    """
    Run GenKI analysis via CLI and return all significantly perturbed genes
    """
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    genes_str = "_".join(genes_of_interest[:3])
    if len(genes_of_interest) > 3:
        genes_str += f"_plus{len(genes_of_interest)-3}more"
    
    output_dir = os.path.join(base_output_dir, f"GenKI_CLI_{genes_str}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"GenKI CLI Analysis Started")
    print(f"Timestamp: {timestamp}")
    print(f"Output Directory: {output_dir}")
    print(f"="*80)
    
    # Load data
    print(f"Loading data from: {adata_path}")
    try:
        adata = build_adata(adata_path, uppercase=False)
        print(f"✓ Data loaded: {adata.n_vars} genes, {adata.n_obs} cells")
    except Exception as e:
        print(f"✗ Error loading data: {str(e)}")
        sys.exit(1)
    
    # Results storage
    all_results = {
        'timestamp': timestamp,
        'output_directory': output_dir,
        'genes_analyzed': genes_of_interest,
        'parameters': {
            'epochs': epochs,
            'learning_rate': learning_rate,
            'beta': beta,
            'seed': seed,
            'n_permutations': n_permutations,
            'alpha': alpha,
            'n_cpus': n_cpus
        },
        'individual_results': {},
        'all_significant_genes': {},
        'adjacency_matrices': {}
    }
    
    # Process each gene
    for i, gene_of_interest in enumerate(genes_of_interest, 1):
        print(f"\n[{i}/{len(genes_of_interest)}] Processing gene: {gene_of_interest}")
        print("-" * 60)
        
        gene_results = {}
        
        try:
            # Setup DataLoader
            print("Setting up DataLoader...")
            data_wrapper = DataLoader(
                adata=adata,
                target_gene=[gene_of_interest],
                target_cell=None,
                obs_label="ident",
                GRN_file_dir="GRNs",
                rebuild_GRN=grn_rebuild,
                pcNet_name=npz_filename,
                verbose=False,  # Reduce verbosity for CLI
                n_cpus=n_cpus,
            )
            
            data_wt = data_wrapper.load_data()
            data_ko = data_wrapper.load_kodata()
            print(f"✓ Data prepared for {gene_of_interest}")
            
            # Train VGAE
            print("Training VGAE model...")
            sensei = VGAE_trainer(
                data_wt,
                epochs=epochs,
                lr=learning_rate,
                log_dir=None,
                beta=beta,
                seed=seed,
                verbose=False,
            )
            sensei.train()
            print("✓ VGAE training completed")
            
            # Get latent representations and KL divergence
            print("Computing KL divergences...")
            z_mu_wt, z_std_wt = sensei.get_latent_vars(data_wt)
            z_mu_ko, z_std_ko = sensei.get_latent_vars(data_ko)
            dis = gk.utils.get_distance(z_mu_ko, z_std_ko, z_mu_wt, z_std_wt, by="KL")
            print("✓ KL divergences computed")
            
            # Get initial ranking (for summary)
            res_raw = utils.get_generank(data_wt, dis, rank=True)
            gene_results['total_genes_ranked'] = len(res_raw)
            
            # Run significance test to get ALL significant genes
            if run_significance_test:
                print(f"Running permutation test ({n_permutations} permutations)...")
                try:
                    null_dist = sensei.pmt(data_ko, n=n_permutations, by="KL")
                    
                    # Get significance results - THIS GIVES US ALL SIGNIFICANT GENES
                    res_significant = utils.get_generank(data_wt, dis, null_dist)
                    
                    # Filter for significant genes based on alpha threshold
                    if 'padj' in res_significant.columns:
                        significant_genes = res_significant[res_significant['padj'] < alpha]
                    elif 'pvalue' in res_significant.columns:
                        significant_genes = res_significant[res_significant['pvalue'] < alpha]
                    else:
                        # If no p-value columns, use the full result
                        print("Warning: No p-value column found, using all genes from permutation test")
                        significant_genes = res_significant
                    
                    gene_results['significance_results'] = res_significant
                    gene_results['significant_genes'] = significant_genes
                    gene_results['n_significant_genes'] = len(significant_genes)
                    
                    print(f"✓ Significance test completed")
                    print(f"✓ Found {len(significant_genes)} significantly perturbed genes (α = {alpha})")
                    
                    # Store all significant genes for this KO
                    all_results['all_significant_genes'][gene_of_interest] = significant_genes
                    
                    # Save individual results
                    gene_output_dir = os.path.join(output_dir, f'{gene_of_interest}_results')
                    os.makedirs(gene_output_dir, exist_ok=True)
                    
                    # Save ALL significant genes
                    sig_genes_path = os.path.join(gene_output_dir, f'ALL_Significant_Genes_{gene_of_interest}_alpha{alpha}.csv')
                    significant_genes.to_csv(sig_genes_path)
                    print(f"✓ All significant genes saved to: {sig_genes_path}")
                    
                    # Save complete significance results
                    all_sig_path = os.path.join(gene_output_dir, f'Complete_Significance_Results_{gene_of_interest}.csv')
                    res_significant.to_csv(all_sig_path)
                    
                    # Also save top N for summary (traditional approach)
                    top_genes = res_raw.head(num_top_genes_for_summary)
                    top_genes_path = os.path.join(gene_output_dir, f'Top{num_top_genes_for_summary}_Ranked_Genes_{gene_of_interest}.csv')
                    top_genes.to_csv(top_genes_path)
                    
                except Exception as e:
                    print(f"✗ Significance test failed: {str(e)}")
                    gene_results['significance_error'] = str(e)
                    # Fall back to top genes without significance
                    top_genes = res_raw.head(num_top_genes_for_summary)
                    all_results['all_significant_genes'][gene_of_interest] = top_genes
            else:
                # No significance test, just use top N genes
                top_genes = res_raw.head(num_top_genes_for_summary)
                all_results['all_significant_genes'][gene_of_interest] = top_genes
                gene_results['n_significant_genes'] = num_top_genes_for_summary
            
            all_results['individual_results'][gene_of_interest] = gene_results
            print(f"✓ Successfully completed analysis for {gene_of_interest}")
            
        except Exception as e:
            print(f"✗ Error processing {gene_of_interest}: {str(e)}")
            all_results['individual_results'][gene_of_interest] = {'error': str(e)}
            continue
    
    # Extract adjacency matrix if requested
    if extract_adjacency:
        print(f"\n{'='*60}")
        print("Extracting GRN adjacency matrix...")
        print("-" * 60)
        
        try:
            pc_fp = f"{npz_filename}.npz"
            if os.path.exists(pc_fp):
                print(f"Loading GRN from: {pc_fp}")
                loader = np.load(pc_fp, allow_pickle=True)
                data, indices, indptr = loader['data'], loader['indices'], loader['indptr']
                shape = tuple(loader['shape'])
                W_sparse = csr_matrix((data, indices, indptr), shape=shape)
                
                # Threshold (top 15%)
                thr = np.percentile(np.abs(W_sparse.data), 85)
                W_thr = W_sparse.copy()
                W_thr.data[np.abs(W_thr.data) < thr] = 0
                W_thr.eliminate_zeros()
                
                # Create adjacency matrices
                genes = adata.var_names.to_list()
                adj_weighted_df = pd.DataFrame(W_thr.toarray(), index=genes, columns=genes)
                
                # Edge list
                rows, cols = W_thr.nonzero()
                edge_weights = W_thr[rows, cols].A1
                edge_list = pd.DataFrame({
                    "source": [genes[i] for i in rows],
                    "target": [genes[j] for j in cols],
                    "weight": edge_weights
                })
                
                all_results['adjacency_matrices'] = {
                    'weighted_adjacency': adj_weighted_df,
                    'edge_list': edge_list,
                    'threshold_used': thr,
                    'total_edges': W_thr.nnz
                }
                
                # Save adjacency matrices
                adj_dir = os.path.join(output_dir, 'adjacency_matrices')
                os.makedirs(adj_dir, exist_ok=True)
                adj_weighted_df.to_csv(os.path.join(adj_dir, f'{npz_filename}_weighted_adjacency.csv'))
                edge_list.to_csv(os.path.join(adj_dir, f'{npz_filename}_edge_list.csv'), index=False)
                
                print(f"✓ Adjacency matrices extracted ({W_thr.nnz} edges)")
                
        except Exception as e:
            print(f"✗ Error extracting adjacency matrix: {str(e)}")
    
    # Create final summary
    print(f"\n{'='*80}")
    print("CREATING FINAL SUMMARY")
    print("="*80)
    
    # Combine all significant genes
    all_significant_combined = []
    for gene_ko, sig_genes in all_results['all_significant_genes'].items():
        if not sig_genes.empty:
            sig_genes_with_ko = sig_genes.copy()
            sig_genes_with_ko['KO_gene'] = gene_ko
            all_significant_combined.append(sig_genes_with_ko)
    
    if all_significant_combined:
        combined_significant = pd.concat(all_significant_combined, ignore_index=False)
        combined_path = os.path.join(output_dir, 'ALL_Significant_Genes_Combined.csv')
        combined_significant.to_csv(combined_path)
        print(f"✓ All significant genes combined saved to: {combined_path}")
        
        # Summary statistics
        total_significant = len(combined_significant)
        unique_significant = len(combined_significant.index.unique())
        
        print(f"✓ Total significant gene-KO pairs: {total_significant}")
        print(f"✓ Unique significant genes: {unique_significant}")
    
    # Save comprehensive summary
    with open(os.path.join(output_dir, 'GenKI_CLI_Summary.txt'), 'w') as f:
        f.write("GenKI CLI Analysis Summary\n")
        f.write("="*50 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Command: python genki_cli.py with specified parameters\n")
        f.write(f"Input Data: {adata_path}\n")
        f.write(f"GRN Network: {npz_filename}.npz\n")
        f.write(f"Output Directory: {output_dir}\n\n")
        
        f.write("Analysis Parameters:\n")
        f.write(f"  Genes analyzed: {', '.join(genes_of_interest)}\n")
        f.write(f"  VGAE epochs: {epochs}\n")
        f.write(f"  Learning rate: {learning_rate}\n")
        f.write(f"  Beta: {beta}\n")
        f.write(f"  CPUs used: {n_cpus}\n")
        f.write(f"  Permutations: {n_permutations}\n")
        f.write(f"  Significance threshold (α): {alpha}\n\n")
        
        f.write("Results Summary:\n")
        successful_genes = [k for k, v in all_results['individual_results'].items() if 'error' not in v]
        f.write(f"  Successfully analyzed genes: {len(successful_genes)}\n")
        
        for gene in successful_genes:
            gene_res = all_results['individual_results'][gene]
            n_sig = gene_res.get('n_significant_genes', 'Unknown')
            f.write(f"  {gene}: {n_sig} significant genes\n")
        
        if all_significant_combined:
            f.write(f"\n  Total significant gene-KO pairs: {total_significant}\n")
            f.write(f"  Unique significant genes across all KOs: {unique_significant}\n")
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"Results directory: {output_dir}")
    print(f"Key files:")
    print(f"  - ALL_Significant_Genes_Combined.csv (main result)")
    print(f"  - Individual gene folders with complete results")
    print(f"  - GenKI_CLI_Summary.txt (analysis summary)")
    
    return all_results

def main():
    parser = argparse.ArgumentParser(
        description='GenKI Command Line Interface - Get all significantly perturbed genes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python genki_cli.py --adata_path "../Data/processed_data/adata_KP_1-2_hvg.h5ad" --genes "Tgfbr2" --n_cpus 20
  
  python genki_cli.py --adata_path "data.h5ad" --genes "Tgfbr2,Tp53,Myc" --n_cpus 20 --extract_adjacency --run_significance_test
        """
    )
    
    # Required arguments
    parser.add_argument('--adata_path', required=True, 
                       help='Path to the AnnData .h5ad file')
    parser.add_argument('--genes', required=True,
                       help='Comma-separated list of genes to analyze (e.g., "Tgfbr2" or "Tgfbr2,Tp53,Myc")')
    
    # Optional arguments
    parser.add_argument('--num_top_genes_for_summary', type=int, default=50,
                       help='Number of top genes to save for summary (default: 50). All significant genes are always saved.')
    parser.add_argument('--npz_filename', default='PertMap_Auto',
                       help='Name for the GRN .npz file (default: PertMap_Auto)')
    parser.add_argument('--n_cpus', type=int, default=20,
                       help='Number of CPUs to use (default: 20)')
    
    # VGAE parameters
    parser.add_argument('--epochs', type=int, default=300,
                       help='Number of training epochs (default: 300)')
    parser.add_argument('--learning_rate', type=float, default=5e-2,
                       help='Learning rate (default: 0.05)')
    parser.add_argument('--beta', type=float, default=5e-4,
                       help='Beta regularization parameter (default: 0.0005)')
    parser.add_argument('--seed', type=int, default=8096,
                       help='Random seed (default: 8096)')
    
    # Significance test parameters
    parser.add_argument('--run_significance_test', action='store_true',
                       help='Run permutation test for significance (recommended)')
    parser.add_argument('--n_permutations', type=int, default=100,
                       help='Number of permutations for significance test (default: 100)')
    parser.add_argument('--alpha', type=float, default=0.05,
                       help='Significance threshold for p-values (default: 0.05)')
    
    # Other options
    parser.add_argument('--extract_adjacency', action='store_true',
                       help='Extract and save GRN adjacency matrices')
    parser.add_argument('--no_grn_rebuild', action='store_true',
                       help='Do not rebuild GRN (use existing)')
    parser.add_argument('--base_output_dir', default='../GenKI',
                       help='Base output directory (default: ../GenKI)')
    
    args = parser.parse_args()
    
    # Parse genes list
    genes_list = [gene.strip() for gene in args.genes.split(',')]
    
    # Validate inputs
    if not os.path.exists(args.adata_path):
        print(f"Error: AnnData file not found: {args.adata_path}")
        sys.exit(1)
    
    # Run analysis
    try:
        results = genki_cli_analysis(
            adata_path=args.adata_path,
            genes_of_interest=genes_list,
            npz_filename=args.npz_filename,
            extract_adjacency=args.extract_adjacency,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            beta=args.beta,
            seed=args.seed,
            grn_rebuild=not args.no_grn_rebuild,
            n_cpus=args.n_cpus,
            run_significance_test=args.run_significance_test,
            n_permutations=args.n_permutations,
            alpha=args.alpha,
            base_output_dir=args.base_output_dir,
            num_top_genes_for_summary=args.num_top_genes_for_summary
        )
        
        print(f"\n🎉 SUCCESS! All significant genes have been identified and saved.")
        print(f"📁 Check the results in: {results['output_directory']}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Analysis failed: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()