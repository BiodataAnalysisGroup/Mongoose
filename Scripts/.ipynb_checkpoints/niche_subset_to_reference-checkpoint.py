#!/usr/bin/env python3
"""
Subset niche data to match reference processed data exactly
"""

import argparse
import logging
import sys
import anndata as ad
import pandas as pd
import numpy as np
from pathlib import Path

def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Setup logging configuration."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )
    return logging.getLogger(__name__)

def subset_niche_to_reference(niche_adata, reference_adata, logger=None):
    """
    Subset niche data to exactly match reference data dimensions and cells.
    
    Parameters:
    -----------
    niche_adata : AnnData
        Full niche data (all genes, all spots)
    reference_adata : AnnData  
        Reference processed data (specific cells and genes)
    
    Returns:
    --------
    niche_subset : AnnData
        Niche data subset to match reference exactly
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f"Input niche data: {niche_adata.shape}")
    logger.info(f"Reference data: {reference_adata.shape}")
    
    # Get reference cells and genes
    reference_cells = reference_adata.obs_names
    reference_genes = reference_adata.var_names
    
    logger.info(f"Reference cells: {len(reference_cells)}")
    logger.info(f"Reference genes: {len(reference_genes)}")
    
    # Check cell overlap
    niche_cells = set(niche_adata.obs_names)
    ref_cells = set(reference_cells)
    common_cells = niche_cells.intersection(ref_cells)
    
    logger.info(f"Common cells: {len(common_cells)}")
    
    if len(common_cells) != len(reference_cells):
        missing_cells = ref_cells - niche_cells
        logger.warning(f"Missing {len(missing_cells)} reference cells in niche data")
        if len(missing_cells) <= 10:
            logger.warning(f"Missing cells: {list(missing_cells)}")
    
    # Check gene overlap  
    niche_genes = set(niche_adata.var_names)
    ref_genes = set(reference_genes)
    common_genes = niche_genes.intersection(ref_genes)
    
    logger.info(f"Common genes: {len(common_genes)}")
    
    if len(common_genes) != len(reference_genes):
        missing_genes = ref_genes - niche_genes
        logger.warning(f"Missing {len(missing_genes)} reference genes in niche data")
        if len(missing_genes) <= 10:
            logger.warning(f"Missing genes: {list(missing_genes)}")
    
    # Subset to common cells and genes, maintaining reference order
    logger.info("Subsetting niche data to match reference...")
    
    # First subset to common elements
    common_cells_list = [cell for cell in reference_cells if cell in niche_cells]
    common_genes_list = [gene for gene in reference_genes if gene in niche_genes]
    
    # Subset niche data
    niche_subset = niche_adata[common_cells_list, common_genes_list].copy()
    
    logger.info(f"Subset niche data: {niche_subset.shape}")
    
    # Verify order matches reference
    assert list(niche_subset.obs_names) == list(reference_adata.obs_names[:len(common_cells_list)])
    assert list(niche_subset.var_names) == list(reference_adata.var_names[:len(common_genes_list)])
    
    # Copy over any missing metadata from reference
    for col in reference_adata.obs.columns:
        if col not in niche_subset.obs.columns:
            niche_subset.obs[col] = reference_adata.obs.loc[niche_subset.obs_names, col]
            logger.info(f"Added metadata column: {col}")
    
    # Add subset info to uns (convert tuples to lists for HDF5 compatibility)
    niche_subset.uns['subset_info'] = {
        'original_niche_shape': list(niche_adata.shape),
        'reference_shape': list(reference_adata.shape),
        'subset_shape': list(niche_subset.shape),
        'common_cells': len(common_cells_list),
        'common_genes': len(common_genes_list),
        'subset_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    logger.info("Niche subset completed successfully")
    return niche_subset

def main():
    parser = argparse.ArgumentParser(
        description="Subset niche data to match reference processed data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--niche-data", required=True,
                       help="Path to full niche data (.h5ad)")
    parser.add_argument("--reference-data", required=True,
                       help="Path to reference processed data (.h5ad)")
    parser.add_argument("--output", required=True,
                       help="Output path for subset niche data (.h5ad)")
    parser.add_argument("--output-dir", default="../Data/processed_data",
                       help="Output directory if output path is relative")
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--log-file", help="Log file path (optional)")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite output file if it exists")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level, args.log_file)
    logger.info("Starting niche data subsetting")
    
    try:
        # Handle output path
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = Path(args.output_dir) / args.output
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"Output file exists: {output_path}. Use --overwrite to replace it.")
        
        # Load data
        logger.info(f"Loading niche data from: {args.niche_data}")
        niche_adata = ad.read_h5ad(args.niche_data)
        
        logger.info(f"Loading reference data from: {args.reference_data}")
        reference_adata = ad.read_h5ad(args.reference_data)
        
        # Subset niche to match reference
        niche_subset = subset_niche_to_reference(niche_adata, reference_adata, logger)
        
        # Save result
        logger.info(f"Saving subset niche data to: {output_path}")
        niche_subset.write(output_path)
        
        logger.info("Niche subsetting completed successfully!")
        
        # Print summary
        logger.info("Summary:")
        logger.info(f"  Original niche: {niche_adata.shape}")
        logger.info(f"  Reference: {reference_adata.shape}") 
        logger.info(f"  Subset niche: {niche_subset.shape}")
        logger.info(f"  Output: {output_path}")
        
        # Verify final dimensions match target
        target_shape = (65, 4001)
        if niche_subset.shape == target_shape:
            logger.info(f"✓ SUCCESS: Final shape {niche_subset.shape} matches target {target_shape}")
        else:
            logger.warning(f"⚠ WARNING: Final shape {niche_subset.shape} does not match target {target_shape}")
        
    except Exception as e:
        logger.error(f"Error during niche subsetting: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()