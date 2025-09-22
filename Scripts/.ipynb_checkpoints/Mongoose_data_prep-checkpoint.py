#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python3
"""
Robust data preparation script for single-cell analysis.
Handles data loading, phenotype assignment, and subsetting with configurable parameters.
"""

import argparse
import logging
import sys
import os
from pathlib import Path
import pandas as pd
import scanpy as sc
from typing import List, Dict, Optional

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
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

def validate_file_exists(filepath: str, description: str) -> Path:
    """Validate that a file exists and return Path object."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {filepath}")
    return path

def load_and_prepare_data(
    adata_path: str,
    pheno_path: str,
    barcode_col: str = "barcode",
    phenotypes_col: str = "phenotypes",
    kmeans_col: str = "kmeans",
    logger: logging.Logger = None
) -> sc.AnnData:
    """Load AnnData and phenotype data, merge them."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f"Loading AnnData from: {adata_path}")
    adata = sc.read_h5ad(adata_path)
    logger.info(f"Loaded AnnData: {adata.shape}")
    
    logger.info(f"Loading phenotype data from: {pheno_path}")
    pheno = pd.read_csv(pheno_path)
    logger.info(f"Loaded phenotype data: {pheno.shape}")
    
    # Set barcode as index
    if barcode_col not in pheno.columns:
        raise ValueError(f"Barcode column '{barcode_col}' not found in phenotype data")
    
    pheno = pheno.set_index(barcode_col)
    
    # Add phenotypes to adata.obs
    if phenotypes_col in pheno.columns:
        adata.obs[phenotypes_col] = adata.obs.index.map(pheno[phenotypes_col])
        logger.info(f"Added {phenotypes_col} column")
    
    if kmeans_col in pheno.columns:
        adata.obs[kmeans_col] = adata.obs.index.map(pheno[kmeans_col])
        logger.info(f"Added {kmeans_col} column")
    
    return adata

def assign_phenotype_groups(
    adata: sc.AnnData,
    excluded_tumors: List[str],
    infl_tumors: List[str],
    tgfb_tumors: List[str],
    jak_tumors: List[str],
    ifng_tumors: Optional[List[str]] = None,
    source_col: str = "phenotypes",
    target_col: str = "phenotypes_mod",
    logger: logging.Logger = None
) -> sc.AnnData:
    """Assign phenotype groups for DE analysis."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Create mapping dictionary
    mapping = {}
    mapping.update({p: "excluded_tumor" for p in excluded_tumors})
    mapping.update({p: "infl_tumor" for p in infl_tumors})
    mapping.update({p: "Tgfbr2_KO" for p in tgfb_tumors})
    mapping.update({p: "Jak2_KO" for p in jak_tumors})
    
    if ifng_tumors:
        mapping.update({p: "Ifngr2_KO" for p in ifng_tumors})
    
    # Apply mapping
    adata.obs[target_col] = adata.obs[source_col].replace(mapping)
    
    logger.info(f"Assigned phenotype groups to {target_col}")
    logger.info(f"Group counts: {adata.obs[target_col].value_counts().to_dict()}")
    
    return adata

def subset_and_process_data(
    adata: sc.AnnData,
    cell_type_filter: str,
    phenotype_filters: List[str],
    n_hvg: int = 2000,
    target_sum: float = 1e4,
    force_include_genes: Optional[List[str]] = None,
    cell_type_col: str = "cell_type",
    phenotypes_col: str = "phenotypes",
    logger: logging.Logger = None
) -> tuple:
    """Subset data and perform normalization/HVG selection."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # 1. Subset to specific cell type and phenotypes
    is_cell_type = adata.obs[cell_type_col] == cell_type_filter
    is_pheno = adata.obs[phenotypes_col].isin(phenotype_filters)
    
    logger.info(f"Filtering for {cell_type_filter} in {phenotype_filters}")
    logger.info(f"Cells matching cell type: {is_cell_type.sum()}")
    logger.info(f"Cells matching phenotypes: {is_pheno.sum()}")
    logger.info(f"Cells matching both: {(is_cell_type & is_pheno).sum()}")
    
    adata_joint = adata[is_cell_type & is_pheno].copy()
    
    if adata_joint.n_obs == 0:
        raise ValueError("No cells found matching the specified criteria")
    
    # 2. Normalize + log-transform
    logger.info("Normalizing and log-transforming data")
    sc.pp.normalize_total(adata_joint, target_sum=target_sum)
    sc.pp.log1p(adata_joint)
    
    # 3. HVG calculation
    logger.info(f"Calculating {n_hvg} highly variable genes")
    sc.pp.highly_variable_genes(adata_joint, n_top_genes=n_hvg)
    
    # Save HVG mask
    hvg_mask = adata_joint.var["highly_variable"].copy()
    logger.info(f"HVGs detected: {hvg_mask.sum()}")
    
    # Force-include specific genes
    if force_include_genes:
        hvg_mask_forced = hvg_mask.copy()
        for gene in force_include_genes:
            if gene in adata.var_names:
                hvg_mask_forced.loc[gene] = True
                logger.info(f"Force-included gene: {gene}")
            else:
                logger.warning(f"Gene '{gene}' not found in adata.var_names")
        
        logger.info(f"Total selected genes (HVGs + forced): {hvg_mask_forced.sum()}")
    else:
        hvg_mask_forced = hvg_mask
    
    return adata_joint, hvg_mask_forced

def create_individual_subsets(
    adata: sc.AnnData,
    hvg_mask: pd.Series,
    individual_phenotypes: List[str],
    cell_type_filter: str,
    cell_type_col: str = "cell_type",
    phenotypes_col: str = "phenotypes",
    logger: logging.Logger = None
) -> Dict[str, sc.AnnData]:
    """Create individual subsets for each phenotype."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    subsets = {}
    
    for pheno in individual_phenotypes:
        logger.info(f"Creating subset for {pheno}")
        
        # Safety: work on copies as lowercase strings
        phenos = adata.obs[phenotypes_col].astype(str).str.strip()
        ctypes = adata.obs[cell_type_col].astype(str).str.strip()
        
        # Create mask
        mask = (phenos == pheno) & (ctypes == cell_type_filter)
        
        if mask.sum() == 0:
            logger.warning(f"No cells found for {pheno} + {cell_type_filter}")
            continue
        
        # Create subset
        adata_subset = adata[mask].copy()
        adata_subset_hvg = adata_subset[:, hvg_mask].copy()
        
        # Add annotation
        adata_subset_hvg.var["highly_variable_joint"] = hvg_mask.loc[adata_subset_hvg.var_names].values
        
        subsets[pheno] = adata_subset_hvg
        logger.info(f"Created {pheno} subset: {adata_subset_hvg.shape}")
    
    return subsets

def save_results(
    adata_joint: sc.AnnData,
    subsets: Dict[str, sc.AnnData],
    output_dir: str,
    logger: logging.Logger = None
):
    """Save processed data."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save joint data
    joint_path = output_path / "adata_joint_processed.h5ad"
    adata_joint.write(joint_path)
    logger.info(f"Saved joint data to: {joint_path}")
    
    # Save individual subsets
    for name, adata_subset in subsets.items():
        subset_path = output_path / f"adata_{name}_hvg.h5ad"
        adata_subset.write(subset_path)
        logger.info(f"Saved {name} subset to: {subset_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Robust data preparation for single-cell analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input files
    parser.add_argument("--adata", required=True, help="Path to input AnnData (.h5ad) file")
    parser.add_argument("--pheno", required=True, help="Path to phenotype CSV file")
    
    # Column names
    parser.add_argument("--barcode-col", default="barcode", help="Barcode column name in phenotype file")
    parser.add_argument("--phenotypes-col", default="phenotypes", help="Phenotypes column name")
    parser.add_argument("--kmeans-col", default="kmeans", help="K-means column name")
    parser.add_argument("--cell-type-col", default="cell_type", help="Cell type column name")
    
    # Phenotype groups (comma-separated)
    parser.add_argument("--excluded-tumors", 
                       default="KP_3-1,KP_3-3,KP_3-2,KP_4-3,KP_4-2,KP_4-4,KP_2-3,KP_1-2,KP_2-1,KP_4-1",
                       help="Comma-separated list of excluded tumor phenotypes")
    parser.add_argument("--infl-tumors", default="KP_2-2,KP_1-1,KP_1-3",
                       help="Comma-separated list of inflammatory tumor phenotypes")
    parser.add_argument("--tgfb-tumors", default="Tgfbr2_4-1,Tgfbr2_2,Tgfbr2_1,Tgfbr2_4-2",
                       help="Comma-separated list of TGFβ tumor phenotypes")
    parser.add_argument("--jak-tumors", default="Jak2_1",
                       help="Comma-separated list of JAK tumor phenotypes")
    parser.add_argument("--ifng-tumors", default="",
                       help="Comma-separated list of IFNγ tumor phenotypes (optional)")
    
    # Analysis parameters
    parser.add_argument("--cell-type-filter", default="Dendritic Cell",
                       help="Cell type to filter for")
    parser.add_argument("--joint-phenotypes", default="KP_1-2,Tgfbr2_1",
                       help="Comma-separated phenotypes for joint analysis")
    parser.add_argument("--individual-phenotypes", default="KP_1-2,Tgfbr2_1",
                       help="Comma-separated phenotypes for individual subsets")
    parser.add_argument("--n-hvg", type=int, default=2000,
                       help="Number of highly variable genes")
    parser.add_argument("--target-sum", type=float, default=1e4,
                       help="Target sum for normalization")
    parser.add_argument("--force-include-genes", default="Tgfbr2",
                       help="Comma-separated genes to force-include in HVG selection")
    
    # Output
    parser.add_argument("--output-dir", default="../Data/processed_data",
                       help="Output directory for processed files")
    
    # Logging
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--log-file", help="Log file path (optional)")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level, args.log_file)
    logger.info("Starting data preparation script")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        # Validate input files
        validate_file_exists(args.adata, "AnnData file")
        validate_file_exists(args.pheno, "Phenotype file")
        
        # Parse comma-separated lists
        excluded_tumors = [x.strip() for x in args.excluded_tumors.split(",") if x.strip()]
        infl_tumors = [x.strip() for x in args.infl_tumors.split(",") if x.strip()]
        tgfb_tumors = [x.strip() for x in args.tgfb_tumors.split(",") if x.strip()]
        jak_tumors = [x.strip() for x in args.jak_tumors.split(",") if x.strip()]
        ifng_tumors = [x.strip() for x in args.ifng_tumors.split(",") if x.strip()] if args.ifng_tumors else None
        joint_phenotypes = [x.strip() for x in args.joint_phenotypes.split(",")]
        individual_phenotypes = [x.strip() for x in args.individual_phenotypes.split(",")]
        force_include_genes = [x.strip() for x in args.force_include_genes.split(",") if x.strip()] if args.force_include_genes else None
        
        # Load and prepare data
        adata = load_and_prepare_data(
            args.adata, args.pheno, 
            args.barcode_col, args.phenotypes_col, args.kmeans_col, 
            logger
        )
        
        # Assign phenotype groups
        adata = assign_phenotype_groups(
            adata, excluded_tumors, infl_tumors, tgfb_tumors, jak_tumors, 
            ifng_tumors, args.phenotypes_col, "phenotypes_mod", logger
        )
        
        # Subset and process data
        adata_joint, hvg_mask = subset_and_process_data(
            adata, args.cell_type_filter, joint_phenotypes,
            args.n_hvg, args.target_sum, force_include_genes,
            args.cell_type_col, args.phenotypes_col, logger
        )
        
        # Create individual subsets
        subsets = create_individual_subsets(
            adata, hvg_mask, individual_phenotypes, args.cell_type_filter,
            args.cell_type_col, args.phenotypes_col, logger
        )
        
        # Save results
        save_results(adata_joint, subsets, args.output_dir, logger)
        
        logger.info("Data preparation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during data preparation: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

