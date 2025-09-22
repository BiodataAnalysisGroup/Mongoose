#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3
"""
Activity Matrix Integration Script
Integrates scMINER activity matrices with existing AnnData objects.
Can optionally integrate phenotype data and assign phenotype groups.
"""

import argparse
import logging
import sys
import os
from pathlib import Path
import pandas as pd
import anndata as ad
import numpy as np
from typing import Optional, List

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

def subset_to_reference_cells(adata, reference_adata, logger=None):
    """Subset adata to match reference cells exactly."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    common_cells = adata.obs_names.intersection(reference_adata.obs_names)
    logger.info(f"Found {len(common_cells)} common cells between datasets")
    
    if len(common_cells) == 0:
        raise ValueError("No common cells found between datasets")
    
    # Subset and reorder to match reference exactly
    adata_subset = adata[reference_adata.obs_names].copy()
    logger.info(f"Subset adata to match reference: {adata_subset.shape}")
    
    return adata_subset

def calculate_highly_variable_activities(adata, n_hvg=2000, normalize=False, logger=None):
    """Calculate highly variable activities and subset."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f"Calculating {n_hvg} highly variable activities")
    
    # Work on a copy
    adata_hvg = adata.copy()
    
    # Optional normalization for HVG calculation
    if normalize:
        logger.info("Normalizing data for HVG calculation")
        import scanpy as sc
        sc.pp.normalize_total(adata_hvg, target_sum=1e4)
        sc.pp.log1p(adata_hvg)
    
    # Calculate HVGs
    import scanpy as sc
    sc.pp.highly_variable_genes(
        adata_hvg,
        n_top_genes=n_hvg,
        subset=False
    )
    
    # Get HVG activities and subset
    hvg_activities = adata_hvg.var_names[adata_hvg.var["highly_variable"]].tolist()
    adata_hvg_subset = adata_hvg[:, hvg_activities].copy()
    
    logger.info(f"Selected {len(hvg_activities)} highly variable activities")
    logger.info(f"Final shape: {adata_hvg_subset.shape}")
    
    # Add HVG info to uns
    adata_hvg_subset.uns["hvg_calculation"] = {
        "n_hvg_requested": n_hvg,
        "n_hvg_found": len(hvg_activities),
        "normalized_for_hvg": normalize,
        "hvg_calculation_date": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return adata_hvg_subset, hvg_activities

def load_and_merge_phenotype_data(
    adata: ad.AnnData,
    pheno_path: str,
    barcode_col: str = "barcode",
    phenotypes_col: str = "phenotypes",
    kmeans_col: str = "kmeans",
    logger: logging.Logger = None
) -> ad.AnnData:
    """Load and merge phenotype data with existing AnnData object."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if not pheno_path:
        logger.info("No phenotype file provided, skipping phenotype integration")
        return adata
    
    logger.info(f"Loading phenotype data from: {pheno_path}")
    pheno = pd.read_csv(pheno_path)
    logger.info(f"Loaded phenotype data: {pheno.shape}")
    
    # Set barcode as index
    if barcode_col not in pheno.columns:
        raise ValueError(f"Barcode column '{barcode_col}' not found in phenotype data")
    
    pheno = pheno.set_index(barcode_col)
    
    # Add phenotypes to adata.obs (only if columns exist and aren't already present)
    if phenotypes_col in pheno.columns:
        adata.obs[phenotypes_col] = adata.obs.index.map(pheno[phenotypes_col])
        logger.info(f"Added/updated {phenotypes_col} column")
    
    if kmeans_col in pheno.columns:
        adata.obs[kmeans_col] = adata.obs.index.map(pheno[kmeans_col])
        logger.info(f"Added/updated {kmeans_col} column")
    
    # Add any other columns from phenotype data
    other_cols = [col for col in pheno.columns if col not in [phenotypes_col, kmeans_col]]
    for col in other_cols:
        adata.obs[col] = adata.obs.index.map(pheno[col])
        logger.info(f"Added/updated {col} column")
    
    return adata

def assign_phenotype_groups(
    adata: ad.AnnData,
    excluded_tumors: List[str],
    infl_tumors: List[str],
    tgfb_tumors: List[str],
    jak_tumors: List[str],
    ifng_tumors: Optional[List[str]] = None,
    source_col: str = "phenotypes",
    target_col: str = "phenotypes_mod",
    logger: logging.Logger = None
) -> ad.AnnData:
    """Assign phenotype groups for DE analysis."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if source_col not in adata.obs.columns:
        logger.warning(f"Source column '{source_col}' not found in adata.obs, skipping phenotype group assignment")
        return adata
    
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

def create_adata_from_activity_matrix(original_adata, activity_matrix_path, logger=None):
    """
    Create a new AnnData object from scMINER activity matrix while preserving
    all metadata from the original AnnData object.

    Parameters:
    -----------
    original_adata : AnnData
        Original Perturb-Map AnnData object
    activity_matrix_path : str
        Path to the scMINER activity matrix CSV file
    logger : logging.Logger
        Logger instance

    Returns:
    --------
    new_adata : AnnData
        New AnnData object with activity matrix as .X and preserved metadata
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f"Loading activity matrix from: {activity_matrix_path}")
    
    # Load the activity matrix
    activity_df = pd.read_csv(activity_matrix_path, index_col=0)
    logger.info(f"Activity matrix shape: {activity_df.shape}")

    # Transpose to have cells as rows and activities as columns (standard AnnData format)
    activity_df = activity_df.T
    logger.info(f"After transpose: {activity_df.shape}")

    # Verify that the cell barcodes match between original and activity matrix
    original_barcodes = set(original_adata.obs_names)
    activity_barcodes = set(activity_df.index)

    # Check for mismatches
    missing_in_activity = original_barcodes - activity_barcodes
    missing_in_original = activity_barcodes - original_barcodes

    if missing_in_activity:
        logger.warning(f"{len(missing_in_activity)} cells from original data missing in activity matrix")
        if len(missing_in_activity) <= 10:
            logger.debug(f"Missing cells: {list(missing_in_activity)}")
    
    if missing_in_original:
        logger.warning(f"{len(missing_in_original)} cells in activity matrix not found in original data")
        if len(missing_in_original) <= 10:
            logger.debug(f"Extra cells: {list(missing_in_original)}")

    # Get common cells (intersection)
    common_cells = list(original_barcodes.intersection(activity_barcodes))
    logger.info(f"Found {len(common_cells)} common cells between datasets")

    if len(common_cells) == 0:
        raise ValueError("No common cells found between original data and activity matrix")

    # Subset both datasets to common cells
    activity_subset = activity_df.loc[common_cells]
    original_subset_idx = [i for i, cell in enumerate(original_adata.obs_names) if cell in common_cells]

    # Create new AnnData object with activity matrix as .X
    new_adata = ad.AnnData(X=activity_subset.values)

    # Set cell barcodes (observations)
    new_adata.obs_names = activity_subset.index

    # Set activity names (variables/features)
    new_adata.var_names = activity_subset.columns
    logger.info(f"Number of activities: {len(new_adata.var_names)}")

    # Transfer all observation metadata (.obs)
    new_adata.obs = original_adata.obs.loc[common_cells].copy()
    logger.info(f"Transferred {len(new_adata.obs.columns)} observation metadata columns")

    # Transfer all variable metadata (.var) - create empty for activities
    new_adata.var['activity_type'] = 'scMINER_activity'

    # Transfer unstructured annotations (.uns)
    new_adata.uns = original_adata.uns.copy()

    # Add information about the transformation
    new_adata.uns['scMINER_info'] = {
        'original_n_genes': original_adata.n_vars,
        'activity_matrix_path': activity_matrix_path,
        'n_activities': new_adata.n_vars,
        'transformation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'common_cells': len(common_cells),
        'original_cells': original_adata.n_obs,
        'activity_cells': len(activity_barcodes)
    }

    # Transfer observation pairwise annotations (.obsp) if they exist
    if original_adata.obsp:
        logger.info("Transferring observation pairwise annotations (.obsp)")
        original_cell_idx = {cell: i for i, cell in enumerate(original_adata.obs_names)}
        new_cell_indices = [original_cell_idx[cell] for cell in common_cells]

        for key in original_adata.obsp.keys():
            # Subset the matrix to common cells (both rows and columns)
            original_matrix = original_adata.obsp[key]
            new_adata.obsp[key] = original_matrix[np.ix_(new_cell_indices, new_cell_indices)]
            logger.debug(f"Transferred .obsp['{key}']")

    # Transfer observation-variable mappings (.obsm) - spatial coordinates, embeddings, etc.
    if original_adata.obsm:
        logger.info("Transferring observation mappings (.obsm)")
        for key in original_adata.obsm.keys():
            new_adata.obsm[key] = original_adata.obsm[key][original_subset_idx].copy()
            logger.debug(f"Transferred .obsm['{key}'] with shape {new_adata.obsm[key].shape}")

    logger.info("Successfully created new AnnData object with activity matrix")
    logger.info(f"  - Shape: {new_adata.shape}")
    logger.info(f"  - Observations (cells): {new_adata.n_obs}")
    logger.info(f"  - Variables (activities): {new_adata.n_vars}")
    logger.info(f"  - Preserved metadata: obs({len(new_adata.obs.columns)}), "
               f"obsm({len(new_adata.obsm)}), obsp({len(new_adata.obsp)}), uns({len(new_adata.uns)})")

    return new_adata

def compare_adata_objects(original_adata, new_adata, logger=None):
    """Compare original and new AnnData objects"""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("=== AnnData Comparison ===")
    logger.info(f"Original shape: {original_adata.shape}")
    logger.info(f"New shape: {new_adata.shape}")
    logger.info(f"Common cells: {len(set(original_adata.obs_names).intersection(set(new_adata.obs_names)))}")
    logger.info(f"Original .obs columns: {list(original_adata.obs.columns)}")
    logger.info(f"New .obs columns: {list(new_adata.obs.columns)}")
    logger.info(f"Original .obsm keys: {list(original_adata.obsm.keys()) if original_adata.obsm else 'None'}")
    logger.info(f"New .obsm keys: {list(new_adata.obsm.keys()) if new_adata.obsm else 'None'}")
    logger.info(f"Original .uns keys: {list(original_adata.uns.keys()) if original_adata.uns else 'None'}")
    logger.info(f"New .uns keys: {list(new_adata.uns.keys()) if new_adata.uns else 'None'}")
    
    # Check if specific metadata was preserved
    if hasattr(new_adata, 'uns') and 'scMINER_info' in new_adata.uns:
        info = new_adata.uns['scMINER_info']
        logger.info(f"scMINER transformation info:")
        for key, value in info.items():
            logger.info(f"  {key}: {value}")

def main():
    parser = argparse.ArgumentParser(
        description="Integrate scMINER activity matrix with AnnData object",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input files
    parser.add_argument("--input-adata", required=True, 
                       help="Path to input AnnData (.h5ad) file")
    parser.add_argument("--activity-matrix", required=True,
                       help="Path to scMINER activity matrix CSV file")
    parser.add_argument("--pheno", 
                       help="Path to phenotype CSV file (optional - adds/updates phenotype data)")
    parser.add_argument("--reference-adata",
                       help="Path to reference AnnData for cell subset matching (optional)")
    
    # HVG parameters
    parser.add_argument("--calculate-hvg", action="store_true",
                       help="Calculate highly variable genes on activity data")
    parser.add_argument("--n-hvg", type=int, default=2000,
                       help="Number of highly variable genes to select")
    parser.add_argument("--normalize-for-hvg", action="store_true",
                       help="Normalize data before HVG calculation")
    
    # Column names for phenotype data
    parser.add_argument("--barcode-col", default="barcode", 
                       help="Barcode column name in phenotype file")
    parser.add_argument("--phenotypes-col", default="phenotypes", 
                       help="Phenotypes column name")
    parser.add_argument("--kmeans-col", default="kmeans", 
                       help="K-means column name")
    
    # Phenotype groups (comma-separated) - only used if pheno file provided
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
    parser.add_argument("--assign-phenotype-groups", action="store_true",
                       help="Assign phenotype groups (creates phenotypes_mod column)")
    
    # Output
    parser.add_argument("--output", required=True,
                       help="Output path for new AnnData file (.h5ad)")
    parser.add_argument("--output-dir", default="../Data/processed_data",
                       help="Output directory (used if --output is just a filename)")
    
    # Options
    parser.add_argument("--compare", action="store_true",
                       help="Compare original and new AnnData objects")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite output file if it exists")
    
    # Logging
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--log-file", help="Log file path (optional)")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level, args.log_file)
    logger.info("Starting activity matrix integration")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        # Validate input files
        validate_file_exists(args.input_adata, "Input AnnData file")
        validate_file_exists(args.activity_matrix, "Activity matrix file")
        
        if args.pheno:
            validate_file_exists(args.pheno, "Phenotype file")
        
        if args.reference_adata:
            validate_file_exists(args.reference_adata, "Reference AnnData file")
        
        # Handle output path
        output_path = Path(args.output)
        if not output_path.is_absolute() and not str(output_path).startswith('.'):
            # If just a filename, put it in output_dir
            output_path = Path(args.output_dir) / args.output
        
        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if output exists
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"Output file exists: {output_path}. Use --overwrite to replace it.")
        
        # Load original AnnData
        logger.info(f"Loading original AnnData from: {args.input_adata}")
        original_adata = ad.read_h5ad(args.input_adata)
        logger.info(f"Loaded original AnnData: {original_adata.shape}")
        
        # Merge phenotype data if provided
        if args.pheno:
            logger.info("Integrating phenotype data with original AnnData")
            original_adata = load_and_merge_phenotype_data(
                original_adata, 
                args.pheno, 
                args.barcode_col, 
                args.phenotypes_col, 
                args.kmeans_col, 
                logger
            )
            
            # Assign phenotype groups if requested
            if args.assign_phenotype_groups:
                # Parse comma-separated lists
                excluded_tumors = [x.strip() for x in args.excluded_tumors.split(",") if x.strip()]
                infl_tumors = [x.strip() for x in args.infl_tumors.split(",") if x.strip()]
                tgfb_tumors = [x.strip() for x in args.tgfb_tumors.split(",") if x.strip()]
                jak_tumors = [x.strip() for x in args.jak_tumors.split(",") if x.strip()]
                ifng_tumors = [x.strip() for x in args.ifng_tumors.split(",") if x.strip()] if args.ifng_tumors else None
                
                original_adata = assign_phenotype_groups(
                    original_adata, excluded_tumors, infl_tumors, tgfb_tumors, jak_tumors, 
                    ifng_tumors, args.phenotypes_col, "phenotypes_mod", logger
                )
        
        # Create new AnnData with activity matrix
        new_adata = create_adata_from_activity_matrix(
            original_adata, 
            args.activity_matrix, 
            logger
        )
        
        # Subset to reference cells if provided
        if args.reference_adata:
            logger.info(f"Loading reference AnnData from: {args.reference_adata}")
            reference_adata = ad.read_h5ad(args.reference_adata)
            logger.info(f"Loaded reference AnnData: {reference_adata.shape}")
            
            new_adata = subset_to_reference_cells(new_adata, reference_adata, logger)
        
        # Calculate HVG if requested
        if args.calculate_hvg:
            new_adata, hvg_activities = calculate_highly_variable_activities(
                new_adata, 
                args.n_hvg, 
                args.normalize_for_hvg, 
                logger
            )
            logger.info(f"Selected {len(hvg_activities)} highly variable activities")
        
        # Compare objects if requested
        if args.compare:
            compare_adata_objects(original_adata, new_adata, logger)
        
        # Save new AnnData
        logger.info(f"Saving new AnnData to: {output_path}")
        new_adata.write(output_path)
        logger.info("Activity matrix integration completed successfully!")
        
        # Print summary
        logger.info(f"Summary:")
        logger.info(f"  Input: {args.input_adata} ({original_adata.shape})")
        logger.info(f"  Activity matrix: {args.activity_matrix}")
        if args.pheno:
            logger.info(f"  Phenotype data: {args.pheno}")
        if args.reference_adata:
            logger.info(f"  Reference data: {args.reference_adata}")
        logger.info(f"  Output: {output_path} ({new_adata.shape})")
        logger.info(f"  Activities: {new_adata.n_vars}")
        if args.calculate_hvg:
            logger.info(f"  HVG activities selected: {args.n_hvg}")
        
    except Exception as e:
        logger.error(f"Error during activity matrix integration: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

