#!/usr/bin/env python3
"""
Niche Modality Calculation Script (UnitedNet Implementation)
Calculates spatial niche modality following the UnitedNet methodology.
Based on: Tang et al. Nature Communications 2023

FIXED VERSION: Aligned with actual UnitedNet implementation + Phenotype support
"""

import argparse
import logging
import sys
import os
from pathlib import Path
import pandas as pd
import anndata as ad
import numpy as np
import scipy.sparse as sp
import networkx as nx
from scipy.spatial import distance
import torch
from typing import Optional

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

class NicheParams:
    """Parameters for niche calculation following UnitedNet methodology."""
    def __init__(self, knn_distance_type='euclidean', k=15, using_mask=False):
        self.knn_distanceType = knn_distance_type  # Distance metric for KNN
        self.k = k                                 # Number of nearest neighbors
        self.using_mask = using_mask               # Use mask for semi-supervised learning

def sparse_mx_to_torch_sparse_tensor(sparse_mx: sp.spmatrix) -> torch.Tensor:
    """Convert a scipy sparse matrix to a torch.sparse_coo_tensor."""
    coo = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64))
    values = torch.from_numpy(coo.data)
    shape = torch.Size(coo.shape)
    return torch.sparse_coo_tensor(indices, values, shape)

def preprocess_graph(adj):
    """Preprocess adjacency matrix following UnitedNet methodology."""
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0], dtype=adj.dtype, format="coo")
    rowsum = np.array(adj_.sum(1)).flatten()
    # avoid division by zero if there are isolated nodes
    rowsum[rowsum == 0.0] = 1.0
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5))
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return adj_normalized, sparse_mx_to_torch_sparse_tensor(adj_normalized)

def edgeList2edgeDict(edgeList, nodesize):
    """Convert edge list to edge dictionary."""
    graphdict = {}
    tdict = {}
    for edge in edgeList:
        end1 = edge[0]
        end2 = edge[1]
        tdict[end1] = ""
        tdict[end2] = ""
        tmplist = graphdict.get(end1, [])
        tmplist.append(end2)
        graphdict[end1] = tmplist

    # ensure all nodes present
    for i in range(nodesize):
        if i not in tdict:
            graphdict[i] = []
    return graphdict

def graph_computing(spatial_coords, cell_num, params, logger=None):
    """
    Compute graph edges based on spatial coordinates.
    Implements UnitedNet's adaptive boundary method.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f"Computing spatial graph with k={params.k}, distance={params.knn_distanceType}")
    
    edgeList = []
    for node_idx in range(cell_num):
        if node_idx % 1000 == 0:
            logger.debug(f"Processing node {node_idx}/{cell_num}")
        
        tmp = spatial_coords[node_idx, :].reshape(1, -1)
        distMat = distance.cdist(tmp, spatial_coords, getattr(params, "knn_distanceType", "euclidean"))
        res = distMat.argsort(axis=1)[:, :params.k + 1]  # self + k nearest
        tmpdist = distMat[0, res[0][1:params.k + 1]]
        boundary = np.mean(tmpdist) + np.std(tmpdist)
        
        for j in range(1, params.k + 1):
            neighbor_idx = int(res[0][j])
            dist_to_neighbor = distMat[0, neighbor_idx]
            w = 1.0 if dist_to_neighbor <= boundary else 0.0
            if w > 0:  # Only add valid edges
                edgeList.append((node_idx, neighbor_idx, w))
    
    logger.info(f"Generated {len(edgeList)} valid edges")
    return edgeList

def graph_construction(spatial_coords, cell_N, params, logger=None):
    """Construct spatial graph following UnitedNet methodology."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("Starting spatial graph construction")
    
    adata_Adj = graph_computing(spatial_coords, cell_N, params, logger)
    graphdict = edgeList2edgeDict(adata_Adj, cell_N)
    G = nx.from_dict_of_lists(graphdict)
    adj_org = nx.adjacency_matrix(G)

    # remove diagonal
    adj_m1 = adj_org - sp.dia_matrix((adj_org.diagonal()[np.newaxis, :], [0]), shape=adj_org.shape)
    adj_m1.eliminate_zeros()

    # normalize + labels
    adj_norm_write, adj_norm_m1 = preprocess_graph(adj_m1)
    adj_label_m1 = adj_m1 + sp.eye(adj_m1.shape[0], format="csr")
    adj_label_m1 = torch.FloatTensor(adj_label_m1.toarray())

    # normalization factor for loss terms (as in typical GCN recipes)
    denom = float((adj_m1.shape[0] * adj_m1.shape[0] - adj_m1.sum()) * 2)
    norm_m1 = (adj_m1.shape[0] * adj_m1.shape[0]) / denom if denom != 0 else 1.0

    graph_dict = {
        "adj_org": adj_org,
        "adj_norm": adj_norm_m1,
        "adj_label": adj_label_m1,
        "norm_value": norm_m1
    }
    if getattr(params, "using_mask", False):
        graph_dict["adj_mask"] = torch.ones(cell_N, cell_N)
    
    logger.info("Spatial graph construction completed")
    return graph_dict, adj_norm_write

def add_phenotype_data(adata, pheno_csv_path, barcode_col, pheno_cols, logger=None):
    """Add phenotype data from CSV to AnnData object."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if not pheno_csv_path:
        return adata
    
    logger.info(f"Loading phenotype data from: {pheno_csv_path}")
    
    # Load phenotype CSV
    pheno = pd.read_csv(pheno_csv_path)
    
    # Set barcodes as index
    if barcode_col in pheno.columns:
        pheno = pheno.set_index(barcode_col)
    else:
        logger.warning(f"Barcode column '{barcode_col}' not found in phenotype CSV")
        return adata
    
    # Add specified phenotype columns
    if pheno_cols:
        for col in pheno_cols:
            if col in pheno.columns:
                adata.obs[col] = adata.obs.index.map(pheno[col])
                logger.info(f"Added phenotype column: {col}")
            else:
                logger.warning(f"Phenotype column '{col}' not found in CSV")
    else:
        # Add all columns except barcode
        for col in pheno.columns:
            if col != barcode_col:
                adata.obs[col] = adata.obs.index.map(pheno[col])
                logger.info(f"Added phenotype column: {col}")
    
    return adata

def calculate_niche_modality_unitednet(adata, spatial_key='spatial', params=None, logger=None):
    """
    Calculate niche modality following UnitedNet's EXACT methodology.
    This is the CORRECTED version that matches the reference implementation.
    
    Key Formula (from UnitedNet paper):
    x_i^(v niche) = sum_{j=1}^J x_j^(v) * w_ij
    
    where w_ij = (1/distance(s_i, s_j)) / sum_{j=1}^J (1/distance(s_i, s_j))
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if params is None:
        params = NicheParams()
    
    logger.info(f"Calculating niche modality for {adata.shape[0]} cells, {adata.shape[1]} genes")
    
    # Check if spatial coordinates exist
    if spatial_key not in adata.obsm:
        raise ValueError(f"Spatial coordinates '{spatial_key}' not found in adata.obsm")
    
    spatial_coords = adata.obsm[spatial_key]
    logger.info(f"Using spatial coordinates with shape: {spatial_coords.shape}")
    
    # Construct spatial graph to get neighbor adjacency
    graph_dict, adj_norm_write = graph_construction(spatial_coords, adata.shape[0], params, logger)
    
    # Get adjacency label matrix (includes self-connections) - this defines neighbors
    adj_label = graph_dict['adj_label'].cpu().detach().numpy().astype(bool)  # (n_obs, n_obs)
    
    # Convert expression matrix to dense if needed
    if hasattr(adata.X, 'toarray'):
        expression_matrix = adata.X.toarray()
    else:
        expression_matrix = adata.X.copy()
    
    logger.info("Calculating niche expression using UnitedNet methodology")
    
    # Initialize niche expression matrix
    n_cells, n_genes = adata.shape
    niche_expression = np.zeros((n_cells, n_genes), dtype=float)
    
    # Calculate niche expression for each cell - CORRECTED TO MATCH REFERENCE
    for ind in range(n_cells):
        if ind % 1000 == 0:
            logger.debug(f"Processing cell {ind}/{n_cells}")
        
        # Get neighbor mask from adjacency matrix
        nbr_mask = adj_label[ind]  # boolean mask of neighbors (includes self)
        nbr_idx = np.flatnonzero(nbr_mask)  # indices of neighbors
        
        if len(nbr_idx) == 0:
            # No neighbors found, use self expression
            niche_expression[ind, :] = expression_matrix[ind, :]
            continue
        
        # Calculate distances from current cell to neighbors
        distMat = distance.cdist(
            spatial_coords[ind, :].reshape(1, -1),
            spatial_coords[nbr_idx, :],
            metric='euclidean'
        )  # shape (1, num_neighbors)
        
        # CORRECTED: Follow reference code exactly
        # invert nonzero distances; keep 0 for self
        nz = distMat > 0
        distMat[nz] = 1.0 / distMat[nz]
        # Note: self-connection (distance=0) stays as 0
        
        # Normalize weights
        s = distMat.sum()
        if s == 0:
            weights = np.ones((1, distMat.shape[1])) / max(1, distMat.shape[1])
        else:
            weights = distMat / s  # shape (1, num_neighbors)
        
        # Get neighbor expression matrix
        X_neighbors = expression_matrix[nbr_idx, :]  # (num_neighbors, n_genes)
        if hasattr(X_neighbors, "toarray"):  # sparse -> dense
            X_neighbors = X_neighbors.toarray()
        
        # Calculate weighted average expression (exactly as in reference)
        niche_expression[ind, :] = (weights @ X_neighbors).ravel()
    
    logger.info("Niche modality calculation completed")
    return niche_expression, graph_dict

def create_niche_adata(original_adata, niche_expression, logger=None):
    """Create new AnnData object with niche expression."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("Creating niche AnnData object")
    
    # Create new AnnData object
    adata_niche = ad.AnnData(X=niche_expression)
    
    # Copy metadata
    adata_niche.obs = original_adata.obs.copy()
    adata_niche.var = original_adata.var.copy()
    adata_niche.obs_names = original_adata.obs_names
    adata_niche.var_names = original_adata.var_names
    
    # Copy other annotations
    if original_adata.obsm:
        for key in original_adata.obsm.keys():
            adata_niche.obsm[key] = original_adata.obsm[key].copy()
    
    if original_adata.uns:
        adata_niche.uns = original_adata.uns.copy()
    
    # Add niche-specific information
    adata_niche.uns["modality"] = "niche"
    adata_niche.uns["niche_method"] = "UnitedNet"
    adata_niche.uns["niche_calculation_date"] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    
    logger.info(f"Created niche AnnData object: {adata_niche.shape}")
    return adata_niche

def save_niche_csv(adata_niche, output_path, logger=None):
    """Save niche expression as CSV file in UnitedNet format."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Create DataFrame with genes as rows, cells as columns (UnitedNet format)
    niche_df = pd.DataFrame(
        adata_niche.X.T, 
        index=adata_niche.var_names, 
        columns=adata_niche.obs_names
    )
    
    niche_df.to_csv(output_path)
    logger.info(f"Saved niche expression CSV to: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Calculate niche modality using UnitedNet methodology",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input files
    parser.add_argument("--input-adata", required=True,
                       help="Path to input AnnData (.h5ad) file")
    parser.add_argument("--pheno-csv", 
                       help="Optional: Path to phenotype CSV file")
    parser.add_argument("--pheno-barcode-col", default="barcode",
                       help="Column name for barcodes in phenotype CSV")
    parser.add_argument("--pheno-cols", nargs="*",
                       help="Phenotype columns to add (e.g., 'phenotypes' 'kmeans')")
    
    # Spatial parameters
    parser.add_argument("--spatial-key", default="spatial",
                       help="Key for spatial coordinates in adata.obsm")
    
    # UnitedNet graph parameters
    parser.add_argument("--k", type=int, default=15,
                       help="Number of nearest neighbors for graph construction")
    parser.add_argument("--distance-type", default="euclidean",
                       choices=["euclidean", "manhattan", "cosine"],
                       help="Distance metric for KNN")
    parser.add_argument("--using-mask", action="store_true",
                       help="Use mask for semi-supervised learning")
    
    # Output
    parser.add_argument("--output", required=True,
                       help="Output path for niche AnnData (.h5ad)")
    parser.add_argument("--output-csv", 
                       help="Optional: save niche expression as CSV")
    parser.add_argument("--output-dir", default="../Data/processed_data",
                       help="Output directory (used if paths are relative)")
    
    # Options
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite output files if they exist")
    
    # Logging
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--log-file", help="Log file path (optional)")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level, args.log_file)
    logger.info("Starting UnitedNet niche modality calculation")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        # Validate input files
        validate_file_exists(args.input_adata, "Input AnnData file")
        
        # Handle output paths
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = Path(args.output_dir) / args.output
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"Output file exists: {output_path}. Use --overwrite to replace it.")
        
        # Load input data
        logger.info(f"Loading input AnnData from: {args.input_adata}")
        adata = ad.read_h5ad(args.input_adata)
        logger.info(f"Loaded AnnData: {adata.shape}")
        
        # Add phenotype data if provided
        adata = add_phenotype_data(adata, args.pheno_csv, args.pheno_barcode_col, args.pheno_cols, logger)
        
        # Set up parameters
        params = NicheParams(
            knn_distance_type=args.distance_type,
            k=args.k,
            using_mask=args.using_mask
        )
        
        # Calculate niche modality using UnitedNet methodology
        niche_expression, graph_dict = calculate_niche_modality_unitednet(
            adata, args.spatial_key, params, logger
        )
        
        # Create niche AnnData
        adata_niche = create_niche_adata(adata, niche_expression, logger)
        
        # Save results
        logger.info(f"Saving niche AnnData to: {output_path}")
        adata_niche.write(output_path)
        
        # Save CSV if requested
        if args.output_csv:
            csv_path = Path(args.output_csv)
            if not csv_path.is_absolute():
                csv_path = Path(args.output_dir) / args.output_csv
            save_niche_csv(adata_niche, csv_path, logger)
        
        logger.info("UnitedNet niche modality calculation completed successfully!")
        
        # Print summary
        logger.info(f"Summary:")
        logger.info(f"  Input: {args.input_adata} ({adata.shape})")
        logger.info(f"  Output: {output_path} ({adata_niche.shape})")
        logger.info(f"  Spatial key: {args.spatial_key}")
        logger.info(f"  Parameters: k={args.k}, distance={args.distance_type}")
        if args.pheno_csv:
            logger.info(f"  Phenotypes: Added from {args.pheno_csv}")
        logger.info(f"  Graph edges: {len(graph_dict['adj_org'].data) if 'adj_org' in graph_dict else 'N/A'}")
        
    except Exception as e:
        logger.error(f"Error during niche calculation: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()