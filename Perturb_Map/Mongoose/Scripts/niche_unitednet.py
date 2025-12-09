#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Niche modality (UnitedNet-style) from spatial coordinates

Features
--------
- Builds a kNN graph over `adata.obsm['spatial']`
- Creates "niche" expression by inverse-distance weighted neighbor averaging
- Preserves AnnData metadata (obs/var/obsm/uns)
- Optional phenotype CSV join, barcode normalization, reference alignment
- Importable API + CLI

Quick use (Python):
-------------------
from niche_unitednet import run_niche_pipeline
niche_adata = run_niche_pipeline(
    input_adata_path="input.h5ad",
    output_path="niche.h5ad",
    k=15, distance_type="euclidean"
)

Quick use (CLI):
----------------
python niche_unitednet.py \
  --input input.h5ad \
  --output niche.h5ad \
  --k 15 --distance euclidean \
  --pheno ./pheno.csv --pheno-barcode-col barcode \
  --pheno-cols phenotypes kmeans
"""
from __future__ import annotations

import sys
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import networkx as nx
from scipy.spatial import distance
import torch


# ---------------------------- Logging & Utils ---------------------------- #

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    fmt = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format=fmt,
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger("niche_unitednet")


def validate_file_exists(filepath: str, label: str) -> Path:
    p = Path(filepath)
    if not p.exists():
        raise FileNotFoundError(f"{label} not found: {filepath}")
    return p


def normalize_barcodes(idx: pd.Index) -> pd.Index:
    """Normalize common 10x/Visium barcode quirks: drop trailing '-1', uppercase."""
    return idx.astype(str).str.replace(r"-1$", "", regex=True).str.upper()


def add_phenotype_data(
    adata: ad.AnnData,
    pheno_csv_path: Optional[str],
    barcode_col: str = "barcode",
    pheno_cols: Optional[List[str]] = None,
    logger: Optional[logging.Logger] = None,
) -> ad.AnnData:
    logger = logger or logging.getLogger("niche_unitednet")
    if not pheno_csv_path:
        return adata
    validate_file_exists(pheno_csv_path, "Phenotype CSV")
    pheno = pd.read_csv(pheno_csv_path)
    if barcode_col not in pheno.columns:
        logger.warning(f"Barcode column '{barcode_col}' not found in phenotype CSV; skipping phenotype join.")
        return adata
    
    # Normalize phenotype barcodes to match AnnData normalization
    pheno[barcode_col] = normalize_barcodes(pd.Index(pheno[barcode_col]))
    pheno = pheno.set_index(barcode_col)
    
    # Check overlap
    overlap = adata.obs.index.intersection(pheno.index)
    if len(overlap) == 0:
        logger.warning(f"No barcode overlap between AnnData and phenotype CSV. Skipping phenotype join.")
        return adata
    logger.info(f"Barcode overlap: {len(overlap)} / {len(adata)} cells")
    
    cols = pheno_cols or list(pheno.columns)
    for col in cols:
        if col not in pheno.columns:
            logger.warning(f"Phenotype column '{col}' not found; skipped.")
            continue
        
        # Map the values
        mapped_values = adata.obs.index.map(pheno[col])
        
        # Convert to categorical if appropriate
        if pheno[col].dtype == 'object' or pheno[col].dtype.name == 'category':
            # String or already categorical -> make categorical
            adata.obs[col] = pd.Categorical(mapped_values)
        elif pd.api.types.is_numeric_dtype(pheno[col]):
            # Numeric: check if it looks categorical (small number of unique values)
            n_unique = pheno[col].nunique()
            if n_unique <= 50:  # threshold for treating as categorical
                adata.obs[col] = pd.Categorical(mapped_values)
            else:
                adata.obs[col] = mapped_values
        else:
            adata.obs[col] = pd.Categorical(mapped_values)
        
        n_matched = mapped_values.notna().sum()
        logger.info(f"Added phenotype column: {col} ({n_matched}/{len(adata)} cells matched, type: {adata.obs[col].dtype})")
    return adata




















# ---------------------------- Spatial helpers ---------------------------- #

def _lowres_scalef(adata: ad.AnnData) -> Optional[float]:
    """Try to read Visium scalefactor (lowres preferred)."""
    if "spatial" not in adata.uns or not adata.uns["spatial"]:
        return None
    lib = list(adata.uns["spatial"].keys())[0]
    scf = adata.uns["spatial"][lib].get("scalefactors", {})
    return scf.get("tissue_lowres_scalef") or scf.get("tissue_hires_scalef")


def harmonize_spatial_scaling(
    target: ad.AnnData, source: ad.AnnData, logger: Optional[logging.Logger] = None
) -> ad.AnnData:
    """Scale source pixel coords to match target pixel scale (if both present)."""
    logger = logger or logging.getLogger("niche_unitednet")
    if "spatial" not in source.obsm or "spatial" not in target.obsm:
        return source
    sa, sb = _lowres_scalef(target), _lowres_scalef(source)
    if sa and sb and sa != sb:
        factor = sa / sb
        out = source.copy()
        out.obsm["spatial"] = out.obsm["spatial"] * factor
        logger.info(f"Harmonized pixel scale: source *= {factor:.6f} to match reference.")
        return out
    return source


def subset_to_reference_cells(
    adata: ad.AnnData, reference_adata: ad.AnnData, normalize_barcodes_flag: bool = True
) -> ad.AnnData:
    """Subset `adata` to `reference_adata` cells (keeping reference order)."""
    A, B = adata, reference_adata
    if normalize_barcodes_flag:
        if not A.obs_names.equals(normalize_barcodes(A.obs_names)):
            A = A.copy(); A.obs_names = normalize_barcodes(A.obs_names)
        if not B.obs_names.equals(normalize_barcodes(B.obs_names)):
            B = B.copy(); B.obs_names = normalize_barcodes(B.obs_names)
    common = B.obs_names.intersection(A.obs_names)  # preserve REF order
    if len(common) == 0:
        raise ValueError("No common cells between adata and reference_adata.")
    return A[common].copy()


def adopt_reference_barcodes_by_coords(adata: ad.AnnData, ref: ad.AnnData) -> ad.AnnData:
    """
    Relabel adata.obs_names to reference barcodes via grid coords (array_row/array_col).
    Requires a 1-to-1 join.
    """
    if not ({'array_row','array_col'} <= set(adata.obs.columns)
            and {'array_row','array_col'} <= set(ref.obs.columns)):
        raise ValueError("array_row/array_col not found on both objects.")
    m1 = adata.obs.reset_index()[['index','array_row','array_col']]
    m2 = ref.obs.reset_index()[['index','array_row','array_col']].rename(columns={'index':'ref_barcode'})
    merged = pd.merge(m1, m2, on=['array_row','array_col'], how='inner')
    if len(merged) != adata.n_obs:
        raise ValueError("Grid coordinate join not 1–1; cannot relabel safely.")
    mapper = dict(zip(merged['index'], merged['ref_barcode']))
    out = adata.copy()
    out.obs_names = out.obs_names.to_series().map(mapper).values
    return out


def sanity_check_alignment(adata: ad.AnnData, ref: ad.AnnData) -> None:
    """Print barcode/coord alignment checks (debug helper)."""
    print("=== Barcode / Coordinate sanity check ===")
    same_barcodes = adata.obs_names.equals(ref.obs_names)
    print(f"Same barcodes & order: {same_barcodes}")
    if not same_barcodes:
        overlap = adata.obs_names.intersection(ref.obs_names)
        print(f"  Overlap: {len(overlap)} / {adata.n_obs} (adata), {ref.n_obs} (ref)")
    if {"array_row","array_col"} <= set(adata.obs.columns) and {"array_row","array_col"} <= set(ref.obs.columns):
        same_grid = np.array_equal(adata.obs[["array_row","array_col"]].values,
                                   ref.obs[["array_row","array_col"]].values)
        print(f"Same grid coords     : {same_grid}")
    if "spatial" in adata.obsm and "spatial" in ref.obsm:
        same_pixels = np.allclose(adata.obsm["spatial"], ref.obsm["spatial"], atol=1e-6)
        max_diff = np.abs(adata.obsm["spatial"] - ref.obsm["spatial"]).max()
        print(f"Same pixel coords    : {same_pixels}")
        print(f"Max abs pixel diff   : {max_diff:.6f}")


# ---------------------------- Graph helpers ---------------------------- #

@dataclass
class NicheParams:
    knn_distance_type: str = "euclidean"  # any scipy.spatial.distance.cdist metric
    k: int = 15
    using_mask: bool = False
    include_self: bool = False            # include the focal cell in weighting
    weight_mode: str = "inverse"          # 'inverse' | 'uniform' | 'gaussian'
    gaussian_sigma: float = 1.0           # used if weight_mode == 'gaussian'


def sparse_mx_to_torch_sparse_tensor(sparse_mx: sp.spmatrix) -> torch.Tensor:
    coo = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64))
    values = torch.from_numpy(coo.data)
    return torch.sparse_coo_tensor(indices, values, torch.Size(coo.shape))


def preprocess_graph(adj: sp.spmatrix) -> Tuple[sp.coo_matrix, torch.Tensor]:
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0], dtype=adj.dtype, format="coo")
    rowsum = np.array(adj_.sum(1)).flatten()
    rowsum[rowsum == 0.0] = 1.0
    d = sp.diags(np.power(rowsum, -0.5))
    adj_norm = adj_.dot(d).transpose().dot(d).tocoo()
    return adj_norm, sparse_mx_to_torch_sparse_tensor(adj_norm)


def edge_list_to_dict(edge_list: List[Tuple[int,int,float]], n: int) -> Dict[int, List[int]]:
    graphdict, seen = {}, set()
    for i, j, _ in edge_list:
        seen.add(i); seen.add(j)
        graphdict.setdefault(i, []).append(j)
    for i in range(n):
        graphdict.setdefault(i, [])
    return graphdict


def _neighbors_for_row(
    row_vec: np.ndarray, full_mat: np.ndarray, k: int, metric: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Return neighbor indices and distances for one row vs all rows."""
    d = distance.cdist(row_vec.reshape(1, -1), full_mat, metric)
    # argpartition for top-k+1 (skip self later)
    k_eff = min(k + 1, d.shape[1])
    inds = np.argpartition(d, kth=k_eff-1, axis=1)[:, :k_eff].ravel()
    d_subset = d[0, inds]
    # Now order those k by distance
    order = np.argsort(d_subset)
    return inds[order], d_subset[order]


def graph_computing(
    spatial_coords: np.ndarray, n_cells: int, params: NicheParams, logger: Optional[logging.Logger] = None
) -> List[Tuple[int, int, float]]:
    """kNN with adaptive boundary (mean+std) like the original UnitedNet code."""
    logger = logger or logging.getLogger("niche_unitednet")
    E: List[Tuple[int,int,float]] = []
    for i in range(n_cells):
        nbr_inds, nbr_dists = _neighbors_for_row(spatial_coords[i, :], spatial_coords, params.k, params.knn_distance_type)
        # remove self if present in top-k
        if nbr_inds[0] == i:
            nbr_inds, nbr_dists = nbr_inds[1:], nbr_dists[1:]
        else:
            nbr_inds, nbr_dists = nbr_inds[:params.k], nbr_dists[:params.k]
        boundary = np.mean(nbr_dists) + np.std(nbr_dists)
        for j, d in zip(nbr_inds, nbr_dists):
            if d <= boundary:
                E.append((i, int(j), 1.0))
    logger.info(f"Computed {len(E)} edges (k={params.k}, metric={params.knn_distance_type}).")
    return E


def graph_construction(
    spatial_coords: np.ndarray, n_cells: int, params: NicheParams, logger: Optional[logging.Logger] = None
) -> Tuple[Dict[str, object], sp.coo_matrix]:
    logger = logger or logging.getLogger("niche_unitednet")
    edges = graph_computing(spatial_coords, n_cells, params, logger)
    graphdict = edge_list_to_dict(edges, n_cells)
    G = nx.from_dict_of_lists(graphdict)
    adj_org = nx.adjacency_matrix(G).astype(np.float32)
    # strip diagonal
    adj_m1 = adj_org - sp.dia_matrix((adj_org.diagonal()[np.newaxis, :], [0]), shape=adj_org.shape)
    adj_m1.eliminate_zeros()
    adj_norm_write, adj_norm_m1 = preprocess_graph(adj_m1)
    # boolean adjacency incl. self for neighbor lookup
    adj_label = (adj_m1 + sp.eye(adj_m1.shape[0], format="csr")).toarray().astype(bool)
    graph_dict = {
        "adj_org": adj_org,
        "adj_norm": adj_norm_m1,
        "adj_label": adj_label,
    }
    if params.using_mask:
        graph_dict["adj_mask"] = torch.ones(n_cells, n_cells)
    return graph_dict, adj_norm_write


# ---------------------------- Niche computation ---------------------------- #

def _weights_from_dist(d: np.ndarray, mode: str, sigma: float) -> np.ndarray:
    """Compute neighbor weights from distances."""
    # d shape: (1, k) or (k,)
    d = np.asarray(d).reshape(-1)
    if mode == "uniform":
        w = np.ones_like(d)
    elif mode == "gaussian":
        # w = exp( - d^2 / (2 sigma^2) ), with d>=0
        # Clip sigma to avoid div-by-zero
        s = sigma if sigma > 1e-12 else 1.0
        w = np.exp(-(d ** 2) / (2 * (s ** 2)))
    else:  # "inverse"
        w = np.where(d > 0, 1.0 / d, 0.0)
    s = w.sum()
    return w / (s if s != 0 else 1.0)


def calculate_niche_modality_unitednet(
    adata: ad.AnnData,
    spatial_key: str = "spatial",
    params: Optional[NicheParams] = None,
    logger: Optional[logging.Logger] = None
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Compute niche expression using inverse-distance (default) weighted neighbor average.

    Returns
    -------
    niche_expr : np.ndarray [n_cells, n_genes]
    graph_dict : dict with 'adj_label' (bool array), 'adj_org', 'adj_norm'
    """
    logger = logger or logging.getLogger("niche_unitednet")
    params = params or NicheParams()
    if spatial_key not in adata.obsm:
        raise ValueError(f"Spatial key '{spatial_key}' not found in adata.obsm")
    spatial_coords = adata.obsm[spatial_key]
    n_cells, n_genes = adata.shape

    graph_dict, _ = graph_construction(spatial_coords, n_cells, params, logger)
    adj_label: np.ndarray = graph_dict["adj_label"]

    # X: dense array (n_cells, n_genes)
    if sp.issparse(adata.X):
        X = adata.X.tocsr()
        to_dense_slice = True
    else:
        X = np.asarray(adata.X, dtype=float)
        to_dense_slice = False

    niche_expr = np.zeros((n_cells, n_genes), dtype=float)

    for i in range(n_cells):
        nbr_idx = np.flatnonzero(adj_label[i])
        if not params.include_self:
            nbr_idx = nbr_idx[nbr_idx != i]
        if nbr_idx.size == 0:
            # fallback to self expression
            if to_dense_slice:
                niche_expr[i, :] = X[i, :].toarray().ravel()
            else:
                niche_expr[i, :] = X[i, :]
            continue

        dists = distance.cdist(
            spatial_coords[i, :].reshape(1, -1),
            spatial_coords[nbr_idx, :],
            metric=params.knn_distance_type
        )[0]  # shape (k,)

        w = _weights_from_dist(dists, params.weight_mode, params.gaussian_sigma).reshape(1, -1)

        if to_dense_slice:
            neigh = X[nbr_idx, :].toarray()  # [k, n_genes]
        else:
            neigh = X[nbr_idx, :]            # [k, n_genes]
        niche_expr[i, :] = (w @ neigh).ravel()

    return niche_expr, graph_dict


def create_niche_adata(original: ad.AnnData, niche_expr: np.ndarray) -> ad.AnnData:
    out = ad.AnnData(X=niche_expr)
    out.obs = original.obs.copy()
    out.var = original.var.copy()
    out.obs_names = original.obs_names
    out.var_names = original.var_names
    for k, v in original.obsm.items():
        out.obsm[k] = v.copy()
    out.uns = original.uns.copy()
    out.uns["modality"] = "niche"
    out.uns["niche_method"] = "UnitedNet"
    return out


# ---------------------------- High-level pipeline ---------------------------- #

def run_niche_pipeline(
    input_adata_path: Optional[str] = None,
    adata_in: Optional[ad.AnnData] = None,
    output_path: Optional[str] = None,
    pheno_csv: Optional[str] = None,
    pheno_barcode_col: str = "barcode",
    pheno_cols: Optional[List[str]] = None,
    spatial_key: str = "spatial",
    k: int = 15,
    distance_type: str = "euclidean",
    using_mask: bool = False,
    include_self: bool = False,
    weight_mode: str = "inverse",          # 'inverse' | 'uniform' | 'gaussian'
    gaussian_sigma: float = 1.0,
    # alignment options
    normalize_barcodes_flag: bool = True,
    reference_adata_path: Optional[str] = None,
    harmonize_pixels_to_reference: bool = True,
    relabel_barcodes_to_reference: bool = False,
    # misc
    overwrite: bool = False,
    log_level: str = "INFO",
) -> ad.AnnData:
    """
    Notebook- and CLI-friendly pipeline. Returns the niche AnnData.
    """
    logger = setup_logging(log_level)

    if adata_in is None and input_adata_path is None:
        raise ValueError("Provide either `input_adata_path` or `adata_in`.")

    if adata_in is not None:
        adata = adata_in.copy()
        logger.info(f"Using provided AnnData object: {adata.shape}")
    else:
        validate_file_exists(input_adata_path, "Input AnnData")
        adata = ad.read_h5ad(input_adata_path)
        logger.info(f"Loaded input AnnData: {adata.shape}")

    # Optional: normalize barcodes
    if normalize_barcodes_flag:
        nb = normalize_barcodes(adata.obs_names)
        if not adata.obs_names.equals(nb):
            adata = adata.copy()
            adata.obs_names = nb
            logger.info("Normalized barcodes.")

    # Optional: phenotypes
    adata = add_phenotype_data(adata, pheno_csv, pheno_barcode_col, pheno_cols, logger)

    # Optional: reference handling
    ref = None
    if reference_adata_path:
        validate_file_exists(reference_adata_path, "Reference AnnData")
        ref = ad.read_h5ad(reference_adata_path)
        if normalize_barcodes_flag:
            rb = normalize_barcodes(ref.obs_names)
            if not ref.obs_names.equals(rb):
                ref = ref.copy(); ref.obs_names = rb
        adata = subset_to_reference_cells(adata, ref, normalize_barcodes_flag=False)
        logger.info(f"Subset to reference: {adata.shape}")
        if harmonize_pixels_to_reference:
            adata = harmonize_spatial_scaling(target=ref, source=adata, logger=logger)
        if relabel_barcodes_to_reference:
            adata = adopt_reference_barcodes_by_coords(adata, ref)
            logger.info("Relabeled barcodes to reference.")

    # Compute niche
    params = NicheParams(
        knn_distance_type=distance_type,
        k=k,
        using_mask=using_mask,
        include_self=include_self,
        weight_mode=weight_mode,
        gaussian_sigma=gaussian_sigma,
    )
    niche_expr, graph_dict = calculate_niche_modality_unitednet(adata, spatial_key, params, logger)
    adata_niche = create_niche_adata(adata, niche_expr)

    # Save if requested
    if output_path:
        outp = Path(output_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        if outp.exists() and not overwrite:
            raise FileExistsError(f"Output exists: {outp}. Use overwrite=True.")
        adata_niche.write(outp)
        logger.info(f"Saved niche AnnData to: {outp}")

    if ref is not None:
        sanity_check_alignment(adata_niche, ref)

    return adata_niche


# ---------------------------- CLI ---------------------------- #

def _parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Compute UnitedNet-style niche modality from spatial coordinates.")
    p.add_argument("--input", type=str, help="Path to input .h5ad", required=True)
    p.add_argument("--output", type=str, help="Path to output .h5ad", required=True)
    p.add_argument("--spatial-key", type=str, default="spatial")
    p.add_argument("--k", type=int, default=15)
    p.add_argument("--distance", type=str, default="euclidean", help="Distance metric for kNN (scipy cdist)")
    p.add_argument("--include-self", action="store_true", help="Include focal cell in neighbor weighting")
    p.add_argument("--weight-mode", type=str, default="inverse", choices=["inverse", "uniform", "gaussian"])
    p.add_argument("--gaussian-sigma", type=float, default=1.0, help="Sigma for gaussian weights")
    p.add_argument("--pheno", type=str, default=None, help="Phenotype CSV (optional)")
    p.add_argument("--pheno-barcode-col", type=str, default="barcode")
    p.add_argument("--pheno-cols", type=str, nargs="*", default=None)
    p.add_argument("--reference", type=str, default=None, help="Reference .h5ad for subsetting/aligning (optional)")
    p.add_argument("--no-barcode-normalize", action="store_true", help="Do NOT normalize barcodes")
    p.add_argument("--no-pixel-harmonize", action="store_true", help="Do NOT harmonize pixel scale to reference")
    p.add_argument("--relabel-to-reference", action="store_true", help="Relabel barcodes to reference via array grid")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args()


def main():
    args = _parse_args()
    run_niche_pipeline(
        input_adata_path=args.input,
        output_path=args.output,
        spatial_key=args.spatial_key,
        k=args.k,
        distance_type=args.distance,
        include_self=args.include_self,
        weight_mode=args.weight_mode,
        gaussian_sigma=args.gaussian_sigma,
        pheno_csv=args.pheno,
        pheno_barcode_col=args.pheno_barcode_col,
        pheno_cols=args.pheno_cols,
        normalize_barcodes_flag=(not args.no_barcode_normalize),
        reference_adata_path=args.reference,
        harmonize_pixels_to_reference=(not args.no_pixel_harmonize),
        relabel_barcodes_to_reference=args.relabel_to_reference,
        overwrite=args.overwrite,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()