#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generic PerturbMap splitter (KP-style)

• Accepts ANY set of modalities (e.g. {"rna": ..., "niche": ..., "activity": ...} or {"flux": ...})
• Normalizes Visium barcodes (-1 suffix removal, uppercase)
• Intersects & aligns obs across ALL modalities (keeps reference order)
• Deterministic train/test split (optional stratify on a chosen modality/column)
• Saves train/test .h5ad files per modality with consistent naming
• Returns paths (and optionally the split AnnData objects)

Usage (Python):
---------------
from pm_splitter import split_any_modalities

paths = split_any_modalities(
    modalities={"rna":"../Data/processedData3/rna_modality.h5ad",
                "niche":"../Data/processedData3/niche_modality.h5ad",
                "activity":"../Data/processedData3/activity_modality.h5ad"},
    output_dir="../Data/UnitedNet/input_data",
    dataset_id="KP2_1",
    train_size=0.80,
    random_state=42,
    stratify_col=None,            # or e.g. "phenotypes"
    stratify_on="rna",            # which modality's .obs to use for stratify
)

# Flux-only? Just pass one modality:
flux_paths = split_any_modalities(
    modalities={"flux":"../Data/processedData3/flux_modality.h5ad"},
    output_dir="../Data/UnitedNet/input_data",
    dataset_id="KP2_1",
)
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Union, Optional, Tuple

import pandas as pd
import scanpy as sc
from sklearn.model_selection import train_test_split


AnnOrPath = Union[str, sc.AnnData]


# ----------------------- helpers ----------------------- #
def norm_barcodes(idx: pd.Index) -> pd.Index:
    """Normalize Visium-like barcodes: drop trailing '-1', uppercase."""
    return idx.astype(str).str.replace(r"-1$", "", regex=True).str.upper()


def _ensure_norm_obsnames(adata: sc.AnnData) -> sc.AnnData:
    nb = norm_barcodes(adata.obs_names)
    if not adata.obs_names.equals(nb):
        adata = adata.copy()
        adata.obs_names = nb
    return adata


def _maybe_stratify_labels(
    adata: sc.AnnData, obs_order: pd.Index, stratify_col: Optional[str]
) -> Optional[pd.Series]:
    if stratify_col is None:
        return None
    if stratify_col not in adata.obs.columns:
        print(f"⚠️  STRATIFY_COL='{stratify_col}' not found in selected modality; proceeding without stratify.")
        return None
    y = adata.obs.loc[obs_order, stratify_col].astype(str)
    vc = y.value_counts()
    ok = (vc >= 2).sum() >= 2
    if not ok:
        print("⚠️  Stratification skipped (insufficient class counts).")
        return None
    return y


def _save_h5ad(adata: sc.AnnData, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(path)
    print(f"Saved: {path}")


# ----------------------- main API ----------------------- #
def split_any_modalities(
    modalities: Dict[str, AnnOrPath],
    output_dir: str,
    dataset_id: str = "KP2_1",
    train_size: float = 0.80,
    random_state: int = 42,
    stratify_col: Optional[str] = None,     # e.g. "phenotypes"
    stratify_on: Optional[str] = None,      # which modality key to read stratify_col from
    normalize_barcodes_flag: bool = True,
    reference_order: Optional[str] = None,  # which modality sets obs order; default='rna' if present else first key
    return_adatas: bool = False,            # also return the split AnnData objects in-memory
) -> Dict[str, Union[Path, sc.AnnData]]:
    """
    Split ANY combination of modalities into train/test with aligned obs.

    Parameters
    ----------
    modalities : dict
        Mapping {name: path_or_adata}. 'name' becomes part of the output filename.
    output_dir : str
        Directory where split files are saved.
    dataset_id : str
        Suffix used in filenames (e.g., KP2_1).
    train_size : float
        Fraction for the train split.
    random_state : int
        RNG seed for reproducibility.
    stratify_col : Optional[str]
        Column name in .obs (of the modality indicated by 'stratify_on') for stratified splitting.
    stratify_on : Optional[str]
        Which modality key to use for picking stratify labels. Defaults to 'reference_order' modality.
    normalize_barcodes_flag : bool
        Apply Visium barcode normalization before intersecting (recommended).
    reference_order : Optional[str]
        Which modality determines obs order in outputs. Defaults to 'rna' if present else the first key.
    return_adatas : bool
        If True, returns the split AnnData objects alongside paths.

    Returns
    -------
    Dict[str, Path or AnnData] :
        Keys like "{name}_train", "{name}_test" -> saved Paths (and optionally AnnData if return_adatas=True).
    """
    if not modalities:
        raise ValueError("No modalities provided.")

    # Load / copy adatas
    loaded: Dict[str, sc.AnnData] = {}
    for name, src in modalities.items():
        if isinstance(src, sc.AnnData):
            adata = src.copy()
        elif isinstance(src, str):
            adata = sc.read_h5ad(src)
        else:
            raise TypeError(f"Unsupported type for modality '{name}': {type(src)}")
        loaded[name] = adata

    # Normalize barcodes
    if normalize_barcodes_flag:
        for k in list(loaded.keys()):
            loaded[k] = _ensure_norm_obsnames(loaded[k])

    # Choose reference for ordering
    if reference_order is None:
        reference_order = "rna" if "rna" in loaded else list(loaded.keys())[0]
    if reference_order not in loaded:
        raise ValueError(f"reference_order='{reference_order}' not among modalities: {list(loaded.keys())}")

    # Intersect obs across all modalities; keep reference order
    ref_obs = loaded[reference_order].obs_names
    common = ref_obs
    for k, a in loaded.items():
        if k == reference_order:
            continue
        common = common.intersection(a.obs_names)
    if len(common) == 0:
        raise ValueError("No common spots across provided modalities after normalization.")

    # Reindex each modality to the common obs (reference order)
    for k in list(loaded.keys()):
        loaded[k] = loaded[k][common].copy()
        loaded[k] = loaded[k][ref_obs.intersection(common)].copy()  # ensure same absolute order as reference

    # Build stratify labels (from stratify_on or reference)
    base_for_stratify = stratify_on or reference_order
    y = _maybe_stratify_labels(loaded[base_for_stratify], loaded[reference_order].obs_names, stratify_col)

    # Deterministic split by obs names
    obs_master = pd.Index(loaded[reference_order].obs_names)
    tr_idx, te_idx = train_test_split(
        obs_master,
        train_size=train_size,
        random_state=random_state,
        stratify=y if y is not None else None
    )
    tr_idx, te_idx = list(tr_idx), list(te_idx)
    print(f"Train: {len(tr_idx)}, Test: {len(te_idx)}")

    # Slice and save
    outdir = Path(output_dir); outdir.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, Union[Path, sc.AnnData]] = {}

    for name, adata in loaded.items():
        train_adata = adata[tr_idx].copy()
        test_adata  = adata[te_idx].copy()

        p_train = outdir / f"adata_{name}_train_perturbmap_{dataset_id}.h5ad"
        p_test  = outdir / f"adata_{name}_test_perturbmap_{dataset_id}.h5ad"

        _save_h5ad(train_adata, p_train)
        _save_h5ad(test_adata,  p_test)

        outputs[f"{name}_train"] = p_train
        outputs[f"{name}_test"]  = p_test

        if return_adatas:
            outputs[f"{name}_train_adata"] = train_adata
            outputs[f"{name}_test_adata"]  = test_adata

    print("Splitting completed.")
    print(f"Files saved to: {outdir}")
    return outputs


# ----------------------- CLI (optional) ----------------------- #
def _parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Split ANY set of modalities (aligned obs, deterministic split).")
    p.add_argument("--mod", action="append", required=True,
                   help="Modality as name=path, e.g., --mod rna=../rna.h5ad --mod niche=../niche.h5ad")
    p.add_argument("--outdir", required=True, help="Output directory")
    p.add_argument("--dataset-id", default="KP2_1")
    p.add_argument("--train-size", type=float, default=0.80)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stratify-col", default=None)
    p.add_argument("--stratify-on", default=None)
    p.add_argument("--no-norm-barcodes", action="store_true")
    p.add_argument("--reference-order", default=None, help="Which modality key sets obs order (default: rna or first)")
    return p.parse_args()


def _main():
    args = _parse_args()

    # parse name=path pairs
    mods: Dict[str, str] = {}
    for m in args.mod:
        if "=" not in m:
            raise ValueError(f"--mod must be name=path, got: {m}")
        name, path = m.split("=", 1)
        mods[name.strip()] = path.strip()

    split_any_modalities(
        modalities=mods,
        output_dir=args.outdir,
        dataset_id=args.dataset_id,
        train_size=args.train_size,
        random_state=args.seed,
        stratify_col=args.stratify_col,
        stratify_on=args.stratify_on,
        normalize_barcodes_flag=(not args.no_norm_barcodes),
        reference_order=args.reference_order,
        return_adatas=False,
    )


if __name__ == "__main__":
    _main()
