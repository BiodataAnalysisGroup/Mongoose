# JUPYTER-FRIENDLY GENKI RUNNER
# Drop-in cell for notebooks

import os
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any, Union

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse

import scanpy as sc
sc.settings.verbosity = 0

# GenKI imports (keep names consistent with your package)
import GenKI as gk
from GenKI.preprocesing import build_adata
from GenKI.dataLoader import DataLoader
from GenKI.train import VGAE_trainer
from GenKI import utils

def run_genki_notebook(
    adata_or_path: Union[str, "anndata.AnnData"],
    genes_of_interest: List[str],
    *,
    # Files & output
    npz_filename: str = "PertMap_Auto",
    base_output_dir: Union[str, Path] = "../GenKI",
    write_to_disk: bool = True,
    save_dense_adjacency: bool = False,          # keep False for memory safety
    num_top_genes_for_summary: int = 50,
    # DataLoader / GRN
    grn_rebuild: bool = True,
    n_cpus: int = 20,
    obs_label: str = "ident",
    grn_dir: Union[str, Path] = "GRNs",
    # VGAE
    epochs: int = 300,
    learning_rate: float = 5e-2,
    beta: float = 5e-4,
    seed: int = 8096,
    # Significance testing
    run_significance_test: bool = True,
    n_permutations: int = 100,
    alpha: float = 0.05,
    # GRN export
    extract_adjacency: bool = True,
    edge_percentile: float = 85.0,               # top X% |weight|
    # Misc
    uppercase_genes_in_adata: bool = False,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Notebook-friendly GenKI runner that returns results and (optionally) writes files.
    """
    # -------- Paths & timestamp --------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    genes_str = "_".join(genes_of_interest[:3]) + (f"_plus{len(genes_of_interest)-3}more" if len(genes_of_interest) > 3 else "")
    base_output_dir = Path(base_output_dir)
    output_dir = base_output_dir / f"GenKI_NB_{genes_str}_{timestamp}"
    if write_to_disk:
        output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[GenKI] Starting notebook run -> {output_dir}")

    # -------- Load AnnData --------
    if isinstance(adata_or_path, str):
        if verbose:
            print(f"[GenKI] Loading AnnData from: {adata_or_path}")
        adata = build_adata(adata_or_path, uppercase=uppercase_genes_in_adata)
    else:
        adata = adata_or_path
        if uppercase_genes_in_adata:
            adata.var_names = adata.var_names.str.upper()
    if verbose:
        print(f"[GenKI] AnnData: {adata.n_vars} genes × {adata.n_obs} cells")

    results: Dict[str, Any] = {
        "timestamp": timestamp,
        "output_directory": str(output_dir),
        "genes_analyzed": genes_of_interest,
        "parameters": {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "beta": beta,
            "seed": seed,
            "n_permutations": n_permutations,
            "alpha": alpha,
            "n_cpus": n_cpus,
            "obs_label": obs_label,
            "grn_rebuild": grn_rebuild,
        },
        "individual": {},
        "all_significant_by_ko": {},     # KO_gene -> DataFrame of significant genes
        "combined_significant": None,    # DataFrame (long format)
        "adjacency": {
            "edge_list": None,           # DataFrame
            "weighted_adjacency_path": None,
            "edge_list_path": None,
            "threshold_used": None,
            "total_edges": None,
        }
    }

    # -------- Iterate over KOs --------
    for idx, gene in enumerate(genes_of_interest, 1):
        if verbose:
            print(f"\n[GenKI] ({idx}/{len(genes_of_interest)}) KO: {gene}")

        gene_payload: Dict[str, Any] = {}
        try:
            # Data wrapper
            data_wrapper = DataLoader(
                adata=adata,
                target_gene=[gene],
                target_cell=None,
                obs_label=obs_label,
                GRN_file_dir=str(grn_dir),
                rebuild_GRN=grn_rebuild,
                pcNet_name=npz_filename,
                verbose=False,
                n_cpus=n_cpus,
            )
            data_wt = data_wrapper.load_data()
            data_ko = data_wrapper.load_kodata()
            if verbose:
                print("  • Data prepared")

            # Train VGAE
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
            if verbose:
                print("  • VGAE trained")

            # Latents + KL
            z_mu_wt, z_std_wt = sensei.get_latent_vars(data_wt)
            z_mu_ko, z_std_ko = sensei.get_latent_vars(data_ko)
            dis = gk.utils.get_distance(z_mu_ko, z_std_ko, z_mu_wt, z_std_wt, by="KL")
            res_ranked = utils.get_generank(data_wt, dis, rank=True)
            gene_payload["total_ranked"] = len(res_ranked)

            # Significance (optional)
            if run_significance_test:
                if verbose:
                    print(f"  • Running permutation test (n={n_permutations})")
                null_dist = sensei.pmt(data_ko, n=n_permutations, by="KL")
                res_sig = utils.get_generank(data_wt, dis, null_dist)

                # pick p-adj or pvalue if present
                if "padj" in res_sig.columns:
                    significant = res_sig[res_sig["padj"] < alpha].copy()
                elif "pvalue" in res_sig.columns:
                    significant = res_sig[res_sig["pvalue"] < alpha].copy()
                else:
                    if verbose:
                        print("  ! No p-value columns found; using full permutation result")
                    significant = res_sig.copy()

                gene_payload["significance_table"] = res_sig
                gene_payload["significant"] = significant
                gene_payload["n_significant"] = len(significant)

                results["all_significant_by_ko"][gene] = significant

                # Optional saves per KO
                if write_to_disk:
                    ko_dir = output_dir / f"{gene}_results"
                    ko_dir.mkdir(exist_ok=True)
                    significant.to_csv(ko_dir / f"ALL_Significant_Genes_{gene}_alpha{alpha}.csv")
                    res_sig.to_csv(ko_dir / f"Complete_Significance_Results_{gene}.csv")
                    res_ranked.head(num_top_genes_for_summary).to_csv(
                        ko_dir / f"Top{num_top_genes_for_summary}_Ranked_Genes_{gene}.csv"
                    )
            else:
                # No permutations -> just provide top summary
                topn = res_ranked.head(num_top_genes_for_summary).copy()
                gene_payload["significant"] = topn
                gene_payload["n_significant"] = len(topn)
                results["all_significant_by_ko"][gene] = topn
                if write_to_disk:
                    ko_dir = output_dir / f"{gene}_results"
                    ko_dir.mkdir(exist_ok=True)
                    topn.to_csv(ko_dir / f"Top{num_top_genes_for_summary}_Ranked_Genes_{gene}.csv")

            if verbose:
                print(f"  • Done. Significant: {gene_payload['n_significant']}")

        except Exception as e:
            gene_payload["error"] = str(e)
            if verbose:
                print(f"  ✗ Error on {gene}: {e}")

        results["individual"][gene] = gene_payload

    # -------- GRN extraction (optional) --------
    if extract_adjacency:
        try:
            pc_fp = Path(f"{npz_filename}.npz")
            if pc_fp.exists():
                if verbose:
                    print("\n[GenKI] Extracting GRN adjacency…")
                loader = np.load(pc_fp, allow_pickle=True)
                data, indices, indptr = loader["data"], loader["indices"], loader["indptr"]
                shape = tuple(loader["shape"])
                W = csr_matrix((data, indices, indptr), shape=shape)

                thr = np.percentile(np.abs(W.data), edge_percentile)
                W_thr = W.copy()
                mask = np.abs(W_thr.data) < thr
                W_thr.data[mask] = 0
                W_thr.eliminate_zeros()

                genes = adata.var_names.to_list()
                rows, cols = W_thr.nonzero()
                weights = W_thr[rows, cols].A1

                edge_list = pd.DataFrame(
                    {"source": [genes[i] for i in rows],
                     "target": [genes[j] for j in cols],
                     "weight": weights}
                )

                results["adjacency"]["edge_list"] = edge_list
                results["adjacency"]["threshold_used"] = float(thr)
                results["adjacency"]["total_edges"] = int(W_thr.nnz)

                if write_to_disk:
                    adj_dir = output_dir / "adjacency_matrices"
                    adj_dir.mkdir(exist_ok=True)
                    edge_path = adj_dir / f"{npz_filename}_edge_list.csv"
                    edge_list.to_csv(edge_path, index=False)
                    results["adjacency"]["edge_list_path"] = str(edge_path)

                    if save_dense_adjacency:
                        # WARNING: this can be huge — only enable if you know it's safe
                        dense_df = pd.DataFrame(W_thr.toarray(), index=genes, columns=genes)
                        dense_path = adj_dir / f"{npz_filename}_weighted_adjacency.csv"
                        dense_df.to_csv(dense_path)
                        results["adjacency"]["weighted_adjacency_path"] = str(dense_path)

                if verbose:
                    print(f"[GenKI] GRN edges: {W_thr.nnz} (top {100-edge_percentile:.0f}% kept)")
            else:
                if verbose:
                    print(f"[GenKI] Skipping GRN export: {pc_fp} not found.")
        except Exception as e:
            if verbose:
                print(f"[GenKI] ✗ GRN extraction failed: {e}")

    # -------- Combine significant across KOs --------
    combined = []
    for ko, df in results["all_significant_by_ko"].items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            c = df.copy()
            c["KO_gene"] = ko
            combined.append(c)
    if combined:
        combined_df = pd.concat(combined, ignore_index=False)
        results["combined_significant"] = combined_df
        if write_to_disk:
            combined_path = Path(results["output_directory"]) / "ALL_Significant_Genes_Combined.csv"
            combined_df.to_csv(combined_path)
        if verbose:
            print("\n[GenKI] Combined significant saved.")
            print(f"  • Pairs: {len(combined_df)} | Unique genes: {combined_df.index.nunique()}")

    # -------- Lightweight text summary --------
    if write_to_disk:
        summary_path = Path(results["output_directory"]) / "GenKI_Notebook_Summary.txt"
        with open(summary_path, "w") as f:
            f.write("GenKI Notebook Run Summary\n")
            f.write("="*50 + "\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Genes: {', '.join(genes_of_interest)}\n")
            f.write(f"Permutations: {n_permutations if run_significance_test else 0}\n")
            f.write(f"Alpha: {alpha}\n")
            for g, payload in results["individual"].items():
                n = payload.get("n_significant", "NA")
                f.write(f"- {g}: {n} significant\n")

    if verbose:
        print("\n[GenKI] Completed. Results in:", results["output_directory"])
    return results
