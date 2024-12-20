# import libraries
import os
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import re

sc.settings.verbosity = 0

import GenKI as gk
from GenKI.preprocesing import build_adata
from GenKI.dataLoader import DataLoader
from GenKI.train import VGAE_trainer
from GenKI import utils

import stringdb
import networkx as nx

from scipy.sparse import csr_matrix
from scipy.sparse import issparse

## Import file

adata = sc.read("DBiTseq_UnitedNet.h5ad")

## Data Normalization

# adata pre-processing to prepare for input in the GenKI tool
adata.layers["norm"] = adata.X.copy()

# The adata.X should be normalised-scaled AND in sparse matrix format!
if not issparse(adata.X):
    sparse_matrix = csr_matrix(adata.X)
    adata.X = sparse_matrix
    print("Converted adata.X to a sparse matrix.")
else:
    print("adata.X is already a sparse matrix.")

## Pre-processing lncRNAs exclude

# Get the list of gene names
gene_names = adata.var.index.tolist()

# Define a regex pattern to match gene names that end with 'Rik'
pattern = r'.*Rik$'

# Identify genes to filter out
genes_to_filter = [gene for gene in gene_names if re.match(pattern, gene)]

print(f"Number of genes to filter out: {len(genes_to_filter)}")
print("Genes to filter out:")
print(genes_to_filter)

# Now filter the adata object to exclude these genes
adata_filtered = adata[:, ~adata.var.index.isin(genes_to_filter)].copy()

print(f"Number of genes after filtering: {adata_filtered.n_vars}")

# Get the list of gene names
genes_of_interest = adata_filtered.var.index[:567].tolist()

# Define a regex pattern to match gene names that end with 'Rik'
pattern = r'.*Rik$'

# Identify genes to filter out
genes_to_filter = [gene for gene in gene_names if re.match(pattern, gene)]

print(f"Number of genes to filter out: {len(genes_to_filter)}")
print("Genes to filter out:")
print(genes_to_filter)

# Now filter the adata object to exclude these genes
adata_filtered = adata[:, ~adata.var.index.isin(genes_to_filter)].copy()

print(f"Number of genes after filtering: {adata_filtered.n_vars}")

## Process

# Create a directory to save the top interactions if it doesn't exist
os.makedirs('Top_Interactions', exist_ok=True)

# Get the first 3 genes in adata_filtered.var.index
genes_of_interest = adata_filtered.var.index[:561].tolist()
print(f"Genes to knock out: {genes_of_interest}")

# Select a dummy gene not in genes_of_interest
all_genes = list(adata_filtered.var.index)
#dummy_gene = next(g for g in all_genes if g not in genes_of_interest)
#print(f"Using dummy gene for WT data: {dummy_gene}")


# # Initialize DataLoader for WT data
# data_wrapper_wt = DataLoader(
#     adata_filtered,
#     target_gene=[dummy_gene],  # Use the dummy gene
#     target_cell=None,
#     obs_label="cell_type",  # Adjust if necessary
#     GRN_file_dir="GRNs",
#     rebuild_GRN=True,  # Build the GRN
#     pcNet_name="DBiTseq_example",
#     verbose=False,  # Set to False to reduce output
#     n_cpus=48,
# )

# # Load WT data (GRN will be built here)
# data_wt = data_wrapper_wt.load_data()

# # Set hyperparameters
# hyperparams = {
#     "epochs": 300,
#     "lr": 5e-2,
#     "beta": 5e-4,
#     "seed": 8096,
# }

# # Train the model on WT data
# sensei = VGAE_trainer(
#     data_wt,
#     epochs=hyperparams["epochs"],
#     lr=hyperparams["lr"],
#     beta=hyperparams["beta"],
#     seed=hyperparams["seed"],
#     verbose=False,  # Set to False to reduce output
# )

# sensei.train()

# # Get latent variables for WT data
# z_mu_wt, z_std_wt = sensei.get_latent_vars(data_wt)

# # Initialize a list to store combined results
combined_results = []

for gene_of_interest in all_genes:
    print(f"\nProcessing gene: {gene_of_interest}")

    # Initialize DataLoader for KO data
    data_wrapper = DataLoader(
        adata_filtered,
        target_gene=[gene_of_interest],
        target_cell=None,
        obs_label="cell_type",
        GRN_file_dir="GRNs",
        rebuild_GRN=False,  # Use the existing GRN
        pcNet_name="DBiTseq_example",
        verbose=False,  # Set to False to reduce output
        n_cpus=48,
    )

    # Load KO data for the gene
    data_wt = data_wrapper.load_data()
    data_ko = data_wrapper.load_kodata()


    # Set hyperparameters
    hyperparams = {
        "epochs": 300,
        "lr": 5e-2,
        "beta": 5e-4,
        "seed": 8096,
    }


        # Train the model on WT data
    sensei = VGAE_trainer(
        data_wt,
        epochs=hyperparams["epochs"],
        lr=hyperparams["lr"],
        beta=hyperparams["beta"],
        seed=hyperparams["seed"],
        verbose=False,  # Set to False to reduce output
    )

    sensei.train()

    # Get latent variables for KO data
    z_mu_ko, z_std_ko = sensei.get_latent_vars(data_ko)
    z_mu_wt, z_std_wt = sensei.get_latent_vars(data_wt)

    # Calculate the distance between WT and KO data
    dis = gk.utils.get_distance(z_mu_ko, z_std_ko, z_mu_wt, z_std_wt, by="KL")

    # Get the ranked list of responsive genes
    res_raw = utils.get_generank(data_wt, dis, rank=True)

    # Store the top 10 responsive genes
    top_genes = res_raw.head(10)
    print(f"Top 10 KO Responsive Genes for {gene_of_interest}:\n{top_genes}")

    os.makedirs('Top10_Responsive_Genes', exist_ok=True)
    top_genes_path = f'Top10_Responsive_Genes/Top10_Responsive_Genes_{gene_of_interest}.csv'
    top_genes.to_csv(top_genes_path)

    # Extract the top genes for the current KO gene
    genki_list = top_genes.index.tolist()

    # Load the data from CSV file
    file_path = 'feature_feature_importance.csv'  # Replace with your file path
    data = pd.read_csv(file_path)

    # Filter data for the genes of interest
    filtered_data = data[data['Source'].isin(genki_list)]

    # Separate data for RNA -> Niche and RNA -> Protein directions
    rna_niche_data = filtered_data[filtered_data['Direction'] == 'RNA -> Niche']
    rna_protein_data = filtered_data[filtered_data['Direction'] == 'RNA -> Protein']

    # Find the top 3 unique RNA -> Niche and RNA -> Protein interactions with the highest 'Value'
    top_rna_niche = (
        rna_niche_data
        .sort_values(by='Value', ascending=False)
        .drop_duplicates(subset=['Source', 'Target'])
        .groupby('Source')
        .head(3)
    )

    top_rna_protein = (
        rna_protein_data
        .sort_values(by='Value', ascending=False)
        .drop_duplicates(subset=['Source', 'Target'])
        .groupby('Source')
        .head(3)
    )

    # Identify cases where the 'Target' value is the same as the 'Source' value
    rna_protein_duplicates = top_rna_protein[top_rna_protein['Target'] == top_rna_protein['Source']]
    rna_niche_duplicates = top_rna_niche[top_rna_niche['Target'] == top_rna_niche['Source']]

    # Remove duplicate entries from the original top lists
    top_rna_protein_cleaned = top_rna_protein[~(top_rna_protein['Target'] == top_rna_protein['Source'])]
    top_rna_niche_cleaned = top_rna_niche[~(top_rna_niche['Target'] == top_rna_niche['Source'])]

    # Find the next highest entries for those with matching 'Target' and 'Source'
    next_rna_protein = (
        rna_protein_data
        .loc[~rna_protein_data.index.isin(rna_protein_duplicates.index)]
        .sort_values(by='Value', ascending=False)
    )

    next_rna_protein_add = (
        next_rna_protein
        .groupby('Source')
        .apply(lambda x: x[~x['Target'].isin(top_rna_protein_cleaned['Target'])].head(1))
        .reset_index(drop=True)
    )

    next_rna_niche = (
        rna_niche_data
        .loc[~rna_niche_data.index.isin(rna_niche_duplicates.index)]
        .sort_values(by='Value', ascending=False)
    )

    next_rna_niche_add = (
        next_rna_niche
        .groupby('Source')
        .apply(lambda x: x[~x['Target'].isin(top_rna_niche_cleaned['Target'])].head(1))
        .reset_index(drop=True)
    )

    # Combine the original top lists with the added entries
    final_top_rna_protein = (
        pd.concat([top_rna_protein_cleaned, next_rna_protein_add])
        .sort_values(by=['Source', 'Value'], ascending=[True, False])
        .groupby('Source')
        .head(3)
    )

    final_top_rna_niche = (
        pd.concat([top_rna_niche_cleaned, next_rna_niche_add])
        .sort_values(by=['Source', 'Value'], ascending=[True, False])
        .groupby('Source')
        .head(3)
    )

    # Combine all results into a single DataFrame
    combined_df = pd.concat([final_top_rna_protein, final_top_rna_niche])

    # Add a column to indicate the KO gene
    combined_df['KO_Gene'] = gene_of_interest

    # Append to the list
    combined_results.append(combined_df)

# Concatenate all results
all_combined_df = pd.concat(combined_results)

# Save the combined results to a single sheet in an Excel file
output_path = 'Top_RNA_Niche_Protein_Interactions_first50_Sheet.xlsx'

with pd.ExcelWriter(output_path) as writer:
    all_combined_df.to_excel(writer, sheet_name='Top Interactions', index=False)

print(f"\nResults have been saved to {output_path}")

## Rest Process

interactions_df = pd.read_excel('Top_RNA_Niche_Protein_Interactions_first50_Sheet.xlsx', sheet_name='Top Interactions')
targets = interactions_df['Target'].tolist()
print(f"Extracted {len(targets)} targets from the Excel file.")

genes_of_interest = adata_filtered.var.index[:3].tolist()
print(f"Genes of interest: {genes_of_interest}")

all_responsive_genes = []

for gene_of_interest in genes_of_interest:
    file_name = f'Top10_Responsive_Genes/Top10_Responsive_Genes_{gene_of_interest}.csv'
    if os.path.exists(file_name):
        df_top_genes = pd.read_csv(file_name, index_col=0)
        responsive_genes = df_top_genes.index.tolist()
        all_responsive_genes.extend(responsive_genes)
        print(f"Added {len(responsive_genes)} responsive genes from {file_name}.")
    else:
        print(f"File {file_name} does not exist.")

combined_list = targets + all_responsive_genes
print(f"Total combined genes before removing duplicates: {len(combined_list)}.")

# Remove duplicates
combined_list = list(set(combined_list))
print(f"Total combined genes after removing duplicates: {len(combined_list)}.")

# Save it to a file
output_file = 'Combined_Genes_List.txt'
with open(output_file, 'w') as f:
    for gene in combined_list:
        f.write(f"{gene}\n")
print(f"Combined list saved to {output_file}.")
