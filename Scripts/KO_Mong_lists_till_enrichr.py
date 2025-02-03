#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 15:09:39 2024

@author: aspaor
"""


import os
import pandas as pd
import json
import enrichrpy.enrichr as een


#os.chdir('/Users/aspaor/Downloads/Mongoose_test/')
#print(f"Current working directory: {os.getcwd()}")

# Step 1: Load the initial data
initial_data_path = 'Top_RNA_Niche_Protein_Interactions_Sheet.xlsx'
data = pd.read_excel(initial_data_path)

# Step 2: Extract unique genes from the KO_Gene column
unique_genes = data['KO_Gene'].unique()

# Step 3: Create a dictionary to store Target and Source values for each KO_Gene
gene_target_mapping = {}
for gene in unique_genes:
    gene_target_mapping[gene] = data.loc[data['KO_Gene'] == gene, 'Target'].tolist()
    gene_target_mapping[gene].extend(data.loc[data['KO_Gene'] == gene, 'Source'].tolist())

# Step 4: Remove duplicates from each KO_Gene's list
unique_gene_target_mapping = {gene: list(set(values)) for gene, values in gene_target_mapping.items()}

# Step 5: Load the mapping file for renaming
mapping_file_path = 'DBiTseq_protein_mapping.xlsx'
mapping_df = pd.read_excel(mapping_file_path)

# Step 6: Create a dictionary to map markers to gene names
marker_to_gene_name = dict(zip(mapping_df['Marker'], mapping_df['Gene Symbol']))

# Step 7: Function to rename genes if they exist in the Marker column
def rename_target_gene(gene):
    return marker_to_gene_name.get(gene, gene)

# Step 8: Rename all values for each KO_Gene
renamed_gene_target_mapping = {
    ko_gene: [rename_target_gene(value) for value in values]
    for ko_gene, values in unique_gene_target_mapping.items()
}

# Step 9: Process all CSV files in a folder and update the dictionary
csv_folder = './'
for file_name in os.listdir(csv_folder):
    if file_name.endswith('.csv'):
        # Extract KO_Gene name from the file name
        ko_gene_name = file_name.replace("Top10_Responsive_Genes_", "").replace(".csv", "")
        
        # Load the CSV file and extract responsive genes
        responsive_genes_path = os.path.join(csv_folder, file_name)
        responsive_genes_df = pd.read_csv(responsive_genes_path, index_col=0)
        responsive_genes_list = responsive_genes_df.index.tolist()
        
        # Add responsive genes to the corresponding KO_Gene in the dictionary
        if ko_gene_name in renamed_gene_target_mapping:
            renamed_gene_target_mapping[ko_gene_name].extend(responsive_genes_list)
            # Remove duplicates
            renamed_gene_target_mapping[ko_gene_name] = list(set(renamed_gene_target_mapping[ko_gene_name]))

# Step 10: Save the updated dictionary to a JSON file
updated_dictionary_path = 'updated_renamed_gene_target_mapping.json'
with open(updated_dictionary_path, 'w') as file:
    json.dump(renamed_gene_target_mapping, file, indent=4)

print(f"Updated dictionary saved to {updated_dictionary_path}")

# Step 11: Prepare output files for enrichment analysis
output_folder = './enrichment_results/'
os.makedirs(output_folder, exist_ok=True)

output_files = {
    'GO_Biological_Process_2023': os.path.join(output_folder, "GO_Biological_Process_2023.xlsx"),
    'Reactome_2022': os.path.join(output_folder, "Reactome_2022.xlsx"),
    'WikiPathway_2023_Human': os.path.join(output_folder, "WikiPathway_2023_Human.xlsx"),
    'KEGG_2021_Human': os.path.join(output_folder, "KEGG_2021_Human.xlsx")
}

gene_set_libraries = list(output_files.keys())

# Step 12: Create a Pandas ExcelWriter for each output file
writers = {library: pd.ExcelWriter(output_files[library], engine='xlsxwriter') for library in gene_set_libraries}

# Step 13: Perform enrichment analysis for each KO_Gene and write results to respective files
for ko_gene, gene_list in renamed_gene_target_mapping.items():
    print(f"Performing enrichment analysis for KO_Gene: {ko_gene}")
    for library in gene_set_libraries:
        try:
            # Enrichment analysis
            enrichment_results = een.get_pathway_enrichment(gene_list, gene_set_library=library)
            
            # Save results to the respective sheet in the output file
            enrichment_results.to_excel(writers[library], sheet_name=ko_gene, index=False)
            print(f"Enrichment results for {ko_gene} saved to {library}.")
        except Exception as e:
            print(f"Error processing {ko_gene} for {library}: {e}")

# Step 14: Save and close all Excel writers
for library, writer in writers.items():
    writer.close()

print(f"Enrichment results saved in {output_folder}")





















###############
import pandas as pd
#import json

# Load the initial data
initial_data_path = './Top_RNA_Niche_Protein_Interactions_first50_Sheet.xlsx'
data = pd.read_excel(initial_data_path)

# Extract unique genes from the KO_Gene column
unique_genes = data['KO_Gene'].unique()

# Create a dictionary to store Target values for each KO_Gene
gene_target_mapping = {}
for gene in unique_genes:
    gene_target_mapping[gene] = data.loc[data['KO_Gene'] == gene, 'Target'].tolist()
    gene_target_mapping[gene].extend(data.loc[data['KO_Gene'] == gene, 'Source'].tolist())

# Remove duplicates from each KO_Gene's list
unique_gene_target_mapping = {gene: list(set(values)) for gene, values in gene_target_mapping.items()}

# Load the mapping file for renaming
mapping_file_path = './DBiTseq_protein_mapping.xlsx'
mapping_df = pd.read_excel(mapping_file_path)

# Create a dictionary to map markers to gene names
marker_to_gene_name = dict(zip(mapping_df['Marker'], mapping_df['Gene Symbol']))

# Function to rename gene if it exists in the Marker column
def rename_target_gene(gene):
    return marker_to_gene_name.get(gene, gene)

# Rename all values for each KO_Gene
renamed_gene_target_mapping = {
    ko_gene: [rename_target_gene(value) for value in values]
    for ko_gene, values in unique_gene_target_mapping.items()
}

print(renamed_gene_target_mapping)
# # Save the final dictionary to a JSON file
# output_file_path = '/path/to/renamed_gene_target_mapping.json'
# with open(output_file_path, 'w') as file:
#     json.dump(renamed_gene_target_mapping, file, indent=4)

# print(f"Final dictionary saved to {output_file_path}")