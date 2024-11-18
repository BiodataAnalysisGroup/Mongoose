import pandas as pd

# Load the data from CSV file
file_path = 'feature_feature_importance.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Define the list of genes of interest
updated_genes_list = [gene_of_interest] + genki_genes

# Filter data for the genes of interest
filtered_data = data[data['Source'].isin(updated_genes_list)]

# Separate data for RNA -> Niche and RNA -> Protein directions
rna_niche_data = filtered_data[filtered_data['Direction'] == 'RNA -> Niche']
rna_protein_data = filtered_data[filtered_data['Direction'] == 'RNA -> Protein']

# Find the top 3 unique RNA -> Niche and RNA -> Protein interactions with the highest 'Value'
top_rna_niche = rna_niche_data.sort_values(by='Value', ascending=False).drop_duplicates(subset=['Source', 'Target']).groupby('Source').head(3)
top_rna_protein = rna_protein_data.sort_values(by='Value', ascending=False).drop_duplicates(subset=['Source', 'Target']).groupby('Source').head(3)

# Identify cases where the 'Target' value is the same as the 'Source' value
rna_protein_duplicates = top_rna_protein[top_rna_protein['Target'] == top_rna_protein['Source']]
rna_niche_duplicates = top_rna_niche[top_rna_niche['Target'] == top_rna_niche['Source']]

# Remove duplicate entries from the original top lists
top_rna_protein_cleaned = top_rna_protein[~(top_rna_protein['Target'] == top_rna_protein['Source'])]
top_rna_niche_cleaned = top_rna_niche[~(top_rna_niche['Target'] == top_rna_niche['Source'])]

# Find the next highest entries for those with matching 'Target' and 'Source'
next_rna_protein = rna_protein_data[~rna_protein_data.isin(rna_protein_duplicates)].sort_values(by='Value', ascending=False)
next_rna_protein_add = next_rna_protein.groupby('Source').apply(lambda x: x[~x['Target'].isin(top_rna_protein_cleaned['Target'])].head(1)).reset_index(drop=True)

next_rna_niche = rna_niche_data[~rna_niche_data.isin(rna_niche_duplicates)].sort_values(by='Value', ascending=False)
next_rna_niche_add = next_rna_niche.groupby('Source').apply(lambda x: x[~x['Target'].isin(top_rna_niche_cleaned['Target'])].head(1)).reset_index(drop=True)

# Combine the original top lists with the added entries
final_top_rna_protein = pd.concat([top_rna_protein_cleaned, next_rna_protein_add]).sort_values(by=['Source', 'Value'], ascending=[True, False]).groupby('Source').head(3)
final_top_rna_niche = pd.concat([top_rna_niche_cleaned, next_rna_niche_add]).sort_values(by=['Source', 'Value'], ascending=[True, False]).groupby('Source').head(3)

# Combine all results into a single DataFrame
combined_df = pd.concat([final_top_rna_protein, final_top_rna_niche])

# Save the results to a single sheet in an Excel file
output_path = f'./Top_RNA_Niche_Protein_Interactions_Single_Sheet_{gene_of_interest}.xlsx'  # Replace with your desired output path

with pd.ExcelWriter(output_path) as writer:
    combined_df.to_excel(writer, sheet_name='Top Interactions', index=False)

print(f"Results have been saved to {output_path}")