import pandas as pd
import json

# Load the dictionary from the JSON file
with open('updated_renamed_gene_target_mapping.json', 'r') as file:
    ko_gene_dict = json.load(file)

# Merge all unique values across all keys in the dictionary
all_unique_genes = list(set(gene for values in ko_gene_dict.values() for gene in values))

# Create a DataFrame with all keys as rows and all unique genes as columns
table = pd.DataFrame(columns=all_unique_genes, index=ko_gene_dict.keys())

# Populate the table with 1 or 0 based on the presence of genes in the dictionary
for row_key in table.index:
    for col_key in table.columns:
        table.loc[row_key, col_key] = 1 if col_key in ko_gene_dict[row_key] else 0

# Convert the table to integers for better clarity
table = table.astype(int)

# Add "_KO" to each row name
table.index = [f"{row}_KO" for row in table.index]

# Export the updated table to a CSV file
csv_file_path = 'KO_Gene_Response_Table_Updated.csv'
table.to_csv(csv_file_path)

# Print the file path for reference
print(f"The updated table has been saved to: {csv_file_path}")
