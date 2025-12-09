import pandas as pd
import anndata as ad
import numpy as np

def create_adata_from_activity_matrix(original_adata, activity_matrix_path):
    """
    Create a new AnnData object from scMINER activity matrix while preserving
    all metadata from the original AnnData object.

    Parameters:
    -----------
    original_adata : AnnData
        Original Perturb-Map AnnData object
    activity_matrix_path : str
        Path to the scMINER activity matrix CSV file

    Returns:
    --------
    new_adata : AnnData
        New AnnData object with activity matrix as .X and preserved metadata
    """

    # Load the activity matrix
    activity_df = pd.read_csv(activity_matrix_path, index_col=0)

    # Transpose to have cells as rows and activities as columns (standard AnnData format)
    activity_df = activity_df.T

    # Verify that the cell barcodes match between original and activity matrix
    original_barcodes = set(original_adata.obs_names)
    activity_barcodes = set(activity_df.index)

    # Check for mismatches
    missing_in_activity = original_barcodes - activity_barcodes
    missing_in_original = activity_barcodes - original_barcodes

    if missing_in_activity:
        print(f"Warning: {len(missing_in_activity)} cells from original data missing in activity matrix")
    if missing_in_original:
        print(f"Warning: {len(missing_in_original)} cells in activity matrix not found in original data")

    # CRITICAL FIX: Get common cells in the ORDER they appear in the original data
    # This ensures proper alignment of metadata
    common_cells_ordered = [cell for cell in original_adata.obs_names if cell in activity_barcodes]
    print(f"Found {len(common_cells_ordered)} common cells between datasets")

    # Subset both datasets to common cells, preserving order
    activity_subset = activity_df.loc[common_cells_ordered]  # Use ordered list
    
    # Create mapping from cell names to indices in original data
    original_cell_to_idx = {cell: i for i, cell in enumerate(original_adata.obs_names)}
    original_subset_idx = [original_cell_to_idx[cell] for cell in common_cells_ordered]

    # Create new AnnData object with activity matrix as .X
    new_adata = ad.AnnData(X=activity_subset.values)

    # Set cell barcodes (observations) - now in correct order
    new_adata.obs_names = activity_subset.index

    # Set activity names (variables/features)
    new_adata.var_names = activity_subset.columns

    # Transfer all observation metadata (.obs) - now properly aligned
    new_adata.obs = original_adata.obs.iloc[original_subset_idx].copy()
    
    # Ensure obs index matches the new obs_names
    new_adata.obs.index = new_adata.obs_names

    # Transfer all variable metadata (.var) - create empty for activities
    # You might want to add specific information about the activities here
    new_adata.var['activity_type'] = 'scMINER_activity'

    # Transfer unstructured annotations (.uns)
    new_adata.uns = original_adata.uns.copy()

    # Add information about the transformation
    new_adata.uns['scMINER_info'] = {
        'original_n_genes': original_adata.n_vars,
        'activity_matrix_path': activity_matrix_path,
        'n_activities': new_adata.n_vars,
        'transformation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    # Transfer observation pairwise annotations (.obsp) if they exist
    if original_adata.obsp:
        for key in original_adata.obsp.keys():
            # Subset the matrix to common cells (both rows and columns) with correct indexing
            original_matrix = original_adata.obsp[key]
            new_adata.obsp[key] = original_matrix[np.ix_(original_subset_idx, original_subset_idx)]

    # Transfer observation-variable mappings (.obsm) - spatial coordinates, embeddings, etc.
    # CRITICAL FIX: Use properly ordered indices
    if original_adata.obsm:
        for key in original_adata.obsm.keys():
            new_adata.obsm[key] = original_adata.obsm[key][original_subset_idx].copy()

    # Transfer variable pairwise annotations (.varp) - skip as we have different variables
    # Transfer layers if you want to preserve any from original (usually not needed for activity matrix)

    print(f"Created new AnnData object:")
    print(f"  - Shape: {new_adata.shape}")
    print(f"  - Observations (cells): {new_adata.n_obs}")
    print(f"  - Variables (activities): {new_adata.n_vars}")
    print(f"  - Preserved metadata: obs({len(new_adata.obs.columns)}), obsm({len(new_adata.obsm)}), obsp({len(new_adata.obsp)}), uns({len(new_adata.uns)})")

    # Verify alignment by checking a few cells
    print(f"\nVerifying alignment:")
    print(f"  - First 3 cell barcodes in new_adata: {list(new_adata.obs_names[:3])}")
    print(f"  - First 3 cell barcodes in activity data: {list(activity_subset.index[:3])}")
    if 'spatial' in new_adata.obsm:
        print(f"  - Spatial coordinates shape: {new_adata.obsm['spatial'].shape}")

    return new_adata

# Additional helper function to compare the two objects
def compare_adata_objects(original_adata, new_adata):
    """Compare original and new AnnData objects"""
    print("=== AnnData Comparison ===")
    print(f"Original shape: {original_adata.shape}")
    print(f"New shape: {new_adata.shape}")
    print(f"Common cells: {len(set(original_adata.obs_names).intersection(set(new_adata.obs_names)))}")
    print(f"Original .obs columns: {list(original_adata.obs.columns)}")
    print(f"New .obs columns: {list(new_adata.obs.columns)}")
    print(f"Original .obsm keys: {list(original_adata.obsm.keys()) if original_adata.obsm else 'None'}")
    print(f"New .obsm keys: {list(new_adata.obsm.keys()) if new_adata.obsm else 'None'}")
    print(f"Original .uns keys: {list(original_adata.uns.keys()) if original_adata.uns else 'None'}")
    print(f"New .uns keys: {list(new_adata.uns.keys()) if new_adata.uns else 'None'}")
    
    # Verify that spatial coordinates are properly aligned
    if 'spatial' in original_adata.obsm and 'spatial' in new_adata.obsm:
        print(f"\n=== Spatial Coordinate Verification ===")
        common_cells = list(set(original_adata.obs_names).intersection(set(new_adata.obs_names)))[:5]
        for cell in common_cells:
            orig_idx = list(original_adata.obs_names).index(cell)
            new_idx = list(new_adata.obs_names).index(cell)
            orig_coord = original_adata.obsm['spatial'][orig_idx]
            new_coord = new_adata.obsm['spatial'][new_idx]
            print(f"  Cell {cell}: Original {orig_coord} vs New {new_coord} - Match: {np.array_equal(orig_coord, new_coord)}")

# Example usage:
# Load your original AnnData object
# original_adata = ad.read_h5ad('path_to_original_perturbmap.h5ad')

# Create new AnnData from activity matrix
# new_adata = create_adata_from_activity_matrix(
#     original_adata,
#     'PerturbMap_KP1_2_scMINER_activity_matrix.csv'
# )

# Save the new AnnData object
# new_adata.write('perturbmap_activity_matrix.h5ad')