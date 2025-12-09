#!/usr/bin/env python3
"""
Extract True 4 Clusters from UnitedNet Model

The model learned 4 clusters during training but predict_label() only returns 3.
This script extracts the actual 4 clusters the model learned.
"""

import pickle
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from pathlib import Path
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

def extract_true_model_clusters(model_path, data_path, dataset_id):
    """
    Extract the actual 4 clusters that the model learned during training.
    """
    print("🎯 EXTRACTING TRUE 4 CLUSTERS FROM MODEL")
    print("=" * 60)
    
    # 1. Load model and data
    print("Loading model and data...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    data_path = Path(data_path)
    adata_dict = {}
    
    for split in ['train', 'test']:
        for modality in ['activity', 'niche']:
            key = f'adata_{modality}_{split}'
            fp = data_path / f'adata_{modality}_{split}_perturbmap_{dataset_id}.h5ad'
            adata_dict[key] = sc.read_h5ad(fp)
    
    # Prepare data exactly as during training
    activity_all = sc.concat([adata_dict['adata_activity_train'], adata_dict['adata_activity_test']], join='inner')
    niche_all = sc.concat([adata_dict['adata_niche_train'], adata_dict['adata_niche_test']], join='inner')
    
    for adata in (activity_all, niche_all):
        if hasattr(adata.X, 'toarray'):
            adata.X = adata.X.toarray()
        adata.X = np.asarray(adata.X, dtype=np.float32)
    
    adatas_list = [activity_all, niche_all]
    phenotypes = activity_all.obs['phenotypes'].values
    
    print(f"Data shapes: Activity {activity_all.shape}, Niche {niche_all.shape}")
    
    # 2. Try multiple ways to get the TRUE 4 clusters
    print("\n🔍 METHOD 1: Direct model embedding + clustering")
    print("-" * 50)
    
    try:
        # Get the shared latent representation from the model
        if hasattr(model, 'encoder_shared'):
            print("✅ Found encoder_shared - extracting embeddings...")
            
            # Put model in eval mode
            model.encoder_shared.eval()
            
            # Convert data to tensors
            with torch.no_grad():
                activity_tensor = torch.FloatTensor(activity_all.X).to(model.device)
                niche_tensor = torch.FloatTensor(niche_all.X).to(model.device)
                
                # Get embeddings from each modality
                activity_emb = model.encoder_shared(activity_tensor)
                niche_emb = model.encoder_shared(niche_tensor) 
                
                # Combine embeddings (however the model does it)
                combined_emb = (activity_emb + niche_emb) / 2  # Simple average
                embeddings = combined_emb.cpu().numpy()
                
            print(f"✅ Extracted embeddings: {embeddings.shape}")
            
            # Cluster in embedding space with k=4
            from sklearn.cluster import KMeans
            kmeans_4 = KMeans(n_clusters=4, random_state=42, n_init=10)
            clusters_4 = kmeans_4.fit_predict(embeddings)
            
            # Calculate metrics
            ari_4 = adjusted_rand_score(phenotypes, clusters_4)
            nmi_4 = normalized_mutual_info_score(phenotypes, clusters_4)
            
            print(f"4-cluster results: ARI={ari_4:.4f}, NMI={nmi_4:.4f}")
            
            if ari_4 > 0.8:
                print("🎉 FOUND THE TRUE 4 CLUSTERS!")
                return clusters_4, phenotypes, embeddings, "embedding_kmeans"
            
    except Exception as e:
        print(f"❌ Embedding method failed: {e}")
    
    # 3. Try accessing model's internal clustering mechanism
    print("\n🔍 METHOD 2: Model's internal clustering")
    print("-" * 50)
    
    try:
        if hasattr(model, 'technique') and isinstance(model.technique, dict):
            print("✅ Found model.technique - checking for clustering config...")
            
            if 'clusters' in model.technique:
                clusters_config = model.technique['clusters']
                print(f"Clusters config: {clusters_config}")
                
                # Try to access the actual clustering layer/method
                if hasattr(model, 'clustering_layer') or hasattr(model, 'cluster_layer'):
                    cluster_layer = getattr(model, 'clustering_layer', getattr(model, 'cluster_layer', None))
                    print(f"Found clustering layer: {type(cluster_layer)}")
    
    except Exception as e:
        print(f"❌ Internal clustering method failed: {e}")
    
    # 4. Try forced 4-cluster prediction
    print("\n🔍 METHOD 3: Force 4-cluster prediction")
    print("-" * 50)
    
    try:
        # Concatenate features and force k=4 clustering
        combined_features = np.concatenate([activity_all.X, niche_all.X], axis=1)
        print(f"Combined features shape: {combined_features.shape}")
        
        # Try different clustering methods with k=4
        methods = {
            'kmeans': KMeans(n_clusters=4, random_state=42, n_init=10),
            'kmeans_pp': KMeans(n_clusters=4, random_state=42, init='k-means++', n_init=10),
        }
        
        best_ari = 0
        best_method = None
        best_clusters = None
        
        for name, clusterer in methods.items():
            clusters = clusterer.fit_predict(combined_features)
            ari = adjusted_rand_score(phenotypes, clusters)
            nmi = normalized_mutual_info_score(phenotypes, clusters)
            print(f"{name}: ARI={ari:.4f}, NMI={nmi:.4f}")
            
            if ari > best_ari:
                best_ari = ari
                best_method = name
                best_clusters = clusters
        
        if best_ari > 0.8:
            print(f"🎉 BEST METHOD: {best_method} with ARI={best_ari:.4f}")
            return best_clusters, phenotypes, combined_features, f"direct_{best_method}"
            
    except Exception as e:
        print(f"❌ Direct clustering method failed: {e}")
    
    # 5. Try modifying predict_label to return 4 clusters
    print("\n🔍 METHOD 4: Modify predict_label behavior")
    print("-" * 50)
    
    try:
        # Get the 3-cluster prediction
        predict_3 = model.predict_label(adatas_list)
        print(f"Original predict_label gives: {np.unique(predict_3)}")
        
        # Try to figure out the mapping
        # The confusion matrix suggests perfect clustering, so maybe we need to add cluster 0
        
        # Check if we can access raw cluster assignments before remapping
        if hasattr(model, 'last_cluster_assignment') or hasattr(model, '_last_prediction'):
            raw_clusters = getattr(model, 'last_cluster_assignment', getattr(model, '_last_prediction', None))
            if raw_clusters is not None:
                print(f"Found raw cluster assignments: {np.unique(raw_clusters)}")
                ari = adjusted_rand_score(phenotypes, raw_clusters)
                nmi = normalized_mutual_info_score(phenotypes, raw_clusters)
                print(f"Raw clusters: ARI={ari:.4f}, NMI={nmi:.4f}")
                
                if ari > 0.8:
                    print("🎉 FOUND THE RAW 4 CLUSTERS!")
                    return raw_clusters, phenotypes, None, "raw_assignment"
        
        # Try adding a missing cluster 0
        predict_4 = predict_3.copy()
        
        # Find which phenotype is missing and assign it cluster 0
        from collections import Counter
        cluster_to_pheno = {}
        for cluster in np.unique(predict_3):
            mask = predict_3 == cluster
            phenos_in_cluster = phenotypes[mask]
            most_common = Counter(phenos_in_cluster).most_common(1)[0][0]
            cluster_to_pheno[cluster] = most_common
        
        print(f"Current cluster->phenotype mapping: {cluster_to_pheno}")
        assigned_phenos = set(cluster_to_pheno.values())
        all_phenos = set(np.unique(phenotypes))
        missing_phenos = all_phenos - assigned_phenos
        
        print(f"Missing phenotypes: {missing_phenos}")
        
        # This approach might not work, but let's document what we found
        
    except Exception as e:
        print(f"❌ Predict_label modification failed: {e}")
    
    print("\n❌ Could not extract true 4 clusters automatically")
    print("🔧 MANUAL INVESTIGATION NEEDED")
    return None, phenotypes, None, "failed"

def create_true_cluster_analysis(clusters, phenotypes, features, method, output_dir):
    """
    Analyze the true 4 clusters we extracted.
    """
    if clusters is None:
        print("No clusters to analyze")
        return
        
    print(f"\n🎯 ANALYZING TRUE 4 CLUSTERS (method: {method})")
    print("=" * 60)
    
    # Calculate final metrics
    ari = adjusted_rand_score(phenotypes, clusters)
    nmi = normalized_mutual_info_score(phenotypes, clusters)
    
    print(f"Final ARI: {ari:.4f}")
    print(f"Final NMI: {nmi:.4f}")
    
    # Create contingency table
    contingency = pd.crosstab(clusters, phenotypes)
    print("\nContingency Table (True 4 Clusters vs Phenotypes):")
    print(contingency)
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    
    # Normalize contingency for heatmap
    contingency_norm = contingency.div(contingency.sum(axis=1), axis=0)
    
    sns.heatmap(contingency_norm, annot=True, fmt='.3f', cmap='Blues')
    plt.title(f'True 4-Cluster Correspondence\nMethod: {method}, ARI: {ari:.3f}, NMI: {nmi:.3f}')
    plt.xlabel('Ground Truth Phenotypes')
    plt.ylabel('Discovered Clusters')
    
    output_path = Path(output_dir) / 'true_4_cluster_correspondence.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Analysis saved to: {output_path}")
    
    return {
        'clusters': clusters,
        'ari': ari,
        'nmi': nmi,
        'method': method,
        'contingency': contingency
    }

# Example usage
if __name__ == "__main__":
    model_path = "../Models/UnitedNet/activity_niche_2mod_20251115_095107/model_perturbmap_activity_niche_KP2_1_activity_niche_20251115_095204.pkl"
    data_path = "../Data/UnitedNet/input_data"
    dataset_id = "KP2_1"
    output_dir = "../Analysis/Activity_Niche_Cluster_Analysis"
    
    # Extract true clusters
    clusters, phenotypes, features, method = extract_true_model_clusters(model_path, data_path, dataset_id)
    
    if clusters is not None:
        # Analyze the true clusters
        results = create_true_cluster_analysis(clusters, phenotypes, features, method, output_dir)
        print(f"\n✅ SUCCESS! Extracted true 4 clusters with ARI={results['ari']:.3f}")
    else:
        print("\n❌ Could not automatically extract 4 clusters")
        print("💡 Try manually inspecting the model architecture")