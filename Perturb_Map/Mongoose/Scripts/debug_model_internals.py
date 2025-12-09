# Debug the Model's Internal Cluster Outputs
# Let's see exactly what the model is producing step by step

import pickle
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from pathlib import Path
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Load model and data
model_path = "../Models/UnitedNet/activity_niche_2mod_20251115_095107/model_perturbmap_activity_niche_KP2_1_activity_niche_20251115_095204.pkl"
data_path = "../Data/UnitedNet/input_data"
dataset_id = "KP2_1"

with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Prepare data
data_path = Path(data_path)
adata_dict = {}
for split in ['train', 'test']:
    for modality in ['activity', 'niche']:
        key = f'adata_{modality}_{split}'
        fp = data_path / f'adata_{modality}_{split}_perturbmap_{dataset_id}.h5ad'
        adata_dict[key] = sc.read_h5ad(fp)

activity_all = sc.concat([adata_dict['adata_activity_train'], adata_dict['adata_activity_test']], join='inner')
niche_all = sc.concat([adata_dict['adata_niche_train'], adata_dict['adata_niche_test']], join='inner')

for adata in (activity_all, niche_all):
    if hasattr(adata.X, 'toarray'):
        adata.X = adata.X.toarray()
    adata.X = np.asarray(adata.X, dtype=np.float32)

modalities = [activity_all, niche_all]
phenotypes = activity_all.obs['phenotypes'].values

print("🔍 DEBUGGING MODEL'S INTERNAL CLUSTER OUTPUTS")
print("=" * 60)

# Monkey patch the model to intercept internal outputs
original_forward = model.forward

def debug_forward(self, modalities_input, labels=None):
    print("🕵️ Intercepting model.forward()...")
    
    # Call the original forward but capture intermediate results
    modalities_input = self.impute_check(modalities_input)
    modalities_input = [modality.to(device=self.device_in_use) for modality in modalities_input]
    
    if self.noise_level != None:
        self.modalities = self.add_noise(inputs=modalities_input, levels=self.noise_level, device=self.device_in_use)
    
    self.labels = labels.to(device=self.device_in_use) if labels is not None else None
    
    # Get latents
    self.latents = [encoder(modality) for (encoder, modality) in zip(self.encoders, self.modalities)]
    
    # Cluster weight normalization
    with torch.no_grad():
        for pt_i in range(self.n_head):
            w = getattr(self.clusters[pt_i], "layers")[0].weight.data.clone()
            w = torch.nn.functional.normalize(w, dim=1, p=2)
            getattr(self.clusters[pt_i], "layers")[0].weight.copy_(w)
    
    # Get other outputs
    self.latent_projection = self.latent_projector(torch.cat(self.latents, dim=0))
    self.translations = [[decoder(latent) for latent in self.latents] for decoder in self.decoders]
    
    # Fusion and clustering - THE CRITICAL PART
    self.fused_latents = [fuser(self.latents) for fuser in self.fusers]
    self.hiddens = [projector(fused_latent) for (projector, fused_latent) in zip(self.projectors, self.fused_latents)]
    
    print(f"  Number of heads: {self.n_head}")
    print(f"  Best head: {self.best_head}")
    
    # CLUSTER OUTPUTS - Check each head
    self.cluster_outputs = [cluster(hidden) for (cluster, hidden) in zip(self.clusters, self.hiddens)]
    
    for head_idx, cluster_output in enumerate(self.cluster_outputs):
        print(f"  Head {head_idx} cluster_output shape: {cluster_output.shape}")
        print(f"  Head {head_idx} cluster_output range: [{cluster_output.min():.3f}, {cluster_output.max():.3f}]")
        
        # Apply softmax
        probs = self.prob_layer(cluster_output)
        print(f"  Head {head_idx} probabilities shape: {probs.shape}")
        print(f"  Head {head_idx} probabilities sum per row (should be ~1): {probs.sum(dim=1)[:5]}")
        
        # Get predictions
        predictions = torch.argmax(probs, axis=1)
        unique_preds = torch.unique(predictions)
        print(f"  Head {head_idx} predictions unique values: {unique_preds.cpu().numpy()}")
        print(f"  Head {head_idx} predictions shape: {predictions.shape}")
        
        if head_idx == self.best_head:
            print(f"  ✅ HEAD {head_idx} IS THE BEST HEAD")
    
    # FINAL PREDICTIONS
    self.predictions = [torch.argmax(self.prob_layer(cluster_outputs), axis=1) for cluster_outputs in self.cluster_outputs]
    
    final_prediction = self.predictions[self.best_head]
    print(f"\n🎯 FINAL RESULTS:")
    print(f"  Best head index: {self.best_head}")
    print(f"  Final prediction shape: {final_prediction.shape}")
    print(f"  Final prediction unique values: {torch.unique(final_prediction).cpu().numpy()}")
    print(f"  Final prediction counts:")
    
    unique, counts = torch.unique(final_prediction, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"    Cluster {u.item()}: {c.item()} cells")
    
    # Compare with training performance
    final_pred_np = final_prediction.cpu().numpy()
    ari = adjusted_rand_score(phenotypes, final_pred_np)
    print(f"  Current ARI: {ari:.4f}")
    print(f"  Training ARI: {model.finetune_ari_list[-1]:.4f}")
    
    # Return the normal outputs
    return (
        [[trans_lb if self.training else trans_lb.cpu().numpy() for trans_lb in trans_la] for trans_la in self.translations],
        self.predictions[self.best_head] if self.training else self.predictions[self.best_head].cpu().numpy(),
        self.fused_latents[self.best_head] if self.training else self.fused_latents[self.best_head].cpu().numpy(),
    )

# Apply the debug wrapper
import types
model.forward = types.MethodType(debug_forward, model)

# Run prediction with debugging
model.eval()
with torch.no_grad():
    debug_output = model(modalities)

print(f"\n📊 DEBUG SUMMARY:")
print(f"The model internally produces the cluster outputs shown above.")
print(f"Check if the best_head is correct and if cluster 0 appears in any head.")