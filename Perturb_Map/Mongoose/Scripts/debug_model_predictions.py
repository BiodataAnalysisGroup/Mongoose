#!/usr/bin/env python3
"""
Debug Model Loading and Prediction Consistency

This script verifies that we're loading the correct model and getting
the same predictions that were used during training/finetuning.
"""

import pickle
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import torch

def debug_model_and_predictions(model_path, data_path, dataset_id):
    """
    Debug model loading and prediction consistency.
    """
    print("🔧 DEBUGGING MODEL AND PREDICTIONS")
    print("=" * 60)
    
    # 1. VERIFY MODEL LOADING
    print("1️⃣ MODEL VERIFICATION:")
    print("-" * 30)
    
    print(f"Loading model from: {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"✅ Model loaded successfully")
    print(f"Model type: {type(model)}")
    
    # Check model attributes
    if hasattr(model, '__dict__'):
        print("Model attributes:")
        for key, value in model.__dict__.items():
            if not key.startswith('_') and not callable(value):
                print(f"  {key}: {type(value)}")
    
    # Check if model has training history
    if hasattr(model, 'history') or hasattr(model, 'training_history'):
        print("📊 Model has training history!")
        history = getattr(model, 'history', getattr(model, 'training_history', None))
        if history is not None:
            print(f"  History length: {len(history) if isinstance(history, (list, dict)) else 'Unknown'}")
    
    # 2. VERIFY DATA LOADING AND SHAPES
    print("\n2️⃣ DATA VERIFICATION:")
    print("-" * 30)
    
    data_path = Path(data_path)
    adata_dict = {}
    
    for split in ['train', 'test']:
        for modality in ['activity', 'niche']:
            key = f'adata_{modality}_{split}'
            fp = data_path / f'adata_{modality}_{split}_perturbmap_{dataset_id}.h5ad'
            adata_dict[key] = sc.read_h5ad(fp)
            print(f"{key}: {adata_dict[key].shape}")
    
    # Prepare data exactly as in the correspondence script
    activity_all = sc.concat([adata_dict['adata_activity_train'], adata_dict['adata_activity_test']], join='inner')
    niche_all = sc.concat([adata_dict['adata_niche_train'], adata_dict['adata_niche_test']], join='inner')
    
    # Ensure dense matrices (exactly as in correspondence script)
    for adata in (activity_all, niche_all):
        if hasattr(adata.X, 'toarray'):
            adata.X = adata.X.toarray()
        adata.X = np.asarray(adata.X, dtype=np.float32)
    
    print(f"✅ Combined activity data: {activity_all.shape}")
    print(f"✅ Combined niche data: {niche_all.shape}")
    
    # 3. CHECK GROUND TRUTH LABELS
    print("\n3️⃣ GROUND TRUTH VERIFICATION:")
    print("-" * 30)
    
    if 'phenotypes' in activity_all.obs.columns:
        phenotypes = activity_all.obs['phenotypes'].values
        unique_phenos = np.unique(phenotypes)
        print(f"✅ Found phenotypes column")
        print(f"Unique phenotypes: {unique_phenos}")
        print(f"Phenotype counts:")
        for pheno in unique_phenos:
            count = np.sum(phenotypes == pheno)
            print(f"  {pheno}: {count} cells")
    else:
        print("❌ No phenotypes column found!")
        return None
    
    # 4. TEST MODEL PREDICTION
    print("\n4️⃣ MODEL PREDICTION TEST:")
    print("-" * 30)
    
    try:
        print("Attempting model.predict_label([activity_all, niche_all])...")
        adatas_list = [activity_all, niche_all]
        
        # Check if model expects specific input format
        predict_label = model.predict_label(adatas_list)
        print(f"✅ Model prediction successful!")
        print(f"Prediction shape: {predict_label.shape}")
        print(f"Unique predicted clusters: {np.unique(predict_label)}")
        print(f"Prediction type: {type(predict_label)}")
        
        # Calculate metrics using the EXACT same ground truth
        if 'label' in activity_all.obs.columns:
            print("\n🎯 Using 'label' column (integer encoded):")
            gt_labels = activity_all.obs['label'].values
            ari_label = adjusted_rand_score(gt_labels, predict_label)
            nmi_label = normalized_mutual_info_score(gt_labels, predict_label)
            print(f"ARI (vs label): {ari_label:.4f}")
            print(f"NMI (vs label): {nmi_label:.4f}")
        
        print("\n🎯 Using 'phenotypes' column (string):")
        ari_pheno = adjusted_rand_score(phenotypes, predict_label)
        nmi_pheno = normalized_mutual_info_score(phenotypes, predict_label)
        print(f"ARI (vs phenotypes): {ari_pheno:.4f}")
        print(f"NMI (vs phenotypes): {nmi_pheno:.4f}")
        
        # 5. DETAILED COMPARISON
        print("\n5️⃣ DETAILED COMPARISON WITH TRAINING METRICS:")
        print("-" * 50)
        print("Expected from training:")
        print("  Finetune ARI: 0.9333")
        print("  Finetune NMI: 0.9150")
        print()
        print("Current prediction results:")
        print(f"  ARI: {ari_pheno:.4f}")
        print(f"  NMI: {nmi_pheno:.4f}")
        print()
        
        if ari_pheno < 0.8 or nmi_pheno < 0.8:
            print("⚠️  MAJOR DISCREPANCY DETECTED!")
            print("Possible causes:")
            print("1. Wrong model file loaded")
            print("2. Data preprocessing mismatch")
            print("3. Different train/test split")
            print("4. Model state not final finetuned version")
            print("5. predict_label method using wrong weights")
        else:
            print("✅ Metrics look consistent with training!")
        
        # 6. CHECK IF WE CAN ACCESS MODEL INTERNALS
        print("\n6️⃣ MODEL INTERNAL STATE:")
        print("-" * 30)
        
        # Try to access model components
        if hasattr(model, 'encoder_shared'):
            print("✅ Found encoder_shared")
        if hasattr(model, 'technique'):
            print("✅ Found technique")
            if isinstance(model.technique, dict):
                print(f"  Technique keys: {list(model.technique.keys())}")
        
        # Check if we can manually compute predictions
        if hasattr(model, 'predict') or hasattr(model, 'transform'):
            print("✅ Model has direct prediction methods")
        
        return {
            'model': model,
            'predictions': predict_label,
            'ground_truth': phenotypes,
            'ari': ari_pheno,
            'nmi': nmi_pheno,
            'adatas': adatas_list
        }
        
    except Exception as e:
        print(f"❌ Model prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_with_training_data(model_path, data_path, dataset_id):
    """
    Try to recreate the exact conditions from training to see if we get the same metrics.
    """
    print("\n🔬 COMPARING WITH TRAINING CONDITIONS:")
    print("=" * 50)
    
    # Check if we can find any training logs or intermediate files
    model_dir = Path(model_path).parent
    print(f"Model directory: {model_dir}")
    
    # Look for training logs, config files, etc.
    log_files = list(model_dir.glob("*.log"))
    config_files = list(model_dir.glob("*.json")) + list(model_dir.glob("*.yaml"))
    metrics_files = list(model_dir.glob("*metrics*"))
    
    print(f"Found {len(log_files)} log files")
    print(f"Found {len(config_files)} config files")
    print(f"Found {len(metrics_files)} metrics files")
    
    # Try to load any metrics files
    for metrics_file in metrics_files:
        try:
            if metrics_file.suffix == '.csv':
                df = pd.read_csv(metrics_file)
                print(f"📊 Loaded metrics from {metrics_file.name}")
                print(f"Last few rows:")
                print(df.tail())
            elif metrics_file.suffix == '.json':
                import json
                with open(metrics_file, 'r') as f:
                    data = json.load(f)
                print(f"📊 Loaded metrics from {metrics_file.name}")
                print(f"Keys: {list(data.keys())}")
        except Exception as e:
            print(f"Could not load {metrics_file}: {e}")

# Example usage
if __name__ == "__main__":
    model_path = "../Models/UnitedNet/activity_niche_2mod_20251115_095107/model_perturbmap_activity_niche_KP2_1_activity_niche_20251115_095204.pkl"
    data_path = "../Data/UnitedNet/input_data"
    dataset_id = "KP2_1"
    
    # Run debugging
    results = debug_model_and_predictions(model_path, data_path, dataset_id)
    
    # Compare with training data
    compare_with_training_data(model_path, data_path, dataset_id)
    
    if results is not None:
        print(f"\n🎯 FINAL DIAGNOSIS:")
        if results['ari'] < 0.8:
            print("❌ PROBLEM CONFIRMED: ARI is too low")
            print("🔧 NEXT STEPS:")
            print("1. Verify this is the final finetuned model")
            print("2. Check model.predict_label() implementation")
            print("3. Ensure data preprocessing matches training")
        else:
            print("✅ Model predictions look correct!")