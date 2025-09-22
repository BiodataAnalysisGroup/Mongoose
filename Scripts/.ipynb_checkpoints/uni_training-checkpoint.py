#!/usr/bin/env python3
"""
UnitedNet PerturbMap Training Script

This script trains a UnitedNet model for multi-modal biological data analysis
using the PerturbMap technique. It supports various configurations and 
automatically handles data loading, preprocessing, and model training.

Usage:
    python uni_training.py --data_path ../Data/perturbmap_stomicsdb --device cuda:0
"""

import argparse
import os
import pickle
import json
import copy as copy
from datetime import datetime
from pathlib import Path
from functools import reduce
import sys
import math
import importlib.util

# Import libraries from general conda environment
import anndata as ad
import anndata
import numpy as np
import scanpy as sc
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr, pearsonr
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from sklearn.metrics import adjusted_rand_score, confusion_matrix, accuracy_score
from sklearn.utils.multiclass import unique_labels
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

# Import libraries from UnitedNet
from unitednet.interface import UnitedNet
from unitednet.configs import *
from unitednet.data import partitions, save_umap, generate_adata
from unitednet.scripts import ordered_cmat, assignmene_align
from unitednet.modules import submodel_trans, submodel_clus
from unitednet.data import save_obj, load_obj, type_specific_mean
from unitednet.plots import markers_chord_plot, type_relevance_chord_plot, feature_relevance_chord_plot, merge_sub_feature

# Import for SHAPs
import shap
import torch
import torch.nn as nn

# Import for plotting
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter
from collections import Counter

# Import STRINGdb for PPI network models and Networkx for visualisation and potential graph analysis
import stringdb
import networkx as nx


def get_timestamp():
    """Generate timestamp for model versioning"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dense_float32(adata):
    """Convert sparse matrices to dense float32 arrays"""
    if sp.issparse(adata.X):
        adata.X = adata.X.tocsr()
        adata.X.sort_indices()
        adata.X = adata.X.astype(np.float32).toarray()
    else:
        adata.X = np.asarray(adata.X, dtype=np.float32)
    return adata


def change_label(adata, batch):
    """Add standardized labels for UnitedNet processing"""
    adata.obs['batch'] = batch
    adata.obs['imagecol'] = adata.obs['array_col']
    adata.obs['imagerow'] = adata.obs['array_row'] 
    adata.obs['label'] = adata.obs['phenotypes']
    return adata


def pre_ps(adata_list, sc_pre=None):
    """Preprocess data with standardization"""
    adata_list_all = [ad_x.copy() for ad_x in adata_list]
    scalers = []
    
    assert (adata_list_all[0].X >= 0).all(), "Contaminated input data detected"
    
    for idx, mod in enumerate(adata_list_all):
        t_x = mod.X
        if sc_pre is not None:
            scaler = sc_pre[idx]
        else:
            scaler = preprocessing.StandardScaler().fit(t_x)
        t_x = scaler.transform(t_x)
        mod.X = t_x
        adata_list_all[idx] = mod
        scalers.append(scaler)
    
    return adata_list_all, scalers


def load_adata_files(data_path, dataset_id):
    """Load all required adata files for training"""
    files_to_load = {
        'adata_rna_train': f'adata_rna_train_perturbmap_{dataset_id}.h5ad',
        'adata_niche_train': f'adata_niche_train_perturbmap_{dataset_id}.h5ad', 
        'adata_activity_train': f'adata_activity_train_perturbmap_{dataset_id}.h5ad',
        'adata_rna_test': f'adata_rna_test_perturbmap_{dataset_id}.h5ad',
        'adata_niche_test': f'adata_niche_test_perturbmap_{dataset_id}.h5ad',
        'adata_activity_test': f'adata_activity_test_perturbmap_{dataset_id}.h5ad'
    }
    
    adata_dict = {}
    for key, filename in files_to_load.items():
        filepath = Path(data_path) / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Required file not found: {filepath}")
        print(f"Loading {key} from {filepath}")
        adata_dict[key] = sc.read_h5ad(filepath)
    
    return adata_dict


def get_default_perturbmap_config():
    """Returns the default PerturbMap configuration"""
    return {
        'train_batch_size': 8,
        'finetune_batch_size': 8,
        'transfer_batch_size': None,
        'train_epochs': 20,
        'finetune_epochs': 10,
        'transfer_epochs': None,
        'train_task': 'cross_model_prediction_clus',
        'finetune_task': 'unsupervised_group_identification',
        'transfer_task': None,
        'train_loss_weight': None,
        'finetune_loss_weight': None,
        'transfer_loss_weight': None,
        'lr': 0.001,
        'checkpoint': 20,
        'n_head': 10,
        'noise_level': [0, 0, 0],
        'fuser_type': 'WeightedMean',
        
        'encoders': [
            {
                'input': 2001,
                'hiddens': [512, 256, 128, 64],
                'output': 64,
                'use_biases': [True, True, True, True, True],
                'dropouts': [0, 0, 0, 0, 0],
                'activations': ['relu', 'relu', 'relu', 'relu', None],
                'use_batch_norms': [False, False, False, False, False],
                'use_layer_norms': [False, False, False, False, True],
                'is_binary_input': False
            },
            {
                'input': 2001,
                'hiddens': [512, 256, 128, 64],
                'output': 64,
                'use_biases': [True, True, True, True, True],
                'dropouts': [0, 0, 0, 0, 0],
                'activations': ['relu', 'relu', 'relu', 'relu', None],
                'use_batch_norms': [False, False, False, False, False],
                'use_layer_norms': [False, False, False, False, True],
                'is_binary_input': False
            },
            {
                'input': 2000,
                'hiddens': [512, 256, 128, 64],
                'output': 64,
                'use_biases': [True, True, True, True, True],
                'dropouts': [0, 0, 0, 0, 0],
                'activations': ['relu', 'relu', 'relu', 'relu', None],
                'use_batch_norms': [False, False, False, False, False],
                'use_layer_norms': [False, False, False, False, True],
                'is_binary_input': False
            }
        ],
        
        'latent_projector': None,
        
        'decoders': [
            {
                'input': 64,
                'hiddens': [64, 128, 256, 512],
                'output': 2001,
                'use_biases': [True, True, True, True, True],
                'dropouts': [0, 0, 0, 0, 0],
                'activations': ['relu', 'relu', 'relu', 'relu', None],
                'use_batch_norms': [False, False, False, False, False],
                'use_layer_norms': [False, False, False, False, False]
            },
            {
                'input': 64,
                'hiddens': [64, 128, 256, 512],
                'output': 2001,
                'use_biases': [True, True, True, True, True],
                'dropouts': [0, 0, 0, 0, 0],
                'activations': ['relu', 'relu', 'relu', 'relu', None],
                'use_batch_norms': [False, False, False, False, False],
                'use_layer_norms': [False, False, False, False, False]
            },
            {
                'input': 64,
                'hiddens': [64, 128, 256, 512],
                'output': 2000,
                'use_biases': [True, True, True, True, True],
                'dropouts': [0, 0, 0, 0, 0],
                'activations': ['relu', 'relu', 'relu', 'relu', None],
                'use_batch_norms': [False, False, False, False, False],
                'use_layer_norms': [False, False, False, False, False]
            }
        ],
        
        'discriminators': [
            {
                'input': 2001,
                'hiddens': [512, 256, 256],
                'output': 1,
                'use_biases': [True, True, True, True],
                'dropouts': [0, 0, 0, 0],
                'activations': ['relu', 'relu', 'relu', 'sigmoid'],
                'use_batch_norms': [False, False, False, False],
                'use_layer_norms': [False, False, False, True]
            },
            {
                'input': 2001,
                'hiddens': [512, 256, 256],
                'output': 1,
                'use_biases': [True, True, True, True],
                'dropouts': [0, 0, 0, 0],
                'activations': ['relu', 'relu', 'relu', 'sigmoid'],
                'use_batch_norms': [False, False, False, False],
                'use_layer_norms': [False, False, False, True]
            },
            {
                'input': 2000,
                'hiddens': [512, 256, 256],
                'output': 1,
                'use_biases': [True, True, True, True],
                'dropouts': [0, 0, 0, 0],
                'activations': ['relu', 'relu', 'relu', 'sigmoid'],
                'use_batch_norms': [False, False, False, False],
                'use_layer_norms': [False, False, False, True]
            }
        ],
        
        'projectors': {
            'input': 64,
            'hiddens': [],
            'output': 100,
            'use_biases': [True],
            'dropouts': [0],
            'activations': ['relu'],
            'use_batch_norms': [False],
            'use_layer_norms': [True]
        },
        
        'clusters': {
            'input': 100,
            'hiddens': [],
            'output': 3,
            'use_biases': [False],
            'dropouts': [0],
            'activations': [None],
            'use_batch_norms': [False],
            'use_layer_norms': [False]
        }
    }


def load_config(config_path):
    """Load configuration from JSON or Python file"""
    config_path = Path(config_path)
    
    if config_path.suffix == '.json':
        with open(config_path, 'r') as f:
            config = json.load(f)
    elif config_path.suffix == '.py':
        # Load Python config file
        spec = importlib.util.spec_from_file_location("config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        # Look for common config variable names
        config_var_names = [
            'perturbmap_config', 'perturbmap_config_corrected', 
            'config', 'model_config', 'CONFIG'
        ]
        
        config = None
        for var_name in config_var_names:
            if hasattr(config_module, var_name):
                config = getattr(config_module, var_name)
                print(f"Loaded config from variable: {var_name}")
                break
        
        if config is None:
            # List available variables in the module
            available_vars = [name for name in dir(config_module) 
                            if not name.startswith('_') and isinstance(getattr(config_module, name), dict)]
            raise ValueError(f"No config found in {config_path}. Available dict variables: {available_vars}")
    else:
        raise ValueError(f"Unsupported config file format: {config_path.suffix}. Use .json or .py")
    
    return config


def find_config_file(config_name, config_dirs=None):
    """Find a configuration file in standard locations"""
    if config_dirs is None:
        config_dirs = [
            Path('./configs'),
            Path('./config_files'), 
            Path('../configs'),
            Path('../config_files'),
            Path('../Model/config_files'),  # Added your directory
            Path('.')
        ]
    
    # If config_name is already a full path and exists, return it
    if Path(config_name).exists():
        return Path(config_name)
    
    # Try different extensions if not specified
    if not any(config_name.endswith(ext) for ext in ['.json', '.py']):
        extensions = ['.json', '.py']
    else:
        extensions = ['']
    
    # Search in standard directories
    for config_dir in config_dirs:
        for ext in extensions:
            config_path = config_dir / (config_name + ext)
            if config_path.exists():
                return config_path
    
    # List available configs if not found
    available_configs = []
    for config_dir in config_dirs:
        if config_dir.exists():
            available_configs.extend([
                str(f.relative_to(config_dir)) 
                for f in config_dir.glob('*.json')
            ])
            available_configs.extend([
                str(f.relative_to(config_dir)) 
                for f in config_dir.glob('*.py')
            ])
    
    if available_configs:
        print(f"Available configurations: {', '.join(set(available_configs))}")
    
    return None


def save_config(config, output_dir, filename="config.json"):
    """Save configuration to JSON file"""
    config_path = output_dir / filename
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to {config_path}")
    return config_path


def setup_output_directory(base_path, technique, timestamp):
    """Create timestamped output directory"""
    if timestamp:
        output_dir = Path(base_path) / f"{technique}_{timestamp}"
    else:
        output_dir = Path(base_path) / technique
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Train UnitedNet PerturbMap model for multi-modal biological data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--data_path', type=str, 
                       default='../Data/UnitedNet/input_data',
                       help='Path to directory containing input data files')
    parser.add_argument('--dataset_id', type=str, default='KP2_1',
                       help='Dataset identifier (e.g., KP2_1)')
    
    # Model arguments  
    parser.add_argument('--technique', type=str, default='perturbmap',
                       choices=['perturbmap'], help='Training technique')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use (cuda:0, cuda:1, cpu)')
    
    # Training arguments
    parser.add_argument('--train_epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--finetune_epochs', type=int, default=10,
                       help='Number of finetuning epochs')
    parser.add_argument('--train_batch_size', type=int, default=8,
                       help='Training batch size')
    parser.add_argument('--finetune_batch_size', type=int, default=8,
                       help='Finetuning batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--n_clusters', type=int, default=3,
                       help='Number of clusters for group identification')
    
    # Output arguments
    parser.add_argument('--output_base', type=str, default='../Model',
                       help='Base directory for saving results')
    parser.add_argument('--model_suffix', type=str, default='',
                       help='Suffix for model filename')
    parser.add_argument('--timestamp', action='store_true',
                       help='Add timestamp to output directory')
    
    # Control arguments
    parser.add_argument('--train_model', action='store_true', default=True,
                       help='Train the model (if False, only load)')
    parser.add_argument('--no_train', dest='train_model', action='store_false',
                       help='Skip training, only load existing model')
    parser.add_argument('--save_intermediates', action='store_true',
                       help='Save intermediate processing results')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Enable verbose output')
    
    # Configuration arguments
    parser.add_argument('--config', type=str, default=None,
                       help='Configuration file name (searches in ./configs, ./config_files, etc.)')
    parser.add_argument('--config_file', type=str, default=None,
                       help='Full path to configuration JSON or Python file')
    parser.add_argument('--list_configs', action='store_true',
                       help='List available configuration files and exit')
    parser.add_argument('--noise_level', type=float, nargs=3, default=[0, 0, 0],
                       help='Noise levels for the 3 modalities')
    
    args = parser.parse_args()
    
    # Handle list configs option
    if args.list_configs:
        print("Searching for available configuration files...")
        config_dirs = [
            Path('./configs'),
            Path('./config_files'), 
            Path('../configs'),
            Path('../config_files'),
            Path('../Model/config_files'),  # Added your directory
            Path('.')
        ]
        
        available_configs = []
        for config_dir in config_dirs:
            if config_dir.exists():
                configs = list(config_dir.glob('*.json')) + list(config_dir.glob('*.py'))
                if configs:
                    print(f"\nIn {config_dir}:")
                    for config_file in configs:
                        print(f"  - {config_file.name}")
                        available_configs.append(f"{config_dir}/{config_file.name}")
        
        if not available_configs:
            print("No configuration files found in standard directories")
            print("Standard directories checked: configs/, config_files/, ../configs/, ../config_files/, ../Model/config_files/, ./")
        else:
            print(f"\nUsage examples:")
            print(f"  --config {Path(available_configs[0]).stem}")  # Without extension
            print(f"  --config_file {available_configs[0]}")
        return 0
    
    # Validate device
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available, switching to CPU")
        args.device = 'cpu'
    
    print(f"Starting UnitedNet {args.technique} training")
    print(f"Device: {args.device}")
    print(f"Data path: {args.data_path}")
    print(f"Dataset ID: {args.dataset_id}")
    
    # Setup output directory
    timestamp = get_timestamp() if args.timestamp else ""
    output_dir = setup_output_directory(args.output_base, args.technique, timestamp)
    root_save_path = str(output_dir)
    
    print(f"Output directory: {output_dir}")
    
    # Load configuration
    config = None
    config_source = "default"
    
    # Priority: 1. --config_file (full path), 2. --config (name search), 3. default
    if args.config_file and Path(args.config_file).exists():
        print(f"Loading configuration from full path: {args.config_file}")
        config = load_config(args.config_file)
        config_source = args.config_file
    elif args.config:
        print(f"Searching for configuration: {args.config}")
        config_path = find_config_file(args.config)
        if config_path:
            print(f"Found configuration at: {config_path}")
            config = load_config(config_path)
            config_source = str(config_path)
        else:
            print(f"Configuration '{args.config}' not found!")
            return 1
    else:
        print("Using default PerturbMap configuration")
        config = get_default_perturbmap_config()
    
    # Update config with command line arguments (only basic ones)
    config['train_epochs'] = args.train_epochs
    config['finetune_epochs'] = args.finetune_epochs
    config['train_batch_size'] = args.train_batch_size
    config['finetune_batch_size'] = args.finetune_batch_size
    config['lr'] = args.lr
    config['noise_level'] = args.noise_level
    config['clusters']['output'] = args.n_clusters
    
    # Save configuration
    config_path = save_config(config, output_dir)
    
    print(f"Configuration loaded from: {config_source}")
    print(f"Configuration saved to: {config_path}")
    
    try:
        # Load data files
        print("\n=== Loading Data ===")
        adata_dict = load_adata_files(args.data_path, args.dataset_id)
        
        # Extract individual datasets
        adata_rna_train = adata_dict['adata_rna_train']
        adata_niche_train = adata_dict['adata_niche_train'] 
        adata_activity_train = adata_dict['adata_activity_train']
        adata_rna_test = adata_dict['adata_rna_test']
        adata_niche_test = adata_dict['adata_niche_test']
        adata_activity_test = adata_dict['adata_activity_test']
        
        print(f"Loaded training data shapes:")
        print(f"  RNA: {adata_rna_train.shape}")
        print(f"  Niche: {adata_niche_train.shape}")
        print(f"  Activity: {adata_activity_train.shape}")
        
        # Add split labels
        print("\n=== Preprocessing Data ===")
        adata_rna_train = change_label(adata_rna_train, 'train')
        adata_niche_train = change_label(adata_niche_train, 'train')
        adata_activity_train = change_label(adata_activity_train, 'train')
        adata_rna_test = change_label(adata_rna_test, 'test')
        adata_niche_test = change_label(adata_niche_test, 'test')
        adata_activity_test = change_label(adata_activity_test, 'test')
        
        # Collect modalities into aligned lists
        adatas_train = [adata_rna_train, adata_niche_train, adata_activity_train]
        adatas_test = [adata_rna_test, adata_niche_test, adata_activity_test]
        
        # Merge train+test per modality
        adatas_all = []
        for ad_train, ad_test in zip(adatas_train, adatas_test):
            ad_all = ad_train.concatenate(ad_test, batch_key='sample')
            ad_all = change_label(ad_all, 'test')  # Keep your original pattern
            adatas_all.append(ad_all)
        
        # Convert to dense float32
        adatas_train = [ensure_dense_float32(ad.copy()) for ad in adatas_train]
        adatas_all = [ensure_dense_float32(ad.copy()) for ad in adatas_all]
        
        print("Data preprocessing completed")
        
        # Auto-sync dimensions with actual data
        print("\n=== Configuring Model Architecture ===")
        for i, ad in enumerate(adatas_train):
            in_dim = ad.X.shape[1]
            config["encoders"][i]["input"] = in_dim
            config["decoders"][i]["output"] = in_dim  
            config["discriminators"][i]["input"] = in_dim
            print(f"Modality {i}: {in_dim} features")
        
        print("Model architecture configured")
        
        # Save intermediate results if requested
        if args.save_intermediates:
            print("Saving intermediate preprocessing results...")
            for i, (name, adata) in enumerate(zip(['rna', 'niche', 'activity'], adatas_train)):
                adata.write(output_dir / f"preprocessed_{name}_train.h5ad")
            for i, (name, adata) in enumerate(zip(['rna', 'niche', 'activity'], adatas_all)):
                adata.write(output_dir / f"preprocessed_{name}_all.h5ad")
        
        # Create and train model
        print(f"\n=== {'Training' if args.train_model else 'Loading'} Model ===")
        model = UnitedNet(root_save_path, device=args.device, technique=config)
        
        if args.train_model:
            print("Starting training phase...")
            model.train(adatas_train, verbose=args.verbose)
            
            print("Starting finetuning phase...")
            model.finetune(adatas_all, verbose=args.verbose)
            
            print("Training completed successfully!")
        else:
            model_path = f"{root_save_path}/train_best.pt"
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            print(f"Loading model from {model_path}")
            model.load_model(model_path, device=args.device)
        
        # Save the trained model
        model_filename = f"model_perturbmap_{args.dataset_id}"
        if args.model_suffix:
            model_filename += f"_{args.model_suffix}"
        if timestamp:
            model_filename += f"_{timestamp}"
        model_filename += ".pkl"
        
        model_path = output_dir / model_filename
        print(f"\n=== Saving Model ===")
        print(f"Saving model to {model_path}")
        
        with open(model_path, 'wb') as file:
            pickle.dump(model, file)
        
        # Save training summary
        summary = {
            'dataset_id': args.dataset_id,
            'technique': args.technique,
            'device': args.device,
            'timestamp': timestamp,
            'train_epochs': args.train_epochs,
            'finetune_epochs': args.finetune_epochs,
            'n_clusters': args.n_clusters,
            'config_source': config_source,
            'data_shapes': {
                'rna': list(adata_rna_train.shape),
                'niche': list(adata_niche_train.shape), 
                'activity': list(adata_activity_train.shape)
            },
            'model_path': str(model_path),
            'config_path': str(output_dir / "config.json")
        }
        
        summary_path = output_dir / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Training summary saved to {summary_path}")
        print(f"\n=== Training Complete ===")
        print(f"Model saved: {model_path}")
        print(f"Output directory: {output_dir}")
        
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())