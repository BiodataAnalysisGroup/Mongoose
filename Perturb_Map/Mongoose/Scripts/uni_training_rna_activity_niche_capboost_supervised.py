#!/usr/bin/env python3
"""
UnitedNet PerturbMap — RNA + Activity + Niche (CapBoost Supervised v2)

IMPROVEMENTS:
- Log-normalization for RNA modality
- Highly variable gene selection option
- Increased RNA noise regularization
- Deeper RNA encoder architecture
- Better CMP weight balancing
- Optimized learning rate and epochs

CLI example
-----------
python uni_training_rna_activity_niche_capboost_supervised_v2.py \
  --data_path ../Data/UnitedNet/input_data \
  --dataset_id KP2_1 \
  --train_epochs 60 --finetune_epochs 30 \
  --lr 1e-4 \
  --scale_rna --scale_activity --scale_niche \
  --log_normalize_rna \
  --use_hvg --n_hvg 2000 \
  --n_clusters 6 \
  --timestamp --verbose
"""

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch

# UnitedNet imports
from unitednet.interface import UnitedNet
from unitednet.data import save_obj, load_obj, type_specific_mean

# ---------- small helpers ----------

def _pad_to(n, seq):
    lst = list(seq) if seq is not None else []
    return lst + [None]*(n - len(lst))

def _get(seq, i):
    try:
        return seq[i]
    except Exception:
        return None

def _normalize_train_return(train_ret, model):
    if isinstance(train_ret, (list, tuple)) and len(train_ret) >= 4:
        return train_ret[0], train_ret[1], train_ret[2], train_ret[3]
    if isinstance(train_ret, dict):
        loss = train_ret.get("loss") or train_ret.get("total_loss")
        return (loss,
                train_ret.get("acc") or train_ret.get("accuracy"),
                train_ret.get("ari"),
                train_ret.get("nmi"))
    return (getattr(model, "train_total_loss_list", None),
            getattr(model, "train_acc_list", None),
            getattr(model, "train_ari_list", None),
            getattr(model, "train_nmi_list", None))

def _normalize_ft_return(ft_ret, model):
    r2_seq = None
    if isinstance(ft_ret, (list, tuple)) and len(ft_ret) >= 4:
        if len(ft_ret) >= 5:
            r2_seq = ft_ret[4]
        return ft_ret[0], ft_ret[1], ft_ret[2], ft_ret[3], r2_seq
    if isinstance(ft_ret, dict):
        r2_seq = ft_ret.get("r2") or ft_ret.get("R2")
        return (ft_ret.get("loss"),
                ft_ret.get("acc"),
                ft_ret.get("ari"),
                ft_ret.get("nmi"),
                r2_seq)
    return (getattr(model, "finetune_total_loss_list", None),
            getattr(model, "finetune_acc_list", None),
            getattr(model, "finetune_ari_list", None),
            getattr(model, "finetune_nmi_list", None),
            getattr(model, "r2_per_epoch", None))

class MetricHistory:
    def __init__(self):
        self.hist = {
            "epoch": [],
            "train_acc": [], "train_nmi": [], "train_ari": [], "train_loss": [],
            "ft_acc": [],    "ft_nmi": [],    "ft_ari": [],    "ft_loss": [],
            "r2_mean": [],
        }

    def load_from_train(self, lists):
        tl, acc, ari, nmi = lists
        tl = tl or []; acc = acc or []; ari = ari or []; nmi = nmi or []
        n = max([len(x) for x in [tl, acc, ari, nmi] if x] or [1])
        self.hist["epoch"] = list(range(1, n+1))
        self.hist["train_loss"] = _pad_to(n, tl)
        self.hist["train_acc"]  = _pad_to(n, acc)
        self.hist["train_ari"]  = _pad_to(n, ari)
        self.hist["train_nmi"]  = _pad_to(n, nmi)
        self.hist["ft_acc"]  = [None]*n
        self.hist["ft_nmi"]  = [None]*n
        self.hist["ft_ari"]  = [None]*n
        self.hist["ft_loss"] = [None]*n
        self.hist["r2_mean"] = [None]*n

    def merge_finetune(self, lists, r2_matrix_per_epoch=None):
        fl, acc, ari, nmi = lists
        fl = fl or []; acc = acc or []; ari = ari or []; nmi = nmi or []
        n0 = len(self.hist["epoch"])
        nf = max([len(x) for x in [fl, acc, ari, nmi] if x] or [0])
        if nf > n0:
            add = list(range(n0+1, n0+nf+1))
            self.hist["epoch"].extend(add)
            for k in ["train_acc","train_nmi","train_ari","train_loss",
                      "ft_acc","ft_nmi","ft_ari","ft_loss","r2_mean"]:
                self.hist[k].extend([None]*len(add))
        if nf > 0:
            start = len(self.hist["epoch"]) - nf
            for i in range(nf):
                idx = start + i
                self.hist["ft_loss"][idx] = _get(fl, i)
                self.hist["ft_acc"][idx]  = _get(acc, i)
                self.hist["ft_ari"][idx]  = _get(ari, i)
                self.hist["ft_nmi"][idx]  = _get(nmi, i)
                if r2_matrix_per_epoch is not None:
                    r2m = _get(r2_matrix_per_epoch, i)
                    if r2m is not None:
                        try:
                            self.hist["r2_mean"][idx] = float(
                                np.nanmean(np.array(r2m, dtype=float))
                            )
                        except Exception:
                            self.hist["r2_mean"][idx] = None

    def save(self, out_dir: Union[str, Path], meta: dict):
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.hist).to_csv(out/"history.csv", index=False)

        def last_valid(key):
            vals = [v for v in self.hist[key] if v is not None]
            return float(vals[-1]) if vals else None

        final = {
            "lr": meta.get("lr"),
            "train_acc":  last_valid("train_acc"),
            "train_nmi":  last_valid("train_nmi"),
            "train_ari":  last_valid("train_ari"),
            "train_loss": last_valid("train_loss"),
            "ft_acc":     last_valid("ft_acc"),
            "ft_nmi":     last_valid("ft_nmi"),
            "ft_ari":     last_valid("ft_ari"),
            "ft_loss":    last_valid("ft_loss"),
            "mean_R2":    last_valid("r2_mean"),
            **{k: v for k, v in meta.items() if k not in ["lr"]},
        }
        json.dump(final, open(out/"final_metrics.json","w"), indent=2)

# ---------- utils ----------

def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_dense_float32(adata):
    if sp.issparse(adata.X):
        X = adata.X.tocsr()
        X.sort_indices()
        adata.X = X.astype(np.float32).toarray()
    else:
        adata.X = np.asarray(adata.X, dtype=np.float32)
    return adata

def change_label(adata, split_name):
    adata = adata.copy()
    adata.obs['split'] = split_name
    if 'array_col' in adata.obs.columns:
        adata.obs['imagecol'] = adata.obs['array_col']
    if 'array_row' in adata.obs.columns:
        adata.obs['imagerow'] = adata.obs['array_row']
    if 'phenotypes' in adata.obs.columns:
        s = adata.obs['phenotypes']
        if not pd.api.types.is_categorical_dtype(s):
            s = pd.Series(pd.Categorical(s), index=adata.obs.index)
        else:
            s = pd.Series(s, index=adata.obs.index)
        codes = s.cat.codes.replace(-1, 0).astype(np.int64)
        adata.obs['label'] = codes
    return adata

def set_global_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_config(config, output_dir: Path, filename="config.json"):
    path = output_dir / filename
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to {path}")
    return path

def setup_output_directory(base_path, technique, timestamp_flag):
    ts = get_timestamp() if timestamp_flag else ""
    out = Path(base_path) / (f"{technique}_{ts}" if ts else technique)
    out.mkdir(parents=True, exist_ok=True)
    return out

def load_adata_files(data_path, dataset_id):
    fn = lambda stem: Path(data_path) / f"{stem}_perturbmap_{dataset_id}.h5ad"
    wanted = {
        "rna_train": fn("adata_rna_train"),
        "rna_test":  fn("adata_rna_test"),
        "activity_train": fn("adata_activity_train"),
        "activity_test":  fn("adata_activity_test"),
        "niche_train":    fn("adata_niche_train"),
        "niche_test":     fn("adata_niche_test"),
    }
    out = {}
    for k, p in wanted.items():
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")
        print(f"Loading {k}: {p}")
        out[k] = sc.read_h5ad(p)
    return out

def assert_same_obs(*adatas, name_hint=""):
    base = adatas[0].obs_names
    for i, ad in enumerate(adatas[1:], 1):
        if not base.equals(ad.obs_names):
            raise ValueError(f"Obs misalignment {name_hint}: adata[0] vs adata[{i}]")
    return True

def concat_train_test(ad_train, ad_test):
    if not ad_train.var_names.equals(ad_test.var_names):
        common = ad_train.var_names.intersection(ad_test.var_names)
        if len(common) == 0:
            raise ValueError("No common features between train and test for a modality.")
        ad_train = ad_train[:, common].copy()
        ad_test  = ad_test[:, common].copy()
        ad_test  = ad_test[:, ad_train.var_names].copy()
    ad_train = change_label(ad_train, "train")
    ad_test  = change_label(ad_test,  "test")
    ad_all   = ad_train.concatenate(ad_test, join="inner", batch_key="batch",
                                    batch_categories=["train","test"])
    return ad_all

def zscore_from_train(train_ad, apply_ad_list, out_dir: Path, prefix: str):
    X = np.asarray(train_ad.X, dtype=np.float32)
    mu = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    for ad in apply_ad_list:
        ad.X = (np.asarray(ad.X, dtype=np.float32) - mu) / std
    np.save(out_dir / f"{prefix}_mean.npy", mu.astype(np.float32))
    np.save(out_dir / f"{prefix}_std.npy",  std.astype(np.float32))
    print(f"Saved scaling params: {prefix}_mean.npy / {prefix}_std.npy")
    return mu, std

def preprocess_rna(rna_tr, rna_te, out_dir: Path, 
                   log_normalize=True, use_hvg=False, n_hvg=2000):
    """
    Preprocess RNA modality with optional log-normalization and HVG selection.
    """
    print("\n=== RNA Preprocessing ===")
    
    if log_normalize:
        print("Log-normalizing RNA (target_sum=1e4)...")
        for adata in [rna_tr, rna_te]:
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
        print("  ✓ Log-normalization complete")
    
    if use_hvg:
        print(f"Selecting top {n_hvg} highly variable genes...")
        # Calculate HVG on training set only
        sc.pp.highly_variable_genes(rna_tr, n_top_genes=n_hvg, flavor='seurat_v3')
        hvg_genes = rna_tr.var_names[rna_tr.var.highly_variable].tolist()
        
        # Subset both train and test
        rna_tr = rna_tr[:, hvg_genes].copy()
        rna_te = rna_te[:, hvg_genes].copy()
        
        # Save HVG list
        pd.Series(hvg_genes).to_csv(out_dir / "rna_hvg_genes.csv", index=False)
        print(f"  ✓ Selected {len(hvg_genes)} HVGs (saved to rna_hvg_genes.csv)")
    
    return rna_tr, rna_te

# ---------- supervised CapBoost config v2 ----------

def get_capboost_supervised_config_v2():
    """
    IMPROVED config with:
    - Deeper RNA encoder (768→512→256→128→64)
    - Higher dropout on RNA for regularization
    - Balanced CMP edge weights
    """
    return {
        'train_batch_size': 16,
        'finetune_batch_size': 16,
        'transfer_batch_size': None,
        'train_epochs': 60,
        'finetune_epochs': 30,
        'transfer_epochs': None,

        # Supervised setup
        'train_task': 'supervised_group_identification',
        'finetune_task': 'cross_model_prediction_clas',
        'transfer_task': None,

        # Let UnitedNet decide supervised loss weights
        'train_loss_weight': None,
        'finetune_loss_weight': None,
        'transfer_loss_weight': None,

        # Balanced edge-specific CMP weights
        'cmp_edge_weights': {
            'Activity->Activity': 2.0,  # Emphasize activity self-consistency
            'RNA->Niche': 1.0,
            'Activity->Niche': 1.0,
        },

        'lr': 0.0001,  # Lower LR for finer optimization
        'checkpoint': 20,
        'n_head': 10,
        'noise_level': [0.05, 0.02, 0.0],   # RNA, Activity, Niche (higher RNA noise)
        'fuser_type': 'WeightedMean',

        'encoders': [
            # RNA encoder - DEEPER with more regularization
            {'input': 2000, 'hiddens':[768, 512, 256, 128, 64], 'output': 64,
             'use_biases':[True]*6,
             'dropouts':[0.1, 0.1, 0.05, 0.05, 0.0, 0.0],  # Higher dropout
             'activations':['relu','relu','relu','relu','relu',None],
             'use_batch_norms':[False]*6,
             'use_layer_norms':[False,False,False,False,False,True],
             'is_binary_input': False},
            # Activity encoder
            {'input': 1500, 'hiddens':[512, 256, 128, 64], 'output': 64,
             'use_biases':[True]*5,
             'dropouts':[0.05, 0.05, 0.0, 0.0, 0.0],
             'activations':['relu','relu','relu','relu',None],
             'use_batch_norms':[False]*5,
             'use_layer_norms':[False,False,False,False,True],
             'is_binary_input': False},
            # Niche encoder (CapBoost widened)
            {'input': 1500, 'hiddens':[768, 256, 128, 64], 'output': 64,
             'use_biases':[True]*5,
             'dropouts':[0.05, 0.03, 0.0, 0.0, 0.0],
             'activations':['relu','relu','relu','relu',None],
             'use_batch_norms':[False]*5,
             'use_layer_norms':[True, False, False, False, True],
             'is_binary_input': False},
        ],
        'latent_projector': None,

        'decoders': [
            # RNA decoder - matches deeper encoder
            {'input': 64, 'hiddens':[64, 128, 256, 512, 768], 'output': 2000,
             'use_biases':[True]*6,
             'dropouts':[0.0, 0.0, 0.0, 0.05, 0.05, 0.0],
             'activations':['relu','relu','relu','relu','relu',None],
             'use_batch_norms':[False]*6,
             'use_layer_norms':[True,False,False,False,False,False]},
            # Activity decoder
            {'input': 64, 'hiddens':[64, 128, 256, 512], 'output': 1500,
             'use_biases':[True]*5,
             'dropouts':[0.0, 0.0, 0.05, 0.05, 0.0],
             'activations':['relu','relu','relu','relu',None],
             'use_batch_norms':[False]*5,
             'use_layer_norms':[True,False,False,False,False]},
            # Niche decoder
            {'input': 64, 'hiddens':[64, 128, 256, 512], 'output': 1500,
             'use_biases':[True]*5,
             'dropouts':[0.0, 0.0, 0.03, 0.03, 0.0],
             'activations':['relu','relu','relu','relu',None],
             'use_batch_norms':[False]*5,
             'use_layer_norms':[True,False,False,False,False]},
        ],

        'discriminators': [
            {'input': 2000, 'hiddens':[768, 512, 256, 128], 'output': 1,
             'use_biases':[True]*5,
             'dropouts':[0.1, 0.1, 0.1, 0.0, 0.0],
             'activations':['relu','relu','relu','relu','sigmoid'],
             'use_batch_norms':[False]*5,
             'use_layer_norms':[False,False,False,False,True]},
            {'input': 1500, 'hiddens':[512, 256, 128], 'output': 1,
             'use_biases':[True]*4,
             'dropouts':[0.1, 0.1, 0.0, 0.0],
             'activations':['relu','relu','relu','sigmoid'],
             'use_batch_norms':[False]*4,
             'use_layer_norms':[False,False,False,True]},
            {'input': 1500, 'hiddens':[512, 256, 128], 'output': 1,
             'use_biases':[True]*4,
             'dropouts':[0.1, 0.1, 0.0, 0.0],
             'activations':['relu','relu','relu','sigmoid'],
             'use_batch_norms':[False]*4,
             'use_layer_norms':[False,False,False,True]},
        ],

        'projectors': {
            'input': 64, 'hiddens': [], 'output': 100,
            'use_biases': [True],
            'dropouts': [0.0],
            'activations': ['relu'],
            'use_batch_norms': [False],
            'use_layer_norms': [True],
        },

        'clusters': {
            'input': 100, 'hiddens': [], 'output': 6,  # will be overridden by args.n_clusters
            'use_biases': [False],
            'dropouts': [0.0],
            'activations': [None],
            'use_batch_norms': [False],
            'use_layer_norms': [False],
        },
    }

# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(
        description="Train UnitedNet PerturbMap - RNA + Activity + Niche (CapBoost Supervised v2 - Improved)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Data
    parser.add_argument('--data_path', type=str, default='../Data/UnitedNet/input_data')
    parser.add_argument('--dataset_id', type=str, default='KP2_1')
    # Model
    parser.add_argument('--technique', type=str,
                        default='perturbmap_rna_activity_niche_capboost_supervised_v2',
                        choices=['perturbmap_rna_activity_niche_capboost_supervised_v2'])
    parser.add_argument('--device', type=str, default='cuda:0')
    # Train
    parser.add_argument('--train_epochs', type=int, default=60)
    parser.add_argument('--finetune_epochs', type=int, default=30)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--finetune_batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--n_clusters', type=int, default=6,
                        help="Number of supervised classes (phenotypes)")
    parser.add_argument('--noise_level', type=float, nargs=3, default=[0.05, 0.02, 0.0])
    parser.add_argument('--seed', type=int, default=42)
    # RNA preprocessing (NEW)
    parser.add_argument('--log_normalize_rna', action='store_true', default=True,
                        help='Apply log-normalization to RNA (highly recommended)')
    parser.add_argument('--no_log_normalize_rna', dest='log_normalize_rna', action='store_false')
    parser.add_argument('--use_hvg', action='store_true', default=False,
                        help='Use only highly variable genes for RNA')
    parser.add_argument('--n_hvg', type=int, default=2000,
                        help='Number of HVGs to select if --use_hvg is set')
    # Scaling
    parser.add_argument('--scale_rna', action='store_true', default=True)
    parser.add_argument('--no_scale_rna', dest='scale_rna', action='store_false')
    parser.add_argument('--scale_activity', action='store_true', default=True)
    parser.add_argument('--no_scale_activity', dest='scale_activity', action='store_false')
    parser.add_argument('--scale_niche', action='store_true', default=True)
    parser.add_argument('--no_scale_niche', dest='scale_niche', action='store_false')
    # Output
    parser.add_argument('--output_base', type=str, default='../Models/UnitedNet')
    parser.add_argument('--model_suffix', type=str, default='')
    parser.add_argument('--timestamp', action='store_true')
    parser.add_argument('--save_intermediates', action='store_true')
    # Flow
    parser.add_argument('--train_model', action='store_true', default=True)
    parser.add_argument('--no_train', dest='train_model', action='store_false')
    parser.add_argument('--verbose', action='store_true', default=True)
    args = parser.parse_args()

    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print("⚠️  CUDA requested but not available → using CPU")
        args.device = 'cpu'

    out_dir = setup_output_directory(args.output_base, args.technique, args.timestamp)
    print(f"Output directory: {out_dir}")

    # Build supervised config v2
    config = get_capboost_supervised_config_v2()
    config['train_epochs'] = args.train_epochs
    config['finetune_epochs'] = args.finetune_epochs
    config['train_batch_size'] = args.train_batch_size
    config['finetune_batch_size'] = args.finetune_batch_size
    config['lr'] = args.lr
    config['clusters']['output'] = args.n_clusters
    config['noise_level'] = [
        float(args.noise_level[0]),
        float(args.noise_level[1]),
        float(args.noise_level[2])
    ]

    save_config(config, out_dir)

    set_global_seed(args.seed)
    print(f"Seed set to {args.seed}. Device: {args.device}")

    # Load data
    print("\n=== Loading Data (RNA + Activity + Niche) ===")
    d = load_adata_files(args.data_path, args.dataset_id)
    rna_tr,      rna_te      = d['rna_train'], d['rna_test']
    activity_tr, activity_te = d['activity_train'], d['activity_test']
    niche_tr,    niche_te    = d['niche_train'], d['niche_test']

    # Align obs
    print("\n=== Validating Alignment ===")
    assert_same_obs(rna_tr, activity_tr, name_hint="train RNA vs Activity")
    assert_same_obs(rna_te, activity_te, name_hint="test RNA vs Activity")
    assert_same_obs(rna_tr, niche_tr, name_hint="train RNA vs Niche")
    assert_same_obs(rna_te, niche_te, name_hint="test RNA vs Niche")

    # Harmonize features within each modality
    for name, a, b in [("RNA", rna_tr, rna_te),
                       ("Activity", activity_tr, activity_te),
                       ("Niche", niche_tr, niche_te)]:
        if not a.var_names.equals(b.var_names):
            common = a.var_names.intersection(b.var_names)
            if len(common) == 0:
                raise ValueError(f"No common features between {name} train/test.")
            a = a[:, common].copy()
            b = b[:, a.var_names].copy()
            if name == "RNA":
                rna_tr, rna_te = a, b
            elif name == "Activity":
                activity_tr, activity_te = a, b
            else:
                niche_tr, niche_te = a, b
            print(f"⚠️  Harmonized {name} features to intersection ({len(common)})")

    # Preprocess RNA (log-normalization + optional HVG)
    rna_tr, rna_te = preprocess_rna(
        rna_tr, rna_te, out_dir,
        log_normalize=args.log_normalize_rna,
        use_hvg=args.use_hvg,
        n_hvg=args.n_hvg
    )

    # Labels
    print("\n=== Applying Split Labels ===")
    rna_tr      = change_label(rna_tr,      'train'); rna_te      = change_label(rna_te,      'test')
    activity_tr = change_label(activity_tr, 'train'); activity_te = change_label(activity_te, 'test')
    niche_tr    = change_label(niche_tr,    'train'); niche_te    = change_label(niche_te,    'test')

    # Ensure dense before scaling
    print("\n=== Converting to dense (required for scaling) ===")
    rna_tr = ensure_dense_float32(rna_tr)
    rna_te = ensure_dense_float32(rna_te)
    activity_tr = ensure_dense_float32(activity_tr)
    activity_te = ensure_dense_float32(activity_te)
    niche_tr = ensure_dense_float32(niche_tr)
    niche_te = ensure_dense_float32(niche_te)

    # Scaling
    print("\n=== Feature scaling ===")
    if args.scale_rna:
        zscore_from_train(rna_tr, [rna_tr, rna_te], out_dir, prefix="rna_zscore")
    if args.scale_activity:
        zscore_from_train(activity_tr, [activity_tr, activity_te], out_dir, prefix="activity_zscore")
    if args.scale_niche:
        zscore_from_train(niche_tr, [niche_tr, niche_te], out_dir, prefix="niche_zscore")

    # Build concatenated sets
    print("\n=== Building Train & All Sets (RNA + Activity + Niche) ===")
    rna_all      = concat_train_test(rna_tr,      rna_te)
    activity_all = concat_train_test(activity_tr, activity_te)
    niche_all    = concat_train_test(niche_tr,    niche_te)

    # Ensure dense for concatenated sets
    rna_all = ensure_dense_float32(rna_all)
    activity_all = ensure_dense_float32(activity_all)
    niche_all = ensure_dense_float32(niche_all)

    train_list = [rna_tr.copy(), activity_tr.copy(), niche_tr.copy()]
    all_list   = [rna_all.copy(), activity_all.copy(), niche_all.copy()]

    # Overwrite I/O dims from data
    print("\n=== Configuring Model I/O from data (CapBoost Supervised v2) ===")
    modality_names = ['RNA','Activity','Niche']
    for i, adata in enumerate(train_list):
        in_dim = adata.X.shape[1]
        config['encoders'][i]['input']       = in_dim
        config['decoders'][i]['output']      = in_dim
        config['discriminators'][i]['input'] = in_dim
        print(f"Modality {i} ({modality_names[i]}): input/output = {in_dim}")

    save_config(config, out_dir, filename="config_resolved.json")

    # Train / load
    print(f"\n=== {'Training' if args.train_model else 'Loading'} UnitedNet (CapBoost Supervised v2) ===")
    model = UnitedNet(str(out_dir), device=args.device, technique=config)

    # Ensure the internal save_path is set
    if not hasattr(model, 'save_path') or model.save_path in (None, ''):
        model.save_path = str(out_dir)

    mh = MetricHistory()
    if args.train_model:
        print("Training...")
        train_ret = model.train(train_list, verbose=args.verbose)
        tr_loss, tr_acc, tr_ari, tr_nmi = _normalize_train_return(train_ret, model)
        mh.load_from_train((tr_loss, tr_acc, tr_ari, tr_nmi))

        print("Finetuning (train+test concatenated per modality)...")
        ft_ret = model.finetune(all_list, verbose=args.verbose)
        ft_loss, ft_acc, ft_ari, ft_nmi, r2_seq = _normalize_ft_return(ft_ret, model)
        mh.merge_finetune((ft_loss, ft_acc, ft_ari, ft_nmi), r2_matrix_per_epoch=r2_seq)

        print("Training complete.")
    else:
        best = Path(out_dir) / "train_best.pt"
        if not best.exists():
            raise FileNotFoundError(f"Model weights not found: {best}")
        print(f"Loading model from {best}")
        model.load_model(str(best), device=args.device)

    # Save outputs
    model_fn = f"model_perturbmap_rna_activity_niche_capboost_supervised_v2_{args.dataset_id}"
    if args.model_suffix:
        model_fn += f"_{args.model_suffix}"
    if args.timestamp:
        model_fn += f"_{get_timestamp()}"
    model_path = out_dir / f"{model_fn}.pkl"

    mh.save(out_dir, meta={
        "lr": args.lr,
        "dataset_id": args.dataset_id,
        "n_clusters": args.n_clusters,
        "technique": args.technique,
        "device": args.device,
        "modalities": "RNA+Activity+Niche",
        "log_normalize_rna": bool(args.log_normalize_rna),
        "use_hvg": bool(args.use_hvg),
        "n_hvg": args.n_hvg if args.use_hvg else None,
        "scale_rna": bool(args.scale_rna),
        "scale_activity": bool(args.scale_activity),
        "scale_niche": bool(args.scale_niche),
    })

    print("\n=== Saving Model & Summary ===")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved model to {model_path}")

    summary = {
        "dataset_id": args.dataset_id,
        "technique": args.technique,
        "device": args.device,
        "modalities": "RNA+Activity+Niche",
        "train_epochs": args.train_epochs,
        "finetune_epochs": args.finetune_epochs,
        "train_batch_size": args.train_batch_size,
        "finetune_batch_size": args.finetune_batch_size,
        "lr": args.lr,
        "n_clusters": args.n_clusters,
        "noise_level": config['noise_level'],
        "out_dir": str(out_dir),
        "model_path": str(model_path),
        "config_path": str(out_dir / "config_resolved.json"),
        "preprocessing": {
            "log_normalize_rna": bool(args.log_normalize_rna),
            "use_hvg": bool(args.use_hvg),
            "n_hvg": args.n_hvg if args.use_hvg else None,
        },
        "scaling": {
            "rna": bool(args.scale_rna),
            "activity": bool(args.scale_activity),
            "niche": bool(args.scale_niche),
        },
        "cmp_edge_weights": config.get('cmp_edge_weights'),
        "shapes": {
            "rna_train": list(rna_tr.shape),
            "activity_train": list(activity_tr.shape),
            "niche_train": list(niche_tr.shape),
            "rna_all": list(rna_all.shape),
            "activity_all": list(activity_all.shape),
            "niche_all": list(niche_all.shape),
        },
    }
    with open(out_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {out_dir/'training_summary.json'}")

    print("\n=== V2 IMPROVEMENTS ===")
    print("✓ Log-normalization for RNA (default ON)")
    print("✓ Optional HVG selection (use --use_hvg)")
    print("✓ Deeper RNA encoder: 768→512→256→128→64")
    print("✓ Higher RNA noise: 0.05 (up from 0.02)")
    print("✓ More dropout on RNA: [0.1, 0.1, 0.05, 0.05, 0.0, 0.0]")
    print("✓ Lower learning rate: 0.0001 (down from 0.0005)")
    print("✓ More epochs: train=60, finetune=30")
    print("✓ Balanced CMP weights: Activity→Activity=2.0, others=1.0")

    print("\n=== Done ===")
    return model

if __name__ == "__main__":
    main()