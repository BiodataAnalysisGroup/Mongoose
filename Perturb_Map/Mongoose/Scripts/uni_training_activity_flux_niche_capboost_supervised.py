#!/usr/bin/env python3
"""
UnitedNet PerturbMap — Activity + Flux + Niche (CapBoost Supervised)

Supervised variant of the CapBoost v1 script.

Key differences from the unsupervised clustering version:
- train_task      = "supervised_group_identification"
- finetune_task   = "cross_model_prediction_clas"
- train_loss_weight / finetune_loss_weight = None (use UnitedNet defaults)
- clusters['output'] = n_clusters = number of phenotype classes (e.g. 4)

CLI example
-----------
python uni_training_activity_flux_niche_capboost_supervised.py \
  --data_path ../Data/UnitedNet/input_data \
  --dataset_id KP2_1 \
  --train_epochs 40 --finetune_epochs 20 \
  --lr 5e-4 \
  --scale_flux --scale_activity --scale_niche \
  --n_clusters 4 \
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
        "activity_train": fn("adata_activity_train"),
        "activity_test":  fn("adata_activity_test"),
        "flux_train":     fn("adata_flux_train"),
        "flux_test":      fn("adata_flux_test"),
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

# ---------- supervised CapBoost config ----------

def get_capboost_supervised_config():
    """
    Three-modality config with Niche-widened branches, but using
    supervised tasks:

    - train_task    = 'supervised_group_identification'
    - finetune_task = 'cross_model_prediction_clas'
    - train_loss_weight / finetune_loss_weight = None (use UnitedNet defaults)
    """
    return {
        'train_batch_size': 16,
        'finetune_batch_size': 16,
        'transfer_batch_size': None,
        'train_epochs': 40,
        'finetune_epochs': 20,
        'transfer_epochs': None,

        # Supervised setup
        'train_task': 'supervised_group_identification',
        'finetune_task': 'cross_model_prediction_clas',
        'transfer_task': None,

        # Let UnitedNet decide supervised loss weights
        'train_loss_weight': None,
        'finetune_loss_weight': None,
        'transfer_loss_weight': None,

        # Optional edge-specific CMP weights
        'cmp_edge_weights': None,

        'lr': 0.0005,
        'checkpoint': 20,
        'n_head': 10,
        'noise_level': [0.0, 0.02, 0.0],   # Activity, Flux, Niche
        'fuser_type': 'WeightedMean',

        'encoders': [
            # Activity encoder
            {'input': 1500, 'hiddens':[512, 256, 128, 64], 'output': 64,
             'use_biases':[True]*5,
             'dropouts':[0.05, 0.05, 0.0, 0.0, 0.0],
             'activations':['relu','relu','relu','relu',None],
             'use_batch_norms':[False]*5,
             'use_layer_norms':[False,False,False,False,True],
             'is_binary_input': False},
            # Flux encoder
            {'input': 70, 'hiddens':[128, 64], 'output': 64,
             'use_biases':[True, True, True],
             'dropouts':[0.0, 0.0, 0.0],
             'activations':['relu','relu',None],
             'use_batch_norms':[False, False, False],
             'use_layer_norms':[False, False, True],
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
            # Activity decoder
            {'input': 64, 'hiddens':[64, 128, 256, 512], 'output': 1500,
             'use_biases':[True]*5,
             'dropouts':[0.0, 0.0, 0.05, 0.05, 0.0],
             'activations':['relu','relu','relu','relu',None],
             'use_batch_norms':[False]*5,
             'use_layer_norms':[True,False,False,False,False]},
            # Flux decoder
            {'input': 64, 'hiddens':[128], 'output': 70,
             'use_biases':[True, True],
             'dropouts':[0.0, 0.0],
             'activations':['relu',None],
             'use_batch_norms':[False, False],
             'use_layer_norms':[True, False]},
            # Niche decoder
            {'input': 64, 'hiddens':[64, 128, 256, 512], 'output': 1500,
             'use_biases':[True]*5,
             'dropouts':[0.0, 0.0, 0.03, 0.03, 0.0],
             'activations':['relu','relu','relu','relu',None],
             'use_batch_norms':[False]*5,
             'use_layer_norms':[True,False,False,False,False]},
        ],

        'discriminators': [
            {'input': 1500, 'hiddens':[512, 256, 128], 'output': 1,
             'use_biases':[True]*4,
             'dropouts':[0.1, 0.1, 0.0, 0.0],
             'activations':['relu','relu','relu','sigmoid'],
             'use_batch_norms':[False]*4,
             'use_layer_norms':[False,False,False,True]},
            {'input': 70, 'hiddens':[128, 64], 'output': 1,
             'use_biases':[True]*3,
             'dropouts':[0.1, 0.0, 0.0],
             'activations':['relu','relu','sigmoid'],
             'use_batch_norms':[False]*3,
             'use_layer_norms':[False,False,True]},
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
            'input': 100, 'hiddens': [], 'output': 4,  # will be overridden by args.n_clusters
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
        description="Train UnitedNet PerturbMap - Activity + Flux + Niche (CapBoost Supervised)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Data
    parser.add_argument('--data_path', type=str, default='../Data/UnitedNet/input_data')
    parser.add_argument('--dataset_id', type=str, default='KP2_1')
    # Model
    parser.add_argument('--technique', type=str,
                        default='perturbmap_activity_flux_niche_capboost_supervised',
                        choices=['perturbmap_activity_flux_niche_capboost_supervised'])
    parser.add_argument('--device', type=str, default='cuda:0')
    # Train
    parser.add_argument('--train_epochs', type=int, default=40)
    parser.add_argument('--finetune_epochs', type=int, default=20)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--finetune_batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--n_clusters', type=int, default=4,
                        help="Number of supervised classes (phenotypes)")
    parser.add_argument('--noise_level', type=float, nargs=3, default=[0.0, 0.02, 0.0])
    parser.add_argument('--seed', type=int, default=42)
    # Scaling
    parser.add_argument('--scale_flux', action='store_true', default=True)
    parser.add_argument('--no_scale_flux', dest='scale_flux', action='store_false')
    parser.add_argument('--scale_activity', action='store_true', default=False)
    parser.add_argument('--scale_niche', action='store_true', default=False)
    # Edge-specific CMP (optional; requires support in UnitedNet)
    parser.add_argument('--cmp_edge_activity_to_niche', type=float, default=0.0,
                        help='Extra weight multiplier for Activity→Niche CMP (0 = off)')
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

    # Build supervised config
    config = get_capboost_supervised_config()
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

    # Optional edge-specific CMP weights
    if args.cmp_edge_activity_to_niche and args.cmp_edge_activity_to_niche > 0:
        config['cmp_edge_weights'] = {"Activity->Niche": float(args.cmp_edge_activity_to_niche)}
        print(f"Edge-specific CMP enabled: Activity→Niche x{args.cmp_edge_activity_to_niche}")

    save_config(config, out_dir)

    set_global_seed(args.seed)
    print(f"Seed set to {args.seed}. Device: {args.device}")

    # Load data
    print("\n=== Loading Data (Activity + Flux + Niche) ===")
    d = load_adata_files(args.data_path, args.dataset_id)
    activity_tr, activity_te = d['activity_train'], d['activity_test']
    flux_tr,     flux_te     = d['flux_train'],     d['flux_test']
    niche_tr,    niche_te    = d['niche_train'],    d['niche_test']

    # Align obs
    print("\n=== Validating Alignment ===")
    assert_same_obs(activity_tr, flux_tr, name_hint="train Act vs Flux")
    assert_same_obs(activity_te, flux_te, name_hint="test Act vs Flux")
    assert_same_obs(activity_tr, niche_tr, name_hint="train Act vs Niche")
    assert_same_obs(activity_te, niche_te, name_hint="test Act vs Niche")

    # Harmonize features within each modality
    for name, a, b in [("Activity", activity_tr, activity_te),
                       ("Flux", flux_tr, flux_te),
                       ("Niche", niche_tr, niche_te)]:
        if not a.var_names.equals(b.var_names):
            common = a.var_names.intersection(b.var_names)
            if len(common) == 0:
                raise ValueError(f"No common features between {name} train/test.")
            a = a[:, common].copy()
            b = b[:, a.var_names].copy()
            if name == "Activity":
                activity_tr, activity_te = a, b
            elif name == "Flux":
                flux_tr, flux_te = a, b
            else:
                niche_tr, niche_te = a, b
            print(f"⚠️  Harmonized {name} features to intersection ({len(common)})")

    # Labels
    print("\n=== Applying Split Labels ===")
    activity_tr = change_label(activity_tr, 'train'); activity_te = change_label(activity_te, 'test')
    flux_tr     = change_label(flux_tr,     'train'); flux_te     = change_label(flux_te,     'test')
    niche_tr    = change_label(niche_tr,    'train'); niche_te    = change_label(niche_te,    'test')

    # Scaling
    print("\n=== Feature scaling ===")
    if args.scale_flux:
        zscore_from_train(flux_tr, [flux_tr, flux_te], out_dir, prefix="flux_zscore")
    if args.scale_activity:
        zscore_from_train(activity_tr, [activity_tr, activity_te], out_dir, prefix="activity_zscore")
    if args.scale_niche:
        zscore_from_train(niche_tr, [niche_tr, niche_te], out_dir, prefix="niche_zscore")

    # Build concatenated sets
    print("\n=== Building Train & All Sets (Activity + Flux + Niche) ===")
    activity_all = concat_train_test(activity_tr, activity_te)
    flux_all     = concat_train_test(flux_tr,     flux_te)
    niche_all    = concat_train_test(niche_tr,    niche_te)

    train_list = [ensure_dense_float32(x.copy()) for x in [activity_tr, flux_tr, niche_tr]]
    all_list   = [ensure_dense_float32(x.copy()) for x in [activity_all, flux_all, niche_all]]

    # Overwrite I/O dims from data
    print("\n=== Configuring Model I/O from data (CapBoost Supervised) ===")
    modality_names = ['Activity','Flux','Niche']
    for i, adata in enumerate(train_list):
        in_dim = adata.X.shape[1]
        config['encoders'][i]['input']       = in_dim
        config['decoders'][i]['output']      = in_dim
        config['discriminators'][i]['input'] = in_dim
        print(f"Modality {i} ({modality_names[i]}): input/output = {in_dim}")

    save_config(config, out_dir, filename="config_resolved.json")

    # Train / load
    print(f"\n=== {'Training' if args.train_model else 'Loading'} UnitedNet (CapBoost Supervised) ===")
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
    model_fn = f"model_perturbmap_activity_flux_niche_capboost_supervised_{args.dataset_id}"
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
        "modalities": "Activity+Flux+Niche",
        "scale_flux": bool(args.scale_flux),
        "scale_activity": bool(args.scale_activity),
        "scale_niche": bool(args.scale_niche),
        "cmp_edge_activity_to_niche": args.cmp_edge_activity_to_niche,
    })

    print("\n=== Saving Model & Summary ===")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved model to {model_path}")

    summary = {
        "dataset_id": args.dataset_id,
        "technique": args.technique,
        "device": args.device,
        "modalities": "Activity+Flux+Niche",
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
        "scaling": {
            "flux": bool(args.scale_flux),
            "activity": bool(args.scale_activity),
            "niche": bool(args.scale_niche),
        },
        "cmp_edge_weights": config.get('cmp_edge_weights'),
        "shapes": {
            "activity_train": list(activity_tr.shape),
            "flux_train": list(flux_tr.shape),
            "niche_train": list(niche_tr.shape),
            "activity_all": list(activity_all.shape),
            "flux_all": list(flux_all.shape),
            "niche_all": list(niche_all.shape),
        },
    }
    with open(out_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {out_dir/'training_summary.json'}")

    print("\n=== Notes ===")
    print("- This supervised variant uses 'supervised_group_identification' in training.")
    print("- Finetuning uses 'cross_model_prediction_clas' to keep CMP label-aware.")
    print("- Use n_clusters equal to number of phenotypes (e.g. 4 for KP_1-1/2/3/periphery).")
    print("- You can still run predict_label() for class predictions at inference.")

    print("\n=== Done ===")
    return model

if __name__ == "__main__":
    main()