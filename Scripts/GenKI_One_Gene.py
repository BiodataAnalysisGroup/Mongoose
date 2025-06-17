# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 11:47:00 2025

@author: aspasiaor
"""
import os
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import re
import anndata as ad

sc.settings.verbosity = 0

import GenKI as gk
from GenKI.preprocesing import build_adata
from GenKI.dataLoader import DataLoader
from GenKI.train import VGAE_trainer
from GenKI import utils

import stringdb
import networkx as nx

from scipy.sparse import csr_matrix
from scipy.sparse import issparse


# Read the dataset
adata = sc.read("VisiumRNABC_v2.h5ad")

# Standardize the data (zero mean, unit variance)
sc.pp.scale(adata, zero_center=True)

# Let's commence with creating a digital KO of Six2 expression in the DBiT-seq mouse embryo dataset.
gene_of_interest = "ESR1"

# Verify that the gene of interest is part of the rownames in adata.var
if gene_of_interest in adata.var.index:
    print(f"The gene {gene_of_interest} is present in the rownames of adata.var.")
else:
    print(f"The gene {gene_of_interest} is not present in the rownames of adata.var.")

# adata pre-processing to prepare for input in the GenKI tool
adata.layers["norm"] = adata.X.copy()

# The adata.X should be normalised-scaled AND in sparse matrix format!
if not issparse(adata.X):
    sparse_matrix = csr_matrix(adata.X)
    adata.X = sparse_matrix
    print("Converted adata.X to a sparse matrix.")
else:
    print("adata.X is already a sparse matrix.")



# Create the GRN

data_wrapper =  DataLoader(
                adata, # adata object
                target_gene = [gene_of_interest], # KO gene name
                target_cell = None, # obsname for cell type, if none use all
                obs_label = "ident", # colname for genes
                GRN_file_dir = "GRNs", # folder name for GRNs
                rebuild_GRN = True, # whether build GRN by pcNet
                pcNet_name = "Visium_breast_cancer_example", # GRN file name
                verbose = True, # whether verbose
                n_cpus = 8, # multiprocessing
                )

data_wt = data_wrapper.load_data()
data_ko = data_wrapper.load_kodata()






#Changed the hyperparameters to better adapt to the spatial dataset in hand

hyperparams = {
    "epochs": 300,  # Increased epochs for more training
    "lr": 5e-2,  # Adjusted learning rate
    "beta": 5e-4,  # Increased beta for stronger regularization
    "seed": 8096  # Trying a different seed
}


log_dir = None

sensei = VGAE_trainer(
    data_wt,
    epochs=hyperparams["epochs"],
    lr=hyperparams["lr"],
    log_dir=log_dir,
    beta=hyperparams["beta"],
    seed=hyperparams["seed"],
    verbose=False,
)


sensei.train()
sensei.save_model('Visium3K_breast_cancer_model_example')


# get distance between wt and ko

z_mu_wt, z_std_wt = sensei.get_latent_vars(data_wt)
z_mu_ko, z_std_ko = sensei.get_latent_vars(data_ko)
dis = gk.utils.get_distance(z_mu_ko, z_std_ko, z_mu_wt, z_std_wt, by="KL")
print(dis.shape)


# raw ranked gene list

res_raw = utils.get_generank(data_wt, dis, rank=True)
res_raw.head(10)



null = sensei.pmt(data_ko, n=100, by="KL")
res = utils.get_generank(data_wt, dis, null,)
#                       save_significant_as = 'gene_list_example')
res


sc.pl.spatial(adata, img_key="hires", color=["leiden", "ESR1"])







