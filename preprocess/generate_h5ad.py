import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import os
from collections import Counter

def normalize(adata, highly_genes = None, filter_min_counts=True, size_factors=True, normalize_input=True,
              logtrans_input=True, pca=50):
    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=3)
        sc.pp.filter_cells(adata, min_counts=200)
    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata
    if size_factors:
        sc.pp.normalize_total(adata, target_sum=1e4)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0
    if logtrans_input:
        sc.pp.log1p(adata)
    if highly_genes != None:
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes = highly_genes, subset=True)
        adata = adata[:, adata.var.highly_variable]
        # adata.raw = adata.raw[:, adata.var.highly_variable]
    if normalize_input:
        sc.pp.scale(adata, max_value=10, zero_center=True)
    if pca:
        sc.tl.pca(adata, n_comps=pca)
    return adata


if __name__=="__main__":
    file = "Qx_Mammary_Glan"
    celltype_name = "Group"
    input_h5ad_path = "../data/"+file+"/"+file+".h5ad"
    save_h5ad_dir = "../data/"+file+"/"+file+"_preprocessed.h5ad"

    adata = sc.read_h5ad(input_h5ad_path)
    print("******** ", file, " **********")
    type_counter = Counter(adata.obs[celltype_name])
    adata.obs['Group'] = adata.obs[celltype_name]
    adata = normalize(adata, highly_genes=5000, size_factors=True, normalize_input=True,
                      logtrans_input=True)
    adata.write(save_h5ad_dir)
    print(adata)

