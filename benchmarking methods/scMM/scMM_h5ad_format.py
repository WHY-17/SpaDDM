import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.sparse import issparse
from pathlib import Path



dataset = '../Data/Dataset10_Mouse_Brain_H3K27me3/'

adata_RNA = sc.read_h5ad(dataset + 'adata_RNA.h5ad')
adata_ATAC = sc.read_h5ad(dataset + 'adata_Peaks.h5ad')


# if expression matrix is dense matrix, need to tranform to sparse matrix
if not issparse(adata_RNA.X):
    adata_RNA.X = sparse.coo_matrix(adata_RNA.X)
if not issparse(adata_ATAC.X):
    adata_ATAC.X = sparse.coo_matrix(adata_ATAC.X)


#### RNA+Protein
# RNA
from scipy.io import mmwrite
RNA_count = adata_RNA.X.copy()
save_path_RNA = Path(dataset + 'RNA-seq')
save_path_RNA.mkdir(parents=True, exist_ok=True)
mmwrite(str(save_path_RNA) + '/' + "RNA_count.mtx", RNA_count.T)
barcode = pd.DataFrame(index=adata_RNA.obs_names)
barcode.to_csv(str(save_path_RNA) + '/' + 'barcode.tsv', sep='\t', header=None)

gene = pd.DataFrame(index=adata_RNA.var_names)
gene.to_csv(str(save_path_RNA) + '/' + 'gene.tsv', sep='\t', header=None)


# ATAC
ATAC_count = adata_ATAC.X.copy()
save_path_ATAC = Path(dataset + 'ATAC-seq')
save_path_ATAC.mkdir(parents=True, exist_ok=True)
mmwrite(str(save_path_ATAC) + '/' + "ATAC_count.mtx", ATAC_count.T)

barcode = pd.DataFrame(index=adata_ATAC.obs_names)
barcode.to_csv(str(save_path_ATAC) + '/' + 'barcode.tsv', sep='\t', header=None)

ATAC = pd.DataFrame(index=adata_ATAC.var_names)
ATAC.to_csv(str(save_path_ATAC) + '/' + 'peak.tsv', sep='\t', header=None)









"""
dataset = '../Data/Dataset17_Simulation5/'

# Thymus
adata_RNA = sc.read_h5ad(dataset + 'adata_RNA.h5ad')
adata_ADT = sc.read_h5ad(dataset + 'adata_ADT.h5ad')


# if expression matrix is dense matrix, need to tranform to sparse matrix
if not issparse(adata_RNA.X):
    adata_RNA.X = sparse.coo_matrix(adata_RNA.X)
if not issparse(adata_ADT.X):
    adata_ADT.X = sparse.coo_matrix(adata_ADT.X)


#### RNA+Protein
# RNA
from scipy.io import mmwrite
RNA_count = adata_RNA.X.copy()
save_path_RNA = Path(dataset + 'RNA-seq')
save_path_RNA.mkdir(parents=True, exist_ok=True)
mmwrite(str(save_path_RNA) + '/' + "RNA_count.mtx", RNA_count.T)

barcode = pd.DataFrame(index=adata_RNA.obs_names)
barcode.to_csv(str(save_path_RNA) + '/' + 'barcode.tsv', sep='\t', header=None)

gene = pd.DataFrame(index=adata_RNA.var_names)
gene.to_csv(str(save_path_RNA) + '/' + 'gene.tsv', sep='\t', header=None)


# ADT
Protein_count = adata_ADT.X.copy()
save_path_ADT = Path(dataset + 'CITE-seq')
save_path_ADT.mkdir(parents=True, exist_ok=True)
mmwrite(str(save_path_ADT) + '/' + "Protein_count.mtx", Protein_count.T)

barcode = pd.DataFrame(index=adata_ADT.obs_names)
barcode.to_csv(str(save_path_ADT) + '/' + 'barcode.tsv', sep='\t', header=None)

protein = pd.DataFrame(index=adata_ADT.var_names)
protein.to_csv(str(save_path_ADT) + '/' + 'protein.tsv', sep='\t', header=None)
"""

