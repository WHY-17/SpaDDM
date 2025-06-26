import os
import torch
import pandas as pd
import scanpy as sc
import dgl



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ['R_HOME'] = 'D:\software\R-4.3.3'
# read data
file_fold = 'data/Data_SpatialGlue/Dataset17_Simulation5/' #please replace 'file_fold' with the download path

adata_omics1 = sc.read_h5ad(file_fold + 'adata_RNA.h5ad')
adata_omics2 = sc.read_h5ad(file_fold + 'adata_ADT.h5ad')
adata_omics1.var_names_make_unique()
adata_omics2.var_names_make_unique()


# Fix random seed
from Multi_Diffusion.preprocess import fix_seed
random_seed = 2024
fix_seed(random_seed)


from Multi_Diffusion.preprocess import clr_normalize_each_cell, pca, lsi
n_protein = adata_omics2.n_vars
sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata_omics1, target_sum=1e4)
sc.pp.log1p(adata_omics1)

adata_omics1 = adata_omics1[:, adata_omics1.var['highly_variable']]
# adata_omics1.obsm['feat'] = pca(adata_omics1, n_comps=n_protein)
adata_omics1.obsm['feat'] = adata_omics1.X

# Protein
adata_omics2 = clr_normalize_each_cell(adata_omics2)
# adata_omics2.obsm['feat'] = pca(adata_omics2, n_comps=n_protein)
adata_omics2.obsm['feat'] = adata_omics2.X


from Multi_Diffusion.preprocess import construct_neighbor_graph
data = construct_neighbor_graph(adata_omics1, adata_omics2)
dgl_omics_1 = dgl.graph((data['adata_omics_1'].uns['adj_spatial']['x'], data['adata_omics_1'].uns['adj_spatial']['y']))
dgl_omics_2 = dgl.graph((data['adata_omics_2'].uns['adj_spatial']['x'], data['adata_omics_2'].uns['adj_spatial']['y']))
dgl_omics_1.ndata['feat'] = torch.FloatTensor(adata_omics1.obsm['feat'])
dgl_omics_2.ndata['feat'] = torch.FloatTensor(adata_omics2.obsm['feat'])
dgl_omics_1 = dgl.add_self_loop(dgl_omics_1)
dgl_omics_2 = dgl.add_self_loop(dgl_omics_2)
dgl_graph = {'dgl_omics_1': dgl_omics_1,
             'dgl_omics_2': dgl_omics_2}


# define model
from Multi_Diffusion.SpatialDDM import Train_SpatialDDM
model = Train_SpatialDDM(data=data, graph=dgl_graph, device=device)
# train model
output = model.train()


adata = adata_omics1.copy()
adata.obsm['emb_latent_omics_1'] = output['emb_omics_1'].copy()
adata.obsm['emb_latent_omics_2'] = output['emb_omics_2'].copy()
adata.obsm['SpatialDDM'] = output['SpatialDDM'].copy()
adata.obsm['alpha'] = output['alpha']

# Set ground truth
import numpy as np
adata.obs['ground_truth'] = 1*np.array(adata.obsm['spfac'][:,0] + 2*adata.obsm['spfac'][:,1] +
                                       3*adata.obsm['spfac'][:,2] + 4*adata.obsm['spfac'][:,3])
adata.obs['annotation'] = adata.obs['ground_truth']
adata.obs['annotation'].replace({1.0:'factor1',
                                   2.0:'factor2',
                                   3.0:'factor3',
                                   4.0:'factor4',
                                   0.0:'backgr'  #'backgr' means background.
                                              }, inplace=True)
list_ = ['factor1','factor2','factor3','factor4','backgr']
adata.obs['annotation'] = pd.Categorical(adata.obs['annotation'],
                      categories=list_,
                      ordered=True)



# We set 'mclust' as clustering tool by default. Users can also select 'leiden' and 'louvain'
from Multi_Diffusion.utils import clustering
tool = 'mclust' # mclust, leiden, and louvain
clustering(adata, key='SpatialDDM', add_key='SpatialDDM', n_clusters=5, method=tool, use_pca=False)


#clustering performance evaluation
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score, homogeneity_score, mutual_info_score, v_measure_score


adata_obs = adata.obs.dropna()
ARI = adjusted_rand_score(adata_obs['annotation'], adata_obs['SpatialDDM'])
NMI = normalized_mutual_info_score(adata_obs['annotation'], adata_obs['SpatialDDM'])
AMI = adjusted_mutual_info_score(adata_obs['annotation'], adata_obs['SpatialDDM'])
Hom_score = homogeneity_score(adata_obs['annotation'], adata_obs['SpatialDDM'])
Mut_info_score = mutual_info_score(adata_obs['annotation'], adata_obs['SpatialDDM'])
v_score = v_measure_score(adata_obs['annotation'], adata_obs['SpatialDDM'])
print("ARI-score:  ", ARI)
print("NMI-score:  ", NMI)
print("AMI-score:  ", AMI)
print("Hom-score:  ", Hom_score)
print("Mul_info-score:  ", Mut_info_score)
print("V-score:  ", v_score)


"""
# visualization
import matplotlib.pyplot as plt
fig, ax_list = plt.subplots(1, 2, figsize=(7, 3))
sc.pp.neighbors(adata, use_rep='SpatialDDM', n_neighbors=30)
sc.tl.umap(adata)
sc.pl.umap(adata, color='SpatialDDM', ax=ax_list[0], title='SpatialDDM', s=60, show=False)
sc.pl.embedding(adata, basis='spatial', color='SpatialDDM', ax=ax_list[1], title='SpatialDDM', s=90, show=False)
plt.tight_layout(w_pad=0.3)
plt.show()
# plt.savefig(file_fold + 'visualization.jpg', dpi=300)
"""
