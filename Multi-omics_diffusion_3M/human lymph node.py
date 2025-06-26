import os
import torch
import pandas as pd
import scanpy as sc
import dgl
import stlearn as st
from pathlib import Path
import anndata
from Multi_Diffusion.image_features import add_image, image_feature, image_crop



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ['R_HOME'] = 'D:\software\R-4.3.3'
# read data
file_fold = 'data/human lymph node/A1_LN/' #please replace 'file_fold' with the download path
# adata_RNA = sc.read_h5ad(file_fold + 'adata_RNA.h5ad')
adata_omics1 = sc.read_h5ad(file_fold + 'adata_RNA_image.h5ad')
adata_omics2 = sc.read_h5ad(file_fold + 'adata_ADT.h5ad')


"""
# coordination
coord = pd.read_csv(file_fold + 'A1LN_tissue_positions.csv', index_col=0)
coord = coord[coord['in_tissue']==1]    # spots in tissue
coord = coord.reindex(adata_RNA.obs_names)   # maintain the same barcode index
adata_RNA.obs = pd.concat([adata_RNA.obs, coord], axis=1)

# adding image
adata_RNA = add_image(adata_RNA, library_id='A1_LN',
                      image_hires_path=file_fold + 'A1LN_tissue_hires_image.png',
                      image_lowres_path=file_fold + 'A1LN_tissue_lowres_image.png')
adata_RNA = st.convert_scanpy(adata_RNA)
save_path1 = file_fold + 'results/'
data_name1 = 'RNA'
save_path_image_crop1 = Path(os.path.join(save_path1, 'Image_crop', f'{data_name1}'))
save_path_image_crop1.mkdir(parents=True, exist_ok=True)
adata_RNA = image_crop(adata_RNA, save_path=save_path_image_crop1)
adata_RNA = image_feature(adata_RNA, pca_components=30)
adata_RNA.write_h5ad(file_fold + 'adata_RNA_image.h5ad', compression='gzip')
"""


adata_omics1.var_names_make_unique()
adata_omics2.var_names_make_unique()

# Specify data type
data_type = '10x'

# Fix random seed
from Multi_Diffusion.preprocess import fix_seed
random_seed = 2024
fix_seed(random_seed)


from Multi_Diffusion.preprocess import clr_normalize_each_cell, pca
# RNA
sc.pp.filter_genes(adata_omics1, min_cells=10)
sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata_omics1, target_sum=1e4)
sc.pp.log1p(adata_omics1)
sc.pp.scale(adata_omics1)

adata_omics1 = adata_omics1[:, adata_omics1.var['highly_variable']]
adata_omics1.obsm['feat'], adata_omics1_pca_weight = pca(adata_omics1, n_comps=adata_omics2.n_vars-1)

# Protein
adata_omics2 = clr_normalize_each_cell(adata_omics2)
sc.pp.scale(adata_omics2)
adata_omics2.obsm['feat'], adata_omics2_pca_weight = pca(adata_omics2, n_comps=adata_omics2.n_vars-1)

from Multi_Diffusion.preprocess import construct_neighbor_graph
data = construct_neighbor_graph(adata_omics1, adata_omics2, datatype=data_type)
dgl_omics_1 = dgl.graph((data['adata_omics_1'].uns['adj_spatial']['x'], data['adata_omics_1'].uns['adj_spatial']['y']))
dgl_omics_2 = dgl.graph((data['adata_omics_2'].uns['adj_spatial']['x'], data['adata_omics_2'].uns['adj_spatial']['y']))
dgl_omics_1.ndata['feat'] = torch.FloatTensor(adata_omics1.obsm['feat'])
dgl_omics_1.ndata['image'] = torch.FloatTensor(adata_omics1.obsm['image_feat_pca'])
dgl_omics_2.ndata['feat'] = torch.FloatTensor(adata_omics2.obsm['feat'])
dgl_omics_1 = dgl.add_self_loop(dgl_omics_1)
dgl_omics_2 = dgl.add_self_loop(dgl_omics_2)
dgl_graph = {'dgl_omics_1': dgl_omics_1,
             'dgl_omics_2': dgl_omics_2}


# define model
from Multi_Diffusion.SpatialDDM import Train_SpatialDDM
model = Train_SpatialDDM(data=data, graph=dgl_graph, datatype=data_type, device=device)
# train model
output = model.train(save_path=file_fold)

adata = adata_omics1.copy()
adata.obsm['emb_latent_omics_1'] = output['emb_omics_1'].copy()
adata.obsm['emb_latent_omics_2'] = output['emb_omics_2'].copy()
adata.obsm['SpatialDDM'] = output['SpatialDDM'].copy()
adata.obsm['alpha'] = output['alpha']
adata.obsm['rec_omics_1'] = output['rec_omics_1']
adata.obsm['rec_omics_2'] = output['rec_omics_2']
adata.obsm['Protin_X_pca'] = adata_omics2.obsm['feat']


# we set 'mclust' as clustering tool by default. Users can also select 'leiden' and 'louvain'
from Multi_Diffusion.utils import clustering
tool = 'mclust' # mclust, leiden, and louvain
clustering(adata, key='SpatialDDM', add_key='SpatialDDM', n_clusters=6, method=tool, use_pca=True)


# load annotations
ann = pd.read_csv('data/human lymph node/A1_LN/' + 'annotation.csv', index_col=0)
adata.obs['Ground Truth'] = list(ann['manual-anno'])
"""
ann = sc.read_h5ad(file_fold+'slice2/s2_adata_rna.h5ad')
adata.obs['Ground Truth'] = pd.Categorical(list(ann.obs['final_annot']))
"""

from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score, homogeneity_score, mutual_info_score, v_measure_score
adata_obs = adata.obs.dropna()
ARI = adjusted_rand_score(adata_obs['Ground Truth'], adata_obs['SpatialDDM'])
NMI = normalized_mutual_info_score(adata_obs['Ground Truth'], adata_obs['SpatialDDM'])
AMI = adjusted_mutual_info_score(adata_obs['Ground Truth'], adata_obs['SpatialDDM'])
Hom_score = homogeneity_score(adata_obs['Ground Truth'], adata_obs['SpatialDDM'])
Mut_info_score = mutual_info_score(adata_obs['Ground Truth'], adata_obs['SpatialDDM'])
v_score = v_measure_score(adata_obs['Ground Truth'], adata_obs['SpatialDDM'])
print("ARI-score:  ", ARI)
print("NMI-score:  ", NMI)
print("AMI-score:  ", AMI)
print("Hom-score:  ", Hom_score)
print("Mul_info-score:  ", Mut_info_score)
print("V-score:  ", v_score)

adata.write_h5ad(file_fold + 'SpatialDDM_results.h5ad', compression='gzip')





"""
# visualization
import matplotlib.pyplot as plt
fig, ax_list = plt.subplots(1, 2, figsize=(7, 3))
sc.pp.neighbors(adata, use_rep='SpatialDDM', n_neighbors=30)
sc.tl.umap(adata)
sc.pl.umap(adata, color='SpatialDDM', ax=ax_list[0], title='SpatialDDM', s=20, show=False)
sc.pl.embedding(adata, basis='spatial', color='SpatialDDM', ax=ax_list[1], title='SpatialDDM', s=25, show=False)
plt.tight_layout(w_pad=0.3)
plt.show()
plt.savefig(file_fold + 'visualization_1800_32.jpg', dpi=300)
"""
