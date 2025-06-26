import os
import torch
import pandas as pd
import scanpy as sc
import Multi_Diffusion
import dgl


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ['R_HOME'] = 'D:\software\R-4.3.3'

# read data
file_fold = 'data/Data_SpatialGlue/Dataset2_Mouse_Spleen2/'
adata_omics_1 = sc.read_h5ad(file_fold + 'adata_RNA.h5ad')
adata_omics_2 = sc.read_h5ad(file_fold + 'adata_ADT.h5ad')
adata_omics_1.var_names_make_unique()
adata_omics_2.var_names_make_unique()


# Specify data type
data_type = 'SPOTS'
# Fix random seed
from Multi_Diffusion.preprocess import fix_seed
random_seed = 2024
fix_seed(random_seed)


from Multi_Diffusion.preprocess import clr_normalize_each_cell, pca
# RNA
sc.pp.filter_genes(adata_omics_1, min_cells=10)
# sc.pp.filter_cells(adata_omics_1, min_genes=80)
sc.pp.filter_genes(adata_omics_2, min_cells=50)
adata_omics_2 = adata_omics_2[adata_omics_1.obs_names].copy()
sc.pp.highly_variable_genes(adata_omics_1, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata_omics_1, target_sum=1e4)
sc.pp.log1p(adata_omics_1)
adata_omics_1 = adata_omics_1[:, adata_omics_1.var['highly_variable']]
adata_omics_1.obsm['feat'] = pca(adata_omics_1, n_comps=adata_omics_2.n_vars-1)

# Protein
adata_omics_2 = clr_normalize_each_cell(adata_omics_2)
adata_omics_2.obsm['feat'] = pca(adata_omics_2, n_comps=adata_omics_2.n_vars-1)


from Multi_Diffusion.preprocess import construct_neighbor_graph
data = construct_neighbor_graph(adata_omics_1, adata_omics_2, datatype=data_type)
dgl_omics_1 = dgl.graph((data['adata_omics_1'].uns['adj_spatial']['x'], data['adata_omics_1'].uns['adj_spatial']['y']))
dgl_omics_2 = dgl.graph((data['adata_omics_2'].uns['adj_spatial']['x'], data['adata_omics_2'].uns['adj_spatial']['y']))
dgl_omics_1.ndata['feat'] = torch.FloatTensor(adata_omics_1.obsm['feat'])
dgl_omics_2.ndata['feat'] = torch.FloatTensor(adata_omics_2.obsm['feat'])
dgl_omics_1 = dgl.add_self_loop(dgl_omics_1)
dgl_omics_2 = dgl.add_self_loop(dgl_omics_2)
dgl_graph = {'dgl_omics_1': dgl_omics_1,
             'dgl_omics_2': dgl_omics_2}

# define model
from Multi_Diffusion.SpatialDDM import Train_SpatialDDM
model = Train_SpatialDDM(data=data, graph=dgl_graph, datatype=data_type, device=device)
# train model
output = model.train()


adata = adata_omics_1.copy()
adata.obsm['emb_latent_omics_1'] = output['emb_omics_1'].copy()
adata.obsm['emb_latent_omics_2'] = output['emb_omics_2'].copy()
adata.obsm['SpatialDDM'] = output['SpatialDDM'].copy()
adata.obsm['alpha'] = output['alpha']
# adata.obsm['alpha_omics_1'] = output['alpha_omics_1']
# adata.obsm['alpha_omics_2'] = output['alpha_omics_2']

# flip tissue image
import numpy as np
adata.obsm['spatial'] = np.rot90(np.rot90(np.rot90(np.array(adata.obsm['spatial'])).T).T).T
adata.obsm['spatial'][:,1] = -1*adata.obsm['spatial'][:,1]

# we set 'mclust' as clustering tool by default. Users can also select 'leiden' and 'louvain'
from Multi_Diffusion.utils import clustering
tool = 'mclust' # mclust, leiden, and louvain
clustering(adata, key='SpatialDDM', add_key='SpatialDDM', n_clusters=5, method=tool, use_pca=True)
adata.write_h5ad(file_fold+'SpatialDDM_results.h5ad', compression='gzip')

# evaluation of Clutering
from sklearn.metrics import silhouette_score
silhouette = silhouette_score(adata.obsm['SpatialDDM'], adata.obs['SpatialDDM'])
print("Silhouette Score:", silhouette)


from sklearn.metrics import calinski_harabasz_score
ch_score = calinski_harabasz_score(adata.obsm['SpatialDDM'], adata.obs['SpatialDDM'])
print("Calinski-Harabasz Index Score:", ch_score)


from sklearn.metrics import davies_bouldin_score
db_score = davies_bouldin_score(adata.obsm['SpatialDDM'], adata.obs['SpatialDDM'])
print("Davies-Bouldin Index Score:", db_score)

# flip tissue image
import numpy as np
adata.obsm['spatial'] = np.rot90(np.rot90(np.rot90(np.array(adata.obsm['spatial'])).T).T).T
adata.obsm['spatial'][:,1] = -1*adata.obsm['spatial'][:,1]



"""
# visualization
import matplotlib.pyplot as plt
fig, ax_list = plt.subplots(1, 2, figsize=(7, 3))
sc.pp.neighbors(adata, use_rep='SpatialDDM', n_neighbors=10)
sc.tl.umap(adata)
sc.pl.umap(adata, color='SpatialDDM', ax=ax_list[0], title='SpatialDDM', s=20, show=False)
sc.pl.embedding(adata, basis='spatial', color='SpatialDDM', ax=ax_list[1], title='SpatialDDM', s=25, show=False)
plt.tight_layout(w_pad=0.3)
plt.show()
plt.savefig(file_fold + 'visualization_200.jpg', dpi=300)
"""


"""
# annotation
adata.obs['SpatialGlue_number'] = adata.obs['SpatialGlue'].copy()

# 'MZMØ' represents marginal zone macrophage, 'MMMØ' represents marginal metallophilic macrophages, 'RpMØ' represents red pulp macrophage
adata.obs['SpatialGlue'].cat.rename_categories({1: 'MMMØ',
                                                   2: 'MZMØ',
                                                   3: 'B cell',
                                                   4: 'T cell',
                                                   5: 'RpMØ'
                                                   }, inplace=True)


# reorder
import pandas as pd
list_ = ['MZMØ','MMMØ','RpMØ','B cell', 'T cell']
adata.obs['SpatialGlue']  = pd.Categorical(adata.obs['SpatialGlue'],
                      categories=list_,
                      ordered=True)


# plotting with annotation
fig, ax_list = plt.subplots(1, 2, figsize=(6.5, 3))

sc.pl.umap(adata, color='SpatialGlue', ax=ax_list[0], title='SpatialGlue', s=20, show=False)
sc.pl.embedding(adata, basis='spatial', color='SpatialGlue', ax=ax_list[1], title='SpatialGlue', s=25, show=False)
ax_list[0].get_legend().remove()

plt.tight_layout(w_pad=0.3)
plt.show()
"""

"""
# plotting modality weight values.
import pandas as pd
import seaborn as sns
plt.rcParams['figure.figsize'] = (5,3)
df = pd.DataFrame(columns=['RNA', 'protein', 'label'])
df['RNA'], df['protein'] = adata.obsm['alpha'][:, 0], adata.obsm['alpha'][:, 1]
df['label'] = adata.obs['SpatialDDM'].values
df = df.set_index('label').stack().reset_index()
df.columns = ['label_SpatialDDM', 'Modality', 'Weight value']
ax = sns.violinplot(data=df, x='label_SpatialDDM', y='Weight value', hue="Modality",
                split=True, inner="quart", linewidth=1)
ax.set_title('RNA vs protein')
ax.set_xlabel('SpatialDDM label')
ax.legend(bbox_to_anchor=(1.4, 1.01), loc='upper right')
plt.tight_layout(w_pad=0.05)
plt.show()
plt.savefig(file_fold + 'weight.jpg', dpi=300)
"""




