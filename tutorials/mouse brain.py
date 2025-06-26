import os
import torch
import pandas as pd
import scanpy as sc
from Multi_Diffusion import SpatialDDM
import dgl


# Environment configuration. SpatialGlue pacakge can be implemented with either CPU or GPU. GPU acceleration is highly recommend for imporoved efficiency.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

# the location of R, which is necessary for mclust algorithm. Please replace the path below with local R installation path
os.environ['R_HOME'] = 'D:\software\R-4.3.3'

# Specify data type
data_type = 'Spatial-epigenome-transcriptome'
# Fix random seed
from Multi_Diffusion.preprocess import fix_seed
random_seed = 2024
fix_seed(random_seed)


# read data
file_fold = 'data/Data_SpatialGlue/Dataset7_Mouse_Brain_ATAC/' #please replace 'file_fold' with the download path
adata_omics1 = sc.read_h5ad(file_fold + 'adata_RNA.h5ad')
adata_omics2 = sc.read_h5ad(file_fold + 'adata_peaks_normalized.h5ad')
adata_omics1.var_names_make_unique()
adata_omics2.var_names_make_unique()


from Multi_Diffusion.preprocess import clr_normalize_each_cell, pca, lsi
# RNA
sc.pp.filter_genes(adata_omics1, min_cells=10)
# sc.pp.filter_cells(adata_omics1, min_genes=200)
sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata_omics1, target_sum=1e4)
sc.pp.log1p(adata_omics1)
sc.pp.scale(adata_omics1)

adata_omics1 = adata_omics1[:, adata_omics1.var['highly_variable']]
adata_omics1.obsm['feat'] = pca(adata_omics1, n_comps=50)

# ATAC
adata_omics2 = adata_omics2[adata_omics1.obs_names].copy() # .obsm['X_lsi'] represents the dimension reduced feature
# sc.pp.highly_variable_genes(adata_omics2, flavor="seurat_v3", n_top_genes=3000)
# adata_omics2 = adata_omics2[:, adata_omics2.var['highly_variable']]

if 'X_lsi' not in adata_omics2.obsm.keys():
    # sc.pp.highly_variable_genes(adata_omics2, flavor="seurat_v3", n_top_genes=3000)
    lsi(adata_omics2, use_highly_variable=False, n_components=51)
adata_omics2.obsm['feat'] = adata_omics2.obsm['X_lsi'].copy()


from Multi_Diffusion.preprocess import construct_neighbor_graph
data = construct_neighbor_graph(adata_omics1, adata_omics2, datatype=data_type)
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
model = Train_SpatialDDM(data=data, graph=dgl_graph, datatype=data_type, device=device)
# train model
output = model.train()


adata = adata_omics1.copy()
adata.obsm['emb_latent_omics_1'] = output['emb_omics_1'].copy()
adata.obsm['emb_latent_omics_2'] = output['emb_omics_2'].copy()
adata.obsm['SpatialDDM'] = output['SpatialDDM'].copy()
adata.obsm['alpha'] = output['alpha']
adata.obsm['rec_omics_1'] = output['rec_omics_1']
adata.obsm['rec_omics_2'] = output['rec_omics_2']



# we set 'mclust' as clustering tool by default. Users can also select 'leiden' and 'louvain'
from Multi_Diffusion.utils import clustering
tool = 'mclust' # mclust, leiden, and louvain
clustering(adata, key='SpatialDDM', add_key='SpatialDDM', n_clusters=18, method=tool, use_pca=True)

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



"""
# visualization
import matplotlib.pyplot as plt
fig, ax_list = plt.subplots(1, 2, figsize=(14, 5))
sc.pp.neighbors(adata, use_rep='SpatialDDM', n_neighbors=30)
sc.tl.umap(adata)
sc.pl.umap(adata, color='SpatialDDM', ax=ax_list[0], title='SpatialDDM', s=60, show=False)
sc.pl.embedding(adata, basis='spatial', color='SpatialDDM', ax=ax_list[1], title='SpatialDDM', s=90, show=False)
plt.tight_layout(w_pad=0.3)
plt.show()
plt.savefig(file_fold + 'visualization_1800_32.jpg', dpi=300)
"""


"""
# plotting modality weight values.
import pandas as pd
import seaborn as sns
plt.rcParams['figure.figsize'] = (7,3)
df = pd.DataFrame(columns=['RNA', 'peak', 'label'])
df['RNA'], df['peak'] = adata.obsm['alpha'][:, 0], adata.obsm['alpha'][:, 1]
df['label'] = adata.obs['SpatialDDM'].values
df = df.set_index('label').stack().reset_index()
df.columns = ['label_SpatialDDM', 'Modality', 'Weight value']
ax = sns.violinplot(data=df, x='label_SpatialDDM', y='Weight value', hue="Modality",
                split=True, inner="quart", linewidth=1)
ax.set_title('RNA vs peak')
ax.set_xlabel('SpatialDDM label')
ax.legend(bbox_to_anchor=(1.2, 1.01), loc='upper right')
plt.tight_layout(w_pad=0.05)
#plt.show()
plt.savefig(file_fold + 'modality weight.jpg', dpi=300)
"""



