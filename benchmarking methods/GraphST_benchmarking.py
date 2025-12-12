import os
import torch
import pandas as pd
import scanpy as sc
from sklearn import metrics
import multiprocessing as mp
from GraphST import GraphST



# Run device，by default, the package is implemented on 'cpu'. We recommend using GPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# the location of R, which is necessary for mclust algorithm. Please replace it with local R installation path
os.environ['R_HOME'] = 'D:\software\R-4.3.3'


dataset = 'Dataset2_Mouse_Spleen2/'
# the number of clusters
n_clusters = 5


# read data
file_path = 'Data/' + dataset #please replace 'file_path' with the download path
adata = sc.read_h5ad(file_path + 'adata_RNA.h5ad')
adata.var_names_make_unique()



# define model
model = GraphST.GraphST(adata, datatype='SPOTS', device=device)
# run model
adata = model.train()



# clustering
from GraphST.utils import clustering
tool = 'mclust' # mclust, leiden, and louvain
# clustering
from GraphST.utils import clustering
if tool == 'mclust':
   clustering(adata, n_clusters, method=tool)
elif tool in ['leiden', 'louvain']:
   clustering(adata, n_clusters, method=tool, start=0.1, end=2.0, increment=0.01)


adata.write_h5ad(file_path + 'GraphST_results.h5ad', compression='gzip')


