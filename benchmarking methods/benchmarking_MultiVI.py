import numpy as np
import scanpy as sc
import scvi
import pandas as pd
import anndata as ad
import time



t = time.time()
scvi.settings.seed = 420

def combine(adata_RNA, adata_ADT):
    adata_RNA.var['modality'] = 'Gene Expression'
    adata_ADT.var['modality'] = 'ADT'
    exp = np.hstack([adata_RNA.X.toarray(), adata_ADT.X])
    cell_name = list(adata_RNA.obs_names)
    gene_name = list(adata_RNA.var_names) + list(adata_ADT.var_names)
    modality = ['Gene Expression'] * adata_RNA.n_vars + ['ADT'] * adata_ADT.n_vars
    obs = pd.DataFrame(index=cell_name)
    var = pd.DataFrame(index=gene_name)
    adata_RNA_ADT = ad.AnnData(X=exp, obs=obs, var=var)
    adata_RNA_ADT.var['modality'] = modality
    adata_RNA_ADT.obsm['spatial'] = adata_RNA.obsm['spatial']
    return adata_RNA_ADT


dataset = 'E18_5-S2'
path = 'MISAR/' + dataset + '/'
adata_RNA = sc.read_h5ad(path + 'adata_RNA.h5ad')
adata_ADT = sc.read_h5ad(path + 'adata_Peak.h5ad')
adata_RNA.var_names_make_unique()
adata_ADT.var_names_make_unique()
adata = combine(adata_RNA, adata_ADT)
adata.var_names_make_unique()


# split to three datasets by modality (RNA, ATAC, Multiome), and corrupt data
# by remove some data to create single-modality data
n = int(0.3*adata.n_obs)
adata_rna = adata[:n].copy()
adata_paired = adata[n:2*n].copy()
adata_atac = adata[2*n:].copy()


# We can now use the organizing method from scvi to concatenate these anndata
adata_mvi = scvi.data.organize_multiome_anndatas(adata_paired, adata_rna, adata_atac)
adata_mvi = adata_mvi[:, adata_mvi.var["modality"].argsort()].copy()
print(adata_mvi.shape)
sc.pp.filter_genes(adata_mvi, min_cells=int(adata_mvi.shape[0] * 0.01))
#sc.pp.filter_cells(adata_mvi, min_genes=3)
print(adata_mvi.shape)

scvi.model.MULTIVI.setup_anndata(adata_mvi, batch_key="modality")
mvi = scvi.model.MULTIVI(adata_mvi, n_genes=(adata_mvi.var["modality"] == "Gene Expression").sum(),
                         n_regions=(adata_mvi.var["modality"] == "ADT").sum())


# fill nan value with 0
import pandas as pd
df = pd.DataFrame(adata_mvi.X)
df.fillna(0, inplace=True)
adata_mvi.X = df.values
mvi.train()

# obtain latent representation
adata_mvi.obsm["X_MultiVI"] = mvi.get_latent_representation()
adata_mvi.write_h5ad(path+dataset+'_MultiVI_results.h5ad', compression='gzip')
