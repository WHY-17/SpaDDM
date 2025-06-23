import mudata as md
# import muon
import scanpy as sc
import scvi
import time
import numpy as np


t = time.time()
scvi.settings.seed = 1234
data_file = '/home/featurize/work/Data/Dataset10_Mouse_Brain_H3K27me3'
adata_rna = sc.read_h5ad(data_file + '/adata_RNA.h5ad')
adata_adt = sc.read_h5ad(data_file + '/adata_Peaks.h5ad')
adata_rna.var_names_make_unique()
adata_adt.var_names_make_unique()

sc.pp.highly_variable_genes(adata_rna, n_top_genes=4000, flavor="seurat_v3")
sc.pp.highly_variable_genes(adata_adt, n_top_genes=4000, flavor="seurat_v3")

adata_rna.X = adata_rna.X.toarray()
adata_rna.X = np.asarray(adata_rna.X)
adata_adt.X = adata_adt.X.toarray()
adata_adt.X = np.asarray(adata_adt.X)
adata_rna.obs.reset_index(inplace=True)
adata_adt.obs.reset_index(inplace=True)
unique_var_rna = adata_rna.var.loc[~adata_rna.var.index.duplicated(keep='first')]
adata_rna = adata_rna[:, unique_var_rna.index]
unique_var_adt = adata_adt.var.loc[~adata_adt.var.index.duplicated(keep='first')]
adata_adt = adata_adt[:, unique_var_adt.index]
adata_rna.obs['batch'] = 'RNA'
adata_rna.layers['counts'] = adata_rna.X.copy()
mdata = md.MuData({"rna": adata_rna, "protein": adata_adt})
# Place subsetted counts in a new modality
mdata.mod["rna_subset"] = mdata.mod["rna"][:, mdata.mod["rna"].var["highly_variable"]].copy()
mdata.mod["protein"] = mdata.mod["protein"][:, mdata.mod["protein"].var["highly_variable"]].copy()
mdata.update()
mdata = mdata.copy()
scvi.model.TOTALVI.setup_mudata(mdata, rna_layer="counts",
                                protein_layer=None, batch_key="batch",
                                modalities={"rna_layer": "rna_subset",
                                            "protein_layer": "protein",
                                            "batch_key": "rna_subset",})
vae = scvi.model.TOTALVI(mdata)
vae.train()
rna = mdata.mod["rna_subset"]
protein = mdata.mod["protein"]
# arbitrarily store latent in rna modality
rna.obsm["X_totalVI"] = vae.get_latent_representation()
rna.write_h5ad(data_file+'/_totalVI_results.h5ad', compression='gzip')





"""
dataset = 'Dataset8_Mouse_Brain_H3K4me3'
adata_rna = sc.read_h5ad('Data_SpatialGlue/' + dataset + '/adata_RNA.h5ad')
adata_adt = sc.read_h5ad('Data_SpatialGlue/' + dataset + '/adata_peaks_normalized.h5ad')
adata_rna.var_names_make_unique()
adata_adt.var_names_make_unique()

sc.pp.highly_variable_genes(adata_rna, n_top_genes=4000, flavor="seurat_v3")

adata_rna.X = adata_rna.X.todense()
adata_rna.X = np.asarray(adata_rna.X)
adata_adt.X = adata_adt.X.todense().astype(np.float32)
adata_adt.X = np.asarray(adata_adt.X)

adata_rna.obs['batch'] = 'Brain'
adata_rna.layers['counts'] = adata_rna.X.copy()

mdata = md.MuData({"rna": adata_rna, "protein": adata_adt})

# Place subsetted counts in a new modality
mdata.mod["rna_subset"] = mdata.mod["rna"][:, mdata.mod["rna"].var["highly_variable"]].copy()

mdata.update()
scvi.model.TOTALVI.setup_mudata(mdata, rna_layer="counts",
                                protein_layer=None, batch_key="batch",
                                modalities={"rna_layer": "rna_subset",
                                            "protein_layer": "protein",
                                            "batch_key": "rna_subset",})
vae = scvi.model.TOTALVI(mdata)
vae.train()

rna = mdata.mod["rna_subset"]
protein = mdata.mod["protein"]
# arbitrarily store latent in rna modality
rna.obsm["X_totalVI"] = vae.get_latent_representation()
path = 'Data_SpatialGlue/' + dataset + '/'
rna.write_h5ad(path+dataset+'_totoalVI_results.h5ad', compression='gzip')




dataset = 'Dataset9_Mouse_Brain_H3K27ac'
adata_rna = sc.read_h5ad('Data_SpatialGlue/' + dataset + '/adata_RNA.h5ad')
adata_adt = sc.read_h5ad('Data_SpatialGlue/' + dataset + '/adata_peaks_normalized.h5ad')
adata_rna.var_names_make_unique()
adata_adt.var_names_make_unique()

sc.pp.highly_variable_genes(adata_rna, n_top_genes=4000, flavor="seurat_v3")

adata_rna.X = adata_rna.X.todense()
adata_rna.X = np.asarray(adata_rna.X)
adata_adt.X = adata_adt.X.todense().astype(np.float32)
adata_adt.X = np.asarray(adata_adt.X)

adata_rna.obs['batch'] = 'Brain'
adata_rna.layers['counts'] = adata_rna.X.copy()

mdata = md.MuData({"rna": adata_rna, "protein": adata_adt})

# Place subsetted counts in a new modality
mdata.mod["rna_subset"] = mdata.mod["rna"][:, mdata.mod["rna"].var["highly_variable"]].copy()

mdata.update()
scvi.model.TOTALVI.setup_mudata(mdata, rna_layer="counts",
                                protein_layer=None, batch_key="batch",
                                modalities={"rna_layer": "rna_subset",
                                            "protein_layer": "protein",
                                            "batch_key": "rna_subset",})
vae = scvi.model.TOTALVI(mdata)
vae.train()

rna = mdata.mod["rna_subset"]
protein = mdata.mod["protein"]
# arbitrarily store latent in rna modality
rna.obsm["X_totalVI"] = vae.get_latent_representation()
path = 'Data_SpatialGlue/' + dataset + '/'
rna.write_h5ad(path+dataset+'_totoalVI_results.h5ad', compression='gzip')





dataset = 'Dataset10_Mouse_Brain_H3K27me3'
adata_rna = sc.read_h5ad('Data_SpatialGlue/' + dataset + '/adata_RNA.h5ad')
adata_adt = sc.read_h5ad('Data_SpatialGlue/' + dataset + '/adata_peaks_normalized.h5ad')
adata_rna.var_names_make_unique()
adata_adt.var_names_make_unique()

sc.pp.highly_variable_genes(adata_rna, n_top_genes=4000, flavor="seurat_v3")

adata_rna.X = adata_rna.X.todense()
adata_rna.X = np.asarray(adata_rna.X)
adata_adt.X = adata_adt.X.todense().astype(np.float32)
adata_adt.X = np.asarray(adata_adt.X)

adata_rna.obs['batch'] = 'Brain'
adata_rna.layers['counts'] = adata_rna.X.copy()

mdata = md.MuData({"rna": adata_rna, "protein": adata_adt})

# Place subsetted counts in a new modality
mdata.mod["rna_subset"] = mdata.mod["rna"][:, mdata.mod["rna"].var["highly_variable"]].copy()

mdata.update()
scvi.model.TOTALVI.setup_mudata(mdata, rna_layer="counts",
                                protein_layer=None, batch_key="batch",
                                modalities={"rna_layer": "rna_subset",
                                            "protein_layer": "protein",
                                            "batch_key": "rna_subset",})
vae = scvi.model.TOTALVI(mdata)
vae.train()

rna = mdata.mod["rna_subset"]
protein = mdata.mod["protein"]
# arbitrarily store latent in rna modality
rna.obsm["X_totalVI"] = vae.get_latent_representation()
path = 'Data_SpatialGlue/' + dataset + '/'
rna.write_h5ad(path+dataset+'_totoalVI_results.h5ad', compression='gzip')
"""


