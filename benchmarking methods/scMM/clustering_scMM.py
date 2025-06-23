import pandas as pd
import scanpy as sc


def clustering(adata, path):
    # read scMM output
    df_train = pd.read_csv(path + 'lat_train_mean.csv', index_col=0)
    df_test = pd.read_csv(path + 'lat_test_mean.csv', index_col=0)
    id_train = pd.read_csv(path + 't_id.csv', index_col=0)
    id_test = pd.read_csv(path + 's_id.csv', index_col=0)

    # combine latent representation with cell id
    df = pd.concat((df_train, df_test))
    id = pd.concat((id_train, id_test))
    id.rename(columns={"0": "ID"}, inplace=True)

    df = pd.concat((id, df), axis=1)
    df.set_index('ID', inplace=True)
    df.sort_index(inplace=True)

    adata_new = sc.AnnData(df,
                           df.index.to_frame(),
                           df.columns.to_frame())
    adata_new.obsm['spatial'] = adata.obsm['spatial']

    # add latent representation to adata
    adata.obsm['X_scMM'] = adata_new.X

    # downstream analysis
    # sc.pp.neighbors(adata_new, n_neighbors=10)
    # sc.tl.umap(adata_new)
    # sc.tl.leiden(adata_new, resolution=0.5)

    # sc.pl.umap(adata_new, color='leiden')
    # sc.pl.embedding(adata_new, basis='spatial', color='leiden',s=60)

    return adata



dataset = 'Dataset7_Mouse_Brain_ATAC'
path = '../Data/' + dataset + '/test/'
adata = sc.read_h5ad('../Data/' + dataset + '/adata_RNA.h5ad')
adata = clustering(adata, path)


# save result
adata.write_h5ad('../Data/' + dataset + '/scMM_results.h5ad', compression='gzip')
