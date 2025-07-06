import os
import pickle
import numpy as np
import scanpy as sc
import anndata
import pandas as pd
import seaborn as sns
from .preprocess import pca
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

#os.environ['R_HOME'] = '/scbio4/tools/R/R-4.0.3_openblas/R-4.0.3'    

def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', random_seed=2020):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata

def clustering(adata, n_clusters=7, key='emb', add_key='SpatialGlue', method='mclust', start=0.1, end=3.0, increment=0.01, use_pca=False, n_comps=20):
    """\
    Spatial clustering based the latent representation.

    Parameters
    ----------
    adata : anndata
        AnnData object of scanpy package.
    n_clusters : int, optional
        The number of clusters. The default is 7.
    key : string, optional
        The key of the input representation in adata.obsm. The default is 'emb'.
    method : string, optional
        The tool for clustering. Supported tools include 'mclust', 'leiden', and 'louvain'. The default is 'mclust'. 
    start : float
        The start value for searching. The default is 0.1. Only works if the clustering method is 'leiden' or 'louvain'.
    end : float 
        The end value for searching. The default is 3.0. Only works if the clustering method is 'leiden' or 'louvain'.
    increment : float
        The step size to increase. The default is 0.01. Only works if the clustering method is 'leiden' or 'louvain'.  
    use_pca : bool, optional
        Whether use pca for dimension reduction. The default is false.

    Returns
    -------
    None.

    """
    
    if use_pca:
       adata.obsm[key + '_pca'], adata.uns['pca_weight'] = pca(adata, use_reps=key, n_comps=n_comps)
    
    if method == 'mclust':
       if use_pca: 
          adata = mclust_R(adata, used_obsm=key + '_pca', num_cluster=n_clusters)
       else:
          adata = mclust_R(adata, used_obsm=key, num_cluster=n_clusters)
       adata.obs[add_key] = adata.obs['mclust']
    elif method == 'leiden':
       if use_pca: 
          res = search_res(adata, n_clusters, use_rep=key + '_pca', method=method, start=start, end=end, increment=increment)
       else:
          res = search_res(adata, n_clusters, use_rep=key, method=method, start=start, end=end, increment=increment) 
       sc.tl.leiden(adata, random_state=0, resolution=res)
       adata.obs[add_key] = adata.obs['leiden']
    elif method == 'louvain':
       if use_pca: 
          res = search_res(adata, n_clusters, use_rep=key + '_pca', method=method, start=start, end=end, increment=increment)
       else:
          res = search_res(adata, n_clusters, use_rep=key, method=method, start=start, end=end, increment=increment) 
       sc.tl.louvain(adata, random_state=0, resolution=res)
       adata.obs[add_key] = adata.obs['louvain']

       
def search_res(adata, n_clusters, method='leiden', use_rep='emb', start=0.1, end=3.0, increment=0.01):
    '''\
    Searching corresponding resolution according to given cluster number
    
    Parameters
    ----------
    adata : anndata
        AnnData object of spatial data.
    n_clusters : int
        Targetting number of clusters.
    method : string
        Tool for clustering. Supported tools include 'leiden' and 'louvain'. The default is 'leiden'.    
    use_rep : string
        The indicated representation for clustering.
    start : float
        The start value for searching.
    end : float 
        The end value for searching.
    increment : float
        The step size to increase.
        
    Returns
    -------
    res : float
        Resolution.
        
    '''
    print('Searching resolution...')
    label = 0
    sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)
    for res in sorted(list(np.arange(start, end, increment)), reverse=True):
        if method == 'leiden':
           sc.tl.leiden(adata, random_state=0, resolution=res)
           count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
           print('resolution={}, cluster number={}'.format(res, count_unique))
        elif method == 'louvain':
           sc.tl.louvain(adata, random_state=0, resolution=res)
           count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique()) 
           print('resolution={}, cluster number={}'.format(res, count_unique))
        if count_unique == n_clusters:
            label = 1
            break

    assert label==1, "Resolution is not found. Please try bigger range or smaller step!." 
       
    return res     

def plot_weight_value(alpha, label, modality1='mRNA', modality2='protein'):
  """\
  Plotting weight values
  
  """  
  import pandas as pd  
  
  df = pd.DataFrame(columns=[modality1, modality2, 'label'])  
  df[modality1], df[modality2] = alpha[:, 0], alpha[:, 1]
  df['label'] = label
  df = df.set_index('label').stack().reset_index()
  df.columns = ['label_SpatialGlue', 'Modality', 'Weight value']
  ax = sns.violinplot(data=df, x='label_SpatialGlue', y='Weight value', hue="Modality",
                split=True, inner="quart", linewidth=1, show=False)
  ax.set_title(modality1 + ' vs ' + modality2) 

  plt.tight_layout(w_pad=0.05)
  plt.show()     




def pseudo_Spatiotemporal_Map(adata_all, emb_name, n_neighbors=20, resolution=1.0):
    """
    Perform pseudo-Spatiotemporal Map for ST data
    :param pSM_values_save_filepath: the default save path for the pSM values
    :type pSM_values_save_filepath: class:`str`, optional, default: "./pSM_values.tsv"
    :param n_neighbors: The size of local neighborhood (in terms of number of neighboring data
    points) used for manifold approximation. See `https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.neighbors.html` for detail
    :type n_neighbors: int, optional, default: 20
    :param resolution: A parameter value controlling the coarseness of the clustering.
    Higher values lead to more clusters. See `https://scanpy.readthedocs.io/en/stable/generated/scanpy.tl.leiden.html` for detail
    :type resolution: float, optional, default: 1.0
    """
    error_message = "No embedding found, please ensure you have run train() method before calculating pseudo-Spatiotemporal Map!"
    max_cell_for_subsampling = 5000
    try:
        print("Performing pseudo-Spatiotemporal Map")
        adata = anndata.AnnData(adata_all.obsm[emb_name])
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep='X')
        sc.tl.umap(adata)
        sc.tl.leiden(adata, resolution=resolution)
        """
        adata.obs['leiden'] = pd.Categorical(adata_all.obs['pred_label'])
        """
        sc.tl.paga(adata)
        if adata.shape[0] < max_cell_for_subsampling:
            sub_adata_x = adata.X
        else:
            indices = np.arange(adata.shape[0])
            selected_ind = np.random.choice(indices, max_cell_for_subsampling, False)
            sub_adata_x = adata.X[selected_ind, :]
        sum_dists = distance_matrix(sub_adata_x, sub_adata_x).sum(axis=1)
        adata.uns['iroot'] = np.argmax(sum_dists)
        sc.tl.diffmap(adata)
        sc.tl.dpt(adata)
        pSM_values = adata.obs['dpt_pseudotime'].to_numpy()
        '''
        save_dir = os.path.dirname(pSM_values_save_filepath)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.savetxt(pSM_values_save_filepath, pSM_values, fmt='%.5f', header='', footer='', comments='')
        print(
            f"pseudo-Spatiotemporal Map(pSM) calculation complete, pSM values of cells or spots saved at {pSM_values_save_filepath}!")
        '''
        adata_all.obsm['pSM_values'] = pSM_values
    except NameError:
        print(error_message)
    except AttributeError:
        print(error_message)



def prepare_figure(rsz=4., csz=4., wspace=.4, hspace=.5, left=0.125, right=0.9, bottom=0.1, top=0.9):
    """
    Prepare the figure and axes given the configuration
    :param rsz: row size of the figure in inches, default: 4.0
    :type rsz: float, optional
    :param csz: column size of the figure in inches, default: 4.0
    :type csz: float, optional
    :param wspace: the amount of width reserved for space between subplots, expressed as a fraction of the average axis width, default: 0.4
    :type wspace: float, optional
    :param hspace: the amount of height reserved for space between subplots, expressed as a fraction of the average axis width, default: 0.4
    :type hspace: float, optional
    :param left: the leftmost position of the subplots of the figure in fraction, default: 0.125
    :type left: float, optional
    :param right: the rightmost position of the subplots of the figure in fraction, default: 0.9
    :type right: float, optional
    :param bottom: the bottom position of the subplots of the figure in fraction, default: 0.1
    :type bottom: float, optional
    :param top: the top position of the subplots of the figure in fraction, default: 0.9
    :type top: float, optional
    """
    fig, axs = plt.subplots(1, 1, figsize=(csz, rsz))
    plt.subplots_adjust(wspace=wspace, hspace=hspace, left=left, right=right, bottom=bottom, top=top)
    return fig, axs




def plot_pSM(adata, camp, colormap='roma', scatter_sz=1., rsz=4.,
             csz=4., wspace=.4, hspace=.5, left=0.125, right=0.9, bottom=0.1, top=0.9):
    """
    Plot the domain segmentation for ST data in spatial
    :param pSM_figure_save_filepath: the default save path for the figure
    :type pSM_figure_save_filepath: class:`str`, optional, default: "./Spatiotemporal-Map.pdf"
    :param colormap: The colormap to use. See `https://www.fabiocrameri.ch/colourmaps-userguide/` for name list of colormaps
    :type colormap: str, optional, default: roma
    :param scatter_sz: The marker size in points**2
    :type scatter_sz: float, optional, default: 1.0
    :param rsz: row size of the figure in inches, default: 4.0
    :type rsz: float, optional
    :param csz: column size of the figure in inches, default: 4.0
    :type csz: float, optional
    :param wspace: the amount of width reserved for space between subplots, expressed as a fraction of the average axis width, default: 0.4
    :type wspace: float, optional
    :param hspace: the amount of height reserved for space between subplots, expressed as a fraction of the average axis width, default: 0.4
    :type hspace: float, optional
    :param left: the leftmost position of the subplots of the figure in fraction, default: 0.125
    :type left: float, optional
    :param right: the rightmost position of the subplots of the figure in fraction, default: 0.9
    :type right: float, optional
    :param bottom: the bottom position of the subplots of the figure in fraction, default: 0.1
    :type bottom: float, optional
    :param top: the top position of the subplots of the figure in fraction, default: 0.9
    :type top: float, optional
    """
    error_message = "No pseudo Spatiotemporal Map data found, please ensure you have run the pseudo_Spatiotemporal_Map() method."
    try:
        fig, ax = prepare_figure(rsz=rsz, csz=csz, wspace=wspace, hspace=hspace, left=left, right=right,
                                      bottom=bottom, top=top)
        x, y = adata.obsm["spatial"][:, 0], adata.obsm["spatial"][:, 1]
        st = ax.scatter(x, y, s=scatter_sz, c=adata.obsm['pSM_values'], cmap=camp, marker=".", edgecolors='face')
        ax.invert_yaxis()
        clb = fig.colorbar(st)
        clb.ax.set_ylabel("pseudotime", labelpad=10, rotation=270, fontsize=10, weight='bold')
        ax.set_title("pseudo-Spatiotemporal Map", fontsize=14)
        ax.set_facecolor("none")
        '''
        save_dir = os.path.dirname(pSM_figure_save_filepath)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(pSM_figure_save_filepath, dpi=300)
        print(f"Plotting complete, pseudo-Spatiotemporal Map figure saved at {pSM_figure_save_filepath} !")
        plt.close('all')
        '''
    except NameError:
        print(error_message)
    except AttributeError:
        print(error_message)
