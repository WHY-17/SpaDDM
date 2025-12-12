import scanpy as sc
import flowsig as fs
import commot as ct



# load data
data_file = './GraphDiffusion/MISAR/E18_5-S1/'
adata_results = sc.read_h5ad(data_file + 'E18_5-S1_spatialDDM_results_noscale.h5ad')


adata = sc.read_h5ad(data_file+'adata_RNA.h5ad')
# Preprocessing the data
adata.var_names_make_unique()
adata.raw = adata
sc.pp.filter_genes(adata, min_cells=10)
sc.pp.filter_cells(adata, min_genes=10)
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata = adata[:, adata.var.highly_variable]


# Commot pipeline
"""
Spatial communication inference
We will use the CellChatDB ligand-receptor database here.
Only the secreted signaling LR pairs will be used.
"""
df_cellchat = ct.pp.ligand_receptor_database(species='mouse', signaling_type='Secreted Signaling', database='CellChat')
print(df_cellchat.shape)


# We then filter the LR pairs to keep only the pairs with both ligand and receptor expressed in at least 1% of the spots.
df_cellchat_filtered = ct.pp.filter_lr_database(df_cellchat, adata, min_cell_pct=0.01)
print(df_cellchat_filtered.shape)


"""
Now perform spatial communication inference for these 24 ligand-receptor pairs with a spatial distance limit of 500.
CellChat database considers heteromeric units.
The signaling results are stored as spot-by-spot matrices in the obsp slots.
For example, the score for spot i signaling to spot j through the LR pair
can be retrieved from adata_dis500.obsp['commot-cellchat-Wnt4-Fzd4_Lrp6'][i,j].
"""


ct.tl.spatial_communication(adata, database_name='cellchat',
                            df_ligrec=df_cellchat_filtered, dis_thr=25,
                            heteromeric=True, pathway_sum=True)


"""
adata.write_h5ad(data_file+"adata_commot_dis25.h5ad", compression='gzip')
ct.tl.communication_direction(adata, database_name='cellchat', pathway_name='APJ', k=5)
ct.pl.plot_cell_communication(adata, database_name='cellchat', pathway_name='APJ', plot_method='grid', background_legend=True,
    scale=0.00003, ndsize=8, grid_density=0.4, summary='sender')
"""

# flowsig pipelines
# We construct 20 gene expression modules from the unnormalized spot counts using NSF.
"""
fs.pp.construct_gems_using_nsf(adata,
                            n_gems = 20,
                            layer_key = None,
                            length_scale = 5.0)

adata.write(data_flie+"adata_nsf.h5ad", compression='gzip')
"""
adata = sc.read_h5ad(data_file+"adata_commot_dis25.h5ad")

"""
import matplotlib.pyplot as plt
pts = adata.obsm['spatial']
s = adata.obsm['commot-cellchat-sum-sender']['s-Fgf3-Fgfr1']
r = adata.obsm['commot-cellchat-sum-receiver']['r-Fgf3-Fgfr1']
fig, ax = plt.subplots(1,2, figsize=(8,4))
ax[0].scatter(pts[:,0], pts[:,1], c=s, s=20, cmap='Blues')
ax[0].invert_yaxis()  # 颠倒 y 轴方向
ax[0].set_title('Sender', fontsize=10)
ax[0].get_yaxis().set_visible(False)
ax[0].get_xaxis().set_visible(False)
ax[0].axis("off")
ax[0].axis("off")

ax[1].scatter(pts[:,0], pts[:,1], c=r, s=20, cmap='Reds')
ax[1].invert_yaxis()  # 颠倒 y 轴方向
ax[1].set_title('Receiver', fontsize=10)
ax[1].get_yaxis().set_visible(False)
ax[1].get_xaxis().set_visible(False)
ax[1].axis("off")
ax[1].axis("off")
plt.savefig(data_file+'Fgf3-Fgfr1_expression.pdf', dpi=300)


ct.tl.communication_direction(adata, database_name='cellchat', lr_pair=('Fgf3', 'Fgfr1'), k=5)
ct.pl.plot_cell_communication(adata, database_name='cellchat', lr_pair=('Fgf3', 'Fgfr1'), 
                              plot_method='grid', background_legend=True,
                              scale=0.02, ndsize=20, grid_density=0.3, summary='sender', 
                              background='summary', clustering='Combined_Clusters_annotation', cmap='Reds',
                              normalize_v = True, normalize_v_quantile=0.995) 
plt.savefig(data_file+'Fgf3-Fgfr1_sender.png', dpi=300)
                                                    
ct.pl.plot_cell_communication(adata, database_name='cellchat', lr_pair=('Fgf3', 'Fgfr1'), 
                              plot_method='grid', background_legend=True,
                              scale=0.03, ndsize=20, grid_density=0.3, summary='receiver', 
                              background='summary', clustering='Combined_Clusters_annotation', cmap='Reds',
                              normalize_v = True, normalize_v_quantile=0.995)
plt.savefig(data_file+'Fgf3-Fgfr1_receptor.png', dpi=300)    


img = plt.imread(data_file+"Fgf3-Fgfr1_sender.png")
plt.imshow(img)
plt.axis('off')
plt.gca().invert_yaxis()
plt.title('Sender', fontsize=10)
plt.savefig(data_file+'Fgf3-Fgfr1_sender.pdf', dpi=300) 

img = plt.imread(data_file+"Fgf3-Fgfr1_receptor.png")
plt.imshow(img)
plt.axis('off')
plt.gca().invert_yaxis()
plt.title('Receiver', fontsize=10)
plt.savefig(data_file+'Fgf3-Fgfr1_receiver.pdf', dpi=300) 
                      
"""



commot_output_key = 'commot-cellchat'
# We first construct the potential cellular flows from the COMMOT output, which has been run previously.
adata.obsm['SpatialDDM_pca'] = adata_results.obsm['SpatialDDM_pca']
fs.pp.construct_flows_from_commot(adata,
                                commot_output_key,
                                gem_expr_key = 'SpatialDDM_pca',
                                scale_gem_expr = False,
                                flowsig_network_key = 'flowsig_network',
                                flowsig_expr_key = 'X_flow')

"""
# Then we subset for "spatially flowing" variables
fs.pp.determine_informative_variables(adata,
                                    flowsig_expr_key = 'X_flow',
                                    flowsig_network_key = 'flowsig_network',
                                    spatial = True,
                                    moran_threshold = 0.05,
                                    coord_type = 'grid',
                                    n_neighbours = 8)
"""


"""
For spatial data, we need to construct spatial blocks that are used for block bootstrapping, 
to preserve the spatial correlation of the gene expression data. 
The idea is that by sampling within these spatial blocks, 
we will better preserve these spatial correlation structures during bootstrapping. 
We construct the blocks using simple K-Means clustering over the spatial locations.
"""
fs.pp.construct_spatial_blocks(adata,
                             n_blocks=20,
                             use_graph=False,
                             spatial_block_key = "spatial_block",
                             spatial_key = "spatial")

# Now we are ready to learn the network
fs.tl.learn_intercellular_flows(adata,
                        flowsig_key = 'flowsig_network',
                        flow_expr_key = 'X_flow',
                        use_spatial = True,
                        block_key = 'spatial_block',
                        n_jobs = 6,
                        n_bootstraps = 10)



"""
Now we do post-learning validation to reorient undirected edges from the learnt CPDAG
so that they flow from inflow to GEM to outflow. After that, we remove low-confidence edges.
"""
# This part is key for reducing false positives
fs.tl.apply_biological_flow(adata,
                        flowsig_network_key = 'flowsig_network',
                        adjacency_key = 'adjacency',
                        validated_key = 'adjacency_validated')

edge_threshold = 0.7
fs.tl.filter_low_confidence_edges(adata,
                                edge_threshold = edge_threshold,
                                flowsig_network_key = 'flowsig_network',
                                adjacency_key = 'adjacency',
                                filtered_key = 'adjacency_filtered')

flow_network =  fs.tl.construct_intercellular_flow_network(adata, adjacency_key='adjacency')
fs.pl.plot_intercellular_flows(adata, flow_network)


import networkx as nx
nx.write_gexf(flow_network, data_file+"flowsig_network.gexf")
adata.write_h5ad(data_file + "adata_flow_network.h5ad", compression='gzip')

