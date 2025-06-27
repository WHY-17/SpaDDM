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
sc.pp.normalize_total(adata, inplace=True)
sc.pp.log1p(adata)

sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
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

"""
ct.tl.spatial_communication(adata, database_name='cellchat',
                            df_ligrec=df_cellchat_filtered, dis_thr=500,
                            heteromeric=True, pathway_sum=True)
"""

"""
adata.write_h5ad(data_flie+"adata_commot.h5ad", compression='gzip')
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
s = adata.obsm['commot-cellchat-sum-sender']['s-Bmp4-Bmpr1a_Acvr2b']
r = adata.obsm['commot-cellchat-sum-receiver']['r-Bmp4-Bmpr1a_Acvr2b']
fig, ax = plt.subplots(1,2, figsize=(10,4))
ax[0].scatter(pts[:,0], pts[:,1], c=s, s=20, cmap='Blues')
ax[0].invert_yaxis()  # 颠倒 y 轴方向
ax[0].set_title('Sender')
ax[1].scatter(pts[:,0], pts[:,1], c=r, s=20, cmap='Reds')
ax[1].invert_yaxis()  # 颠倒 y 轴方向
ax[1].set_title('Receiver')
plt.savefig(data_file+'Bmp4-Bmpr1a_Acvr2b_expression.pdf', dpi=300)


ct.tl.communication_direction(adata, database_name='cellchat', lr_pair=('Bmp4', 'Bmpr1a_Acvr2b'), k=5)
ct.pl.plot_cell_communication(adata, database_name='cellchat', lr_pair=('Bmp4', 'Bmpr1a_Acvr2b'), 
                              plot_method='grid', background_legend=True,
                              scale=0.03, ndsize=20, grid_density=0.3, summary='sender', 
                              background='summary', clustering='Combined_Clusters_annotation', cmap='Reds',
                              normalize_v = True, normalize_v_quantile=0.995)                        
ct.pl.plot_cell_communication(adata, database_name='cellchat', lr_pair=('Bmp4', 'Bmpr1a_Acvr2b'), 
                              plot_method='grid', background_legend=True,
                              scale=0.03, ndsize=20, grid_density=0.3, summary='receiver', 
                              background='summary', clustering='Combined_Clusters_annotation', cmap='Reds',
                              normalize_v = True, normalize_v_quantile=0.995)
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

"""
inflow-Igf1, inflow-Bmp6, inflow-Plg, inflow-Pros1, inflow-Bmp7, inflow-Wnt6, inflow-Igf2,
inflow-Wnt2, inflow-Tgfb3, inflow-Wnt7b, inflow-Bmp4, inflow-Sema3d, inflow-Postn, inflow-Wnt5a,
inflow-Tgfb1, inflow-Ccl1
"""


import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
threshold = 0.7

"""
"inflow-Sema3d" regulate downstream targets
"""
network = nx.DiGraph()
# 第一层连接
in_edges = list(flow_network.out_edges("inflow-Sema3d", data=True))
out_edges = [(source, target, data['weight']) for source, target, data in in_edges]
network.add_weighted_edges_from(out_edges)
# 第二层连接
in_edges_nodes = [x[1] for x in in_edges]
for node in in_edges_nodes:
    out_edges = list(flow_network.out_edges(node, data=True))
    out_edges = [(source, target, data['weight']) for source, target, data in out_edges]
    network.add_weighted_edges_from(out_edges)
# 查看节点和边
print("Nodes:", network.nodes)
print("Edges with weights:")
for u, v, data in network.edges(data=True):
    print(f"{u} -> {v} with weight: {data['weight']}")

network.remove_node("GEM-5")
network.remove_node("GEM-6")
# 删除权重小于0.5的边
edges_to_remove = [(u, v) for u, v, weight in network.edges(data=True) if weight['weight'] < 0.5]
network.remove_edges_from(edges_to_remove)



"""
"inflow-Igf1" regulate downstream targets
"""
network = nx.DiGraph()
# 第一层连接
in_edges = list(flow_network.out_edges("inflow-Igf1", data=True))
out_edges = [(source, target, data['weight']) for source, target, data in in_edges]
network.add_weighted_edges_from(out_edges)
# 第二层连接
in_edges_nodes = [x[1] for x in in_edges]
for node in in_edges_nodes:
    out_edges = list(flow_network.out_edges(node, data=True))
    out_edges = [(source, target, data['weight']) for source, target, data in out_edges]
    network.add_weighted_edges_from(out_edges)
# 查看节点和边
print("Nodes:", network.nodes)
print("Edges with weights:")
for u, v, data in network.edges(data=True):
    print(f"{u} -> {v} with weight: {data['weight']}")



"""
"inflow-Postn" regulate downstream targets
"""
network = nx.DiGraph()
# 第一层连接
in_edges = list(flow_network.out_edges("inflow-Postn", data=True))
out_edges = [(source, target, data['weight']) for source, target, data in in_edges]
network.add_weighted_edges_from(out_edges)
# 第二层连接
in_edges_nodes = [x[1] for x in in_edges]
for node in in_edges_nodes:
    out_edges = list(flow_network.out_edges(node, data=True))
    out_edges = [(source, target, data['weight']) for source, target, data in out_edges]
    network.add_weighted_edges_from(out_edges)
# 查看节点和边
print("Nodes:", network.nodes)
print("Edges with weights:")
for u, v, data in network.edges(data=True):
    print(f"{u} -> {v} with weight: {data['weight']}")
network.remove_node("GEM-15")



"""
"inflow-Igf2" regulate downstream targets
"""
network = nx.DiGraph()
# 第一层连接
in_edges = list(flow_network.out_edges("inflow-Igf2", data=True))
out_edges = [(source, target, data['weight']) for source, target, data in in_edges]
network.add_weighted_edges_from(out_edges)
# 第二层连接
in_edges_nodes = [x[1] for x in in_edges]
for node in in_edges_nodes:
    out_edges = list(flow_network.out_edges(node, data=True))
    out_edges = [(source, target, data['weight']) for source, target, data in out_edges]
    network.add_weighted_edges_from(out_edges)
# 查看节点和边
print("Nodes:", network.nodes)
print("Edges with weights:")
for u, v, data in network.edges(data=True):
    print(f"{u} -> {v} with weight: {data['weight']}")

network.remove_node("GEM-15")
network.remove_node("GEM-19")
network.remove_node("GEM-20")




"""
"inflow-Bmp6" regulate downstream targets
"""
network = nx.DiGraph()
# 第一层连接
in_edges = list(flow_network.out_edges("inflow-Bmp6", data=True))
out_edges = [(source, target, data['weight']) for source, target, data in in_edges]
network.add_weighted_edges_from(out_edges)
# 第二层连接
in_edges_nodes = [x[1] for x in in_edges]
for node in in_edges_nodes:
    out_edges = list(flow_network.out_edges(node, data=True))
    out_edges = [(source, target, data['weight']) for source, target, data in out_edges]
    network.add_weighted_edges_from(out_edges)
# 查看节点和边
print("Nodes:", network.nodes)
print("Edges with weights:")
for u, v, data in network.edges(data=True):
    print(f"{u} -> {v} with weight: {data['weight']}")
network.remove_node("GEM-15")
network.remove_node("GEM-17")
network.remove_node("GEM-19")
network.remove_node("GEM-20")
network.remove_node("GEM-10")
"""
# 删除权重小于0.5的边
edges_to_remove = [(u, v) for u, v, weight in network.edges(data=True) if weight['weight'] < 0.5]
network.remove_edges_from(edges_to_remove)
"""


"""
"inflow-Bmp7" regulate downstream targets
"""
network = nx.DiGraph()
# 第一层连接
in_edges = list(flow_network.out_edges("inflow-Bmp7", data=True))
out_edges = [(source, target, data['weight']) for source, target, data in in_edges]
network.add_weighted_edges_from(out_edges)
# 第二层连接
in_edges_nodes = [x[1] for x in in_edges]
for node in in_edges_nodes:
    out_edges = list(flow_network.out_edges(node, data=True))
    out_edges = [(source, target, data['weight']) for source, target, data in out_edges]
    network.add_weighted_edges_from(out_edges)
# 查看节点和边
print("Nodes:", network.nodes)
print("Edges with weights:")
for u, v, data in network.edges(data=True):
    print(f"{u} -> {v} with weight: {data['weight']}")
network.remove_node("GEM-15")
network.remove_node("GEM-14")
network.remove_node("GEM-6")
network.remove_node("GEM-8")
network.remove_node("GEM-10")

# 删除权重小于0.5的边
edges_to_remove = [(u, v) for u, v, weight in network.edges(data=True) if weight['weight'] < 0.5]
network.remove_edges_from(edges_to_remove)
network.remove_node("GEM-1")
network.remove_node("GEM-7")
network.remove_node("GEM-18")



"""
"inflow-Plg" regulate downstream targets
"""
network = nx.DiGraph()
# 第一层连接
in_edges = list(flow_network.out_edges("inflow-Plg", data=True))
out_edges = [(source, target, data['weight']) for source, target, data in in_edges]
network.add_weighted_edges_from(out_edges)
# 第二层连接
in_edges_nodes = [x[1] for x in in_edges]
for node in in_edges_nodes:
    out_edges = list(flow_network.out_edges(node, data=True))
    out_edges = [(source, target, data['weight']) for source, target, data in out_edges]
    network.add_weighted_edges_from(out_edges)
# 查看节点和边
print("Nodes:", network.nodes)
print("Edges with weights:")
for u, v, data in network.edges(data=True):
    print(f"{u} -> {v} with weight: {data['weight']}")
network.remove_node("GEM-15")
network.remove_node("GEM-8")
network.remove_node("GEM-10")
network.remove_node("GEM-17")
"""
# 删除权重小于0.5的边
edges_to_remove = [(u, v) for u, v, weight in network.edges(data=True) if weight['weight'] < 0.5]
network.remove_edges_from(edges_to_remove)
network.remove_node("GEM-12")
"""


"""
"inflow-Bmp4" regulate downstream targets
"""
network = nx.DiGraph()
# 第一层连接
in_edges = list(flow_network.out_edges("inflow-Bmp4", data=True))
out_edges = [(source, target, data['weight']) for source, target, data in in_edges]
network.add_weighted_edges_from(out_edges)
# 第二层连接
in_edges_nodes = [x[1] for x in in_edges]
for node in in_edges_nodes:
    out_edges = list(flow_network.out_edges(node, data=True))
    out_edges = [(source, target, data['weight']) for source, target, data in out_edges]
    network.add_weighted_edges_from(out_edges)
# 查看节点和边
print("Nodes:", network.nodes)
print("Edges with weights:")
for u, v, data in network.edges(data=True):
    print(f"{u} -> {v} with weight: {data['weight']}")
network.remove_node("GEM-14")
network.remove_node("GEM-20")
"""
# 删除权重小于0.5的边
edges_to_remove = [(u, v) for u, v, weight in network.edges(data=True) if weight['weight'] < 0.5]
network.remove_edges_from(edges_to_remove)
network.remove_node("GEM-7")
network.remove_node("GEM-19")
"""


"""
"inflow-Wnt2" regulate downstream targets
"""
network = nx.DiGraph()
# 第一层连接
in_edges = list(flow_network.out_edges("inflow-Wnt2", data=True))
out_edges = [(source, target, data['weight']) for source, target, data in in_edges]
network.add_weighted_edges_from(out_edges)
# 第二层连接
in_edges_nodes = [x[1] for x in in_edges]
for node in in_edges_nodes:
    out_edges = list(flow_network.out_edges(node, data=True))
    out_edges = [(source, target, data['weight']) for source, target, data in out_edges]
    network.add_weighted_edges_from(out_edges)
# 查看节点和边
print("Nodes:", network.nodes)
print("Edges with weights:")
for u, v, data in network.edges(data=True):
    print(f"{u} -> {v} with weight: {data['weight']}")
network.remove_node("GEM-14")
network.remove_node("GEM-20")
"""
# 删除权重小于0.5的边
edges_to_remove = [(u, v) for u, v, weight in network.edges(data=True) if weight['weight'] < 0.5]
network.remove_edges_from(edges_to_remove)
"""



"""
"inflow-Wnt6" regulate downstream targets
"""
network = nx.DiGraph()
# 第一层连接
in_edges = list(flow_network.out_edges("inflow-Wnt6", data=True))
out_edges = [(source, target, data['weight']) for source, target, data in in_edges]
network.add_weighted_edges_from(out_edges)
# 第二层连接
in_edges_nodes = [x[1] for x in in_edges]
for node in in_edges_nodes:
    out_edges = list(flow_network.out_edges(node, data=True))
    out_edges = [(source, target, data['weight']) for source, target, data in out_edges]
    network.add_weighted_edges_from(out_edges)
# 查看节点和边
print("Nodes:", network.nodes)
print("Edges with weights:")
for u, v, data in network.edges(data=True):
    print(f"{u} -> {v} with weight: {data['weight']}")
network.remove_node("GEM-20")
"""
# 删除权重小于0.5的边
edges_to_remove = [(u, v) for u, v, weight in network.edges(data=True) if weight['weight'] < 0.5]
network.remove_edges_from(edges_to_remove)
network.remove_node("GEM-19")
"""


"""
"inflow-Tgfb1" regulate downstream targets
"""
network = nx.DiGraph()
# 第一层连接
in_edges = list(flow_network.out_edges("inflow-Tgfb1", data=True))
out_edges = [(source, target, data['weight']) for source, target, data in in_edges]
network.add_weighted_edges_from(out_edges)
# 第二层连接
in_edges_nodes = [x[1] for x in in_edges]
for node in in_edges_nodes:
    out_edges = list(flow_network.out_edges(node, data=True))
    out_edges = [(source, target, data['weight']) for source, target, data in out_edges]
    network.add_weighted_edges_from(out_edges)
# 查看节点和边
print("Nodes:", network.nodes)
print("Edges with weights:")
for u, v, data in network.edges(data=True):
    print(f"{u} -> {v} with weight: {data['weight']}")



"""
"inflow-Wnt7b" regulate downstream targets
"""
network = nx.DiGraph()
# 第一层连接
in_edges = list(flow_network.out_edges("inflow-Wnt7b", data=True))
out_edges = [(source, target, data['weight']) for source, target, data in in_edges]
network.add_weighted_edges_from(out_edges)
# 第二层连接
in_edges_nodes = [x[1] for x in in_edges]
for node in in_edges_nodes:
    out_edges = list(flow_network.out_edges(node, data=True))
    out_edges = [(source, target, data['weight']) for source, target, data in out_edges]
    network.add_weighted_edges_from(out_edges)
# 查看节点和边
print("Nodes:", network.nodes)
print("Edges with weights:")
for u, v, data in network.edges(data=True):
    print(f"{u} -> {v} with weight: {data['weight']}")



"""
"inflow-Wnt5a" regulate downstream targets
"""
network = nx.DiGraph()
# 第一层连接
in_edges = list(flow_network.out_edges("inflow-Wnt5a", data=True))
out_edges = [(source, target, data['weight']) for source, target, data in in_edges]
network.add_weighted_edges_from(out_edges)
# 第二层连接
in_edges_nodes = [x[1] for x in in_edges]
for node in in_edges_nodes:
    out_edges = list(flow_network.out_edges(node, data=True))
    out_edges = [(source, target, data['weight']) for source, target, data in out_edges]
    network.add_weighted_edges_from(out_edges)
# 查看节点和边
print("Nodes:", network.nodes)
print("Edges with weights:")
for u, v, data in network.edges(data=True):
    print(f"{u} -> {v} with weight: {data['weight']}")

network.remove_node("GEM-5")
network.remove_node("GEM-6")
network.remove_node("GEM-20")
"""
# 删除权重小于0.5的边
edges_to_remove = [(u, v) for u, v, weight in network.edges(data=True) if weight['weight'] < 0.5]
network.remove_edges_from(edges_to_remove)
"""



"""
"inflow-Tgfb3" regulate downstream targets
"""
network = nx.DiGraph()
# 第一层连接
in_edges = list(flow_network.out_edges("inflow-Tgfb3", data=True))
out_edges = [(source, target, data['weight']) for source, target, data in in_edges]
network.add_weighted_edges_from(out_edges)
# 第二层连接
in_edges_nodes = [x[1] for x in in_edges]
for node in in_edges_nodes:
    out_edges = list(flow_network.out_edges(node, data=True))
    out_edges = [(source, target, data['weight']) for source, target, data in out_edges]
    network.add_weighted_edges_from(out_edges)
# 查看节点和边
print("Nodes:", network.nodes)
print("Edges with weights:")
for u, v, data in network.edges(data=True):
    print(f"{u} -> {v} with weight: {data['weight']}")




"""
"inflow-Ccl1" regulate downstream targets
"""
network = nx.DiGraph()
# 第一层连接
in_edges = list(flow_network.out_edges("inflow-Ccl1", data=True))
out_edges = [(source, target, data['weight']) for source, target, data in in_edges]
network.add_weighted_edges_from(out_edges)
# 第二层连接
in_edges_nodes = [x[1] for x in in_edges]
for node in in_edges_nodes:
    out_edges = list(flow_network.out_edges(node, data=True))
    out_edges = [(source, target, data['weight']) for source, target, data in out_edges]
    network.add_weighted_edges_from(out_edges)
# 查看节点和边
print("Nodes:", network.nodes)
print("Edges with weights:")
for u, v, data in network.edges(data=True):
    print(f"{u} -> {v} with weight: {data['weight']}")



"""
"inflow-Pros1" regulate downstream targets
"""
network = nx.DiGraph()
# 第一层连接
in_edges = list(flow_network.out_edges("inflow-Pros1", data=True))
out_edges = [(source, target, data['weight']) for source, target, data in in_edges]
network.add_weighted_edges_from(out_edges)
# 第二层连接
in_edges_nodes = [x[1] for x in in_edges]
for node in in_edges_nodes:
    out_edges = list(flow_network.out_edges(node, data=True))
    out_edges = [(source, target, data['weight']) for source, target, data in out_edges]
    network.add_weighted_edges_from(out_edges)
# 查看节点和边
print("Nodes:", network.nodes)
print("Edges with weights:")
for u, v, data in network.edges(data=True):
    print(f"{u} -> {v} with weight: {data['weight']}")
network.remove_node("GEM-14")
network.remove_node("GEM-8")


"""
'Bmp7', 'Sema3d', 'Tgfb1', 'Wnt2', 'Tgfb3', 'Kitl', 'Wnt5a', 'Igf2', 'Pros1',
'Agt', 'Ccl1', 'Penk', 'Wnt6', 'Postn', 'Igf1', 'Wnt7b', 'Nampt', 'Bmp6', 'Bmp4', 'Plg'
"""

"""
received regulated signals
"""
network = nx.DiGraph()
# 第一层连接
in_edges = list(flow_network.in_edges("Sema3d", data=True))
in_edges = [(source, target, data['weight']) for source, target, data in in_edges]
network.add_weighted_edges_from(in_edges)
# 第二层连接
in_edges_nodes = [x[0] for x in in_edges]
for node in in_edges_nodes:
    in_edges = list(flow_network.in_edges(node, data=True))
    in_edges = [(source, target, data['weight']) for source, target, data in in_edges]
    network.add_weighted_edges_from(in_edges)
# 查看节点和边
print("Nodes:", network.nodes)
print("Edges with weights:")
for u, v, data in network.edges(data=True):
    print(f"{u} -> {v} with weight: {data['weight']}")
network.remove_node("GEM-1")


"""
received regulated signals
"""
network = nx.DiGraph()
# 第一层连接
in_edges = list(flow_network.in_edges("Igf1", data=True))
in_edges = [(source, target, data['weight']) for source, target, data in in_edges]
network.add_weighted_edges_from(in_edges)
# 第二层连接
in_edges_nodes = [x[0] for x in in_edges]
for node in in_edges_nodes:
    in_edges = list(flow_network.in_edges(node, data=True))
    in_edges = [(source, target, data['weight']) for source, target, data in in_edges]
    network.add_weighted_edges_from(in_edges)
# 查看节点和边
print("Nodes:", network.nodes)
print("Edges with weights:")
for u, v, data in network.edges(data=True):
    print(f"{u} -> {v} with weight: {data['weight']}")



network = nx.DiGraph()
# 第一层连接
in_edges = list(flow_network.in_edges("Postn", data=True))
in_edges = [(source, target, data['weight']) for source, target, data in in_edges]
network.add_weighted_edges_from(in_edges)
# 第二层连接
in_edges_nodes = [x[0] for x in in_edges]
for node in in_edges_nodes:
    in_edges = list(flow_network.in_edges(node, data=True))
    in_edges = [(source, target, data['weight']) for source, target, data in in_edges]
    network.add_weighted_edges_from(in_edges)
# 查看节点和边
print("Nodes:", network.nodes)
print("Edges with weights:")
for u, v, data in network.edges(data=True):
    print(f"{u} -> {v} with weight: {data['weight']}")



network = nx.DiGraph()
# 第一层连接
in_edges = list(flow_network.in_edges("Igf2", data=True))
in_edges = [(source, target, data['weight']) for source, target, data in in_edges]
network.add_weighted_edges_from(in_edges)
# 第二层连接
in_edges_nodes = [x[0] for x in in_edges]
for node in in_edges_nodes:
    in_edges = list(flow_network.in_edges(node, data=True))
    in_edges = [(source, target, data['weight']) for source, target, data in in_edges]
    network.add_weighted_edges_from(in_edges)
# 查看节点和边
print("Nodes:", network.nodes)
print("Edges with weights:")
for u, v, data in network.edges(data=True):
    print(f"{u} -> {v} with weight: {data['weight']}")
network.remove_node("GEM-12")
network.remove_node("GEM-20")



network = nx.DiGraph()
# 第一层连接
in_edges = list(flow_network.in_edges("Bmp6", data=True))
in_edges = [(source, target, data['weight']) for source, target, data in in_edges]
network.add_weighted_edges_from(in_edges)
# 第二层连接
in_edges_nodes = [x[0] for x in in_edges]
for node in in_edges_nodes:
    in_edges = list(flow_network.in_edges(node, data=True))
    in_edges = [(source, target, data['weight']) for source, target, data in in_edges]
    network.add_weighted_edges_from(in_edges)
# 查看节点和边
print("Nodes:", network.nodes)
print("Edges with weights:")
for u, v, data in network.edges(data=True):
    print(f"{u} -> {v} with weight: {data['weight']}")
network.remove_node("GEM-1")



network = nx.DiGraph()
# 第一层连接
in_edges = list(flow_network.in_edges("Bmp7", data=True))
in_edges = [(source, target, data['weight']) for source, target, data in in_edges]
network.add_weighted_edges_from(in_edges)
# 第二层连接
in_edges_nodes = [x[0] for x in in_edges]
for node in in_edges_nodes:
    in_edges = list(flow_network.in_edges(node, data=True))
    in_edges = [(source, target, data['weight']) for source, target, data in in_edges]
    network.add_weighted_edges_from(in_edges)
# 查看节点和边
print("Nodes:", network.nodes)
print("Edges with weights:")
for u, v, data in network.edges(data=True):
    print(f"{u} -> {v} with weight: {data['weight']}")
network.remove_node("GEM-1")



network = nx.DiGraph()
# 第一层连接
in_edges = list(flow_network.in_edges("Plg", data=True))
in_edges = [(source, target, data['weight']) for source, target, data in in_edges]
network.add_weighted_edges_from(in_edges)
# 第二层连接
in_edges_nodes = [x[0] for x in in_edges]
for node in in_edges_nodes:
    in_edges = list(flow_network.in_edges(node, data=True))
    in_edges = [(source, target, data['weight']) for source, target, data in in_edges]
    network.add_weighted_edges_from(in_edges)
# 查看节点和边
print("Nodes:", network.nodes)
print("Edges with weights:")
for u, v, data in network.edges(data=True):
    print(f"{u} -> {v} with weight: {data['weight']}")
network.remove_node("GEM-20")


network = nx.DiGraph()
# 第一层连接
in_edges = list(flow_network.in_edges("Bmp4", data=True))
in_edges = [(source, target, data['weight']) for source, target, data in in_edges]
network.add_weighted_edges_from(in_edges)
# 第二层连接
in_edges_nodes = [x[0] for x in in_edges]
for node in in_edges_nodes:
    in_edges = list(flow_network.in_edges(node, data=True))
    in_edges = [(source, target, data['weight']) for source, target, data in in_edges]
    network.add_weighted_edges_from(in_edges)
# 查看节点和边
print("Nodes:", network.nodes)
print("Edges with weights:")
for u, v, data in network.edges(data=True):
    print(f"{u} -> {v} with weight: {data['weight']}")



network = nx.DiGraph()
# 第一层连接
in_edges = list(flow_network.in_edges("Wnt2", data=True))
in_edges = [(source, target, data['weight']) for source, target, data in in_edges]
network.add_weighted_edges_from(in_edges)
# 第二层连接
in_edges_nodes = [x[0] for x in in_edges]
for node in in_edges_nodes:
    in_edges = list(flow_network.in_edges(node, data=True))
    in_edges = [(source, target, data['weight']) for source, target, data in in_edges]
    network.add_weighted_edges_from(in_edges)
# 查看节点和边
print("Nodes:", network.nodes)
print("Edges with weights:")
for u, v, data in network.edges(data=True):
    print(f"{u} -> {v} with weight: {data['weight']}")
network.remove_node("GEM-1")



network = nx.DiGraph()
# 第一层连接
in_edges = list(flow_network.in_edges("Wnt6", data=True))
in_edges = [(source, target, data['weight']) for source, target, data in in_edges]
network.add_weighted_edges_from(in_edges)
# 第二层连接
in_edges_nodes = [x[0] for x in in_edges]
for node in in_edges_nodes:
    in_edges = list(flow_network.in_edges(node, data=True))
    in_edges = [(source, target, data['weight']) for source, target, data in in_edges]
    network.add_weighted_edges_from(in_edges)
# 查看节点和边
print("Nodes:", network.nodes)
print("Edges with weights:")
for u, v, data in network.edges(data=True):
    print(f"{u} -> {v} with weight: {data['weight']}")
network.remove_node("GEM-20")



network = nx.DiGraph()
# 第一层连接
in_edges = list(flow_network.in_edges("Tgfb1", data=True))
in_edges = [(source, target, data['weight']) for source, target, data in in_edges]
network.add_weighted_edges_from(in_edges)
# 第二层连接
in_edges_nodes = [x[0] for x in in_edges]
for node in in_edges_nodes:
    in_edges = list(flow_network.in_edges(node, data=True))
    in_edges = [(source, target, data['weight']) for source, target, data in in_edges]
    network.add_weighted_edges_from(in_edges)
# 查看节点和边
print("Nodes:", network.nodes)
print("Edges with weights:")
for u, v, data in network.edges(data=True):
    print(f"{u} -> {v} with weight: {data['weight']}")
network.remove_node("GEM-6")
network.remove_node("GEM-12")



network = nx.DiGraph()
# 第一层连接
in_edges = list(flow_network.in_edges("Wnt7b", data=True))
in_edges = [(source, target, data['weight']) for source, target, data in in_edges]
network.add_weighted_edges_from(in_edges)
# 第二层连接
in_edges_nodes = [x[0] for x in in_edges]
for node in in_edges_nodes:
    in_edges = list(flow_network.in_edges(node, data=True))
    in_edges = [(source, target, data['weight']) for source, target, data in in_edges]
    network.add_weighted_edges_from(in_edges)
# 查看节点和边
print("Nodes:", network.nodes)
print("Edges with weights:")
for u, v, data in network.edges(data=True):
    print(f"{u} -> {v} with weight: {data['weight']}")
network.remove_node("GEM-6")
network.remove_node("GEM-12")
network.remove_node("GEM-20")




network = nx.DiGraph()
# 第一层连接
in_edges = list(flow_network.in_edges("Wnt5a", data=True))
in_edges = [(source, target, data['weight']) for source, target, data in in_edges]
network.add_weighted_edges_from(in_edges)
# 第二层连接
in_edges_nodes = [x[0] for x in in_edges]
for node in in_edges_nodes:
    in_edges = list(flow_network.in_edges(node, data=True))
    in_edges = [(source, target, data['weight']) for source, target, data in in_edges]
    network.add_weighted_edges_from(in_edges)
# 查看节点和边
print("Nodes:", network.nodes)
print("Edges with weights:")
for u, v, data in network.edges(data=True):
    print(f"{u} -> {v} with weight: {data['weight']}")
network.remove_node("GEM-1")




network = nx.DiGraph()
# 第一层连接
in_edges = list(flow_network.in_edges("Tgfb3", data=True))
in_edges = [(source, target, data['weight']) for source, target, data in in_edges]
network.add_weighted_edges_from(in_edges)
# 第二层连接
in_edges_nodes = [x[0] for x in in_edges]
for node in in_edges_nodes:
    in_edges = list(flow_network.in_edges(node, data=True))
    in_edges = [(source, target, data['weight']) for source, target, data in in_edges]
    network.add_weighted_edges_from(in_edges)
# 查看节点和边
print("Nodes:", network.nodes)
print("Edges with weights:")
for u, v, data in network.edges(data=True):
    print(f"{u} -> {v} with weight: {data['weight']}")
network.remove_node("GEM-12")


network = nx.DiGraph()
# 第一层连接
in_edges = list(flow_network.in_edges("Agt", data=True))
in_edges = [(source, target, data['weight']) for source, target, data in in_edges]
network.add_weighted_edges_from(in_edges)
# 第二层连接
in_edges_nodes = [x[0] for x in in_edges]
for node in in_edges_nodes:
    in_edges = list(flow_network.in_edges(node, data=True))
    in_edges = [(source, target, data['weight']) for source, target, data in in_edges]
    network.add_weighted_edges_from(in_edges)
# 查看节点和边
print("Nodes:", network.nodes)
print("Edges with weights:")
for u, v, data in network.edges(data=True):
    print(f"{u} -> {v} with weight: {data['weight']}")
network.remove_node("GEM-1")
network.remove_node("GEM-16")
network.remove_node("GEM-20")



network = nx.DiGraph()
# 第一层连接
in_edges = list(flow_network.in_edges("Pros1", data=True))
in_edges = [(source, target, data['weight']) for source, target, data in in_edges]
network.add_weighted_edges_from(in_edges)
# 第二层连接
in_edges_nodes = [x[0] for x in in_edges]
for node in in_edges_nodes:
    in_edges = list(flow_network.in_edges(node, data=True))
    in_edges = [(source, target, data['weight']) for source, target, data in in_edges]
    network.add_weighted_edges_from(in_edges)
# 查看节点和边
print("Nodes:", network.nodes)
print("Edges with weights:")
for u, v, data in network.edges(data=True):
    print(f"{u} -> {v} with weight: {data['weight']}")
