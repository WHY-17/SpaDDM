import scanpy as sc
import flowsig as fs
import commot as ct
import networkx as nx
import anndata as ad
import networkx as nx
import scglue
from itertools import chain
import pandas as pd
import pickle
import numpy as np
import snapatac2 as snap
import polars as pl
from signal_flow_network_inference import *
from gene_peak_association_circular_bipartite_rotated import *
from inflow_cer_tf_outflow_multi_layers_construction import *







# load data
data_file = './MISAR/E18_5-S1/'
adata_results = sc.read_h5ad(data_file + 'E18_5-S1_spatialDDM_results_noscale.h5ad')

adata = sc.read_h5ad(data_file + 'adata_RNA.h5ad')
# Preprocessing the data
adata = preprocessing(adata)
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
Now perform spatial communication inference for these ligand-receptor pairs with a spatial distance limit.
CellChat database considers heteromeric units.
The signaling results are stored as spot-by-spot matrices in the obsp slots.
"""
adata = spatial_communication_inference_using_commot(adata, df_cellchat_filtered)
adata.write_h5ad(data_file+"adata_commot_dis25.h5ad", compression='gzip')

"""
Now, we perform global signal flow inference using FlowSig
"""
commot_output_key = 'commot-cellchat'
# We first construct the potential cellular flows from the COMMOT output, which has been run previously.
adata.obsm['SpatialDDM_pca'] = adata_results.obsm['SpatialDDM_pca']
adata, flow_network = global_signal_flow_using_flowsig(adata, commot_output_key)
rename_map = {n: rename_node(n) for n in flow_network.nodes if rename_node(n) != n}
flow_network = nx.relabel_nodes(flow_network, rename_map)
flow_var_info = adata.uns['flowsig_network']['flow_var_info']
flow_var_info.index = [rename_node(n) for n in flow_var_info.index]
flow_vars = adata.uns['flowsig_network']['network']['flow_vars']
adata.uns['flowsig_network']['network']['flow_vars'] = np.array([rename_node(n) for n in flow_vars])


nx.write_gexf(flow_network, data_file + "flowsig_network.gexf")
adata.write_h5ad(data_file + "adata_flow_network.h5ad", compression='gzip')

"""
construct gene peak association using scglue
"""
rna = ad.read_h5ad("adata_RNA.h5ad")
rna.var_names_make_unique()
atac = ad.read_h5ad("adata_Peak.h5ad")
atac.var_names_make_unique()

rna, atac, guidance = preprocessing_for_scglue_and_gene_annotation(rna, atac)
rna, atac, guidance_hvf = scglue_model_fit(rna, atac, guidance)

gene2peak = gene_peak_association(rna, atac, guidance_hvf)

rename_map = {n: rename_node(n) for n in flow_network.nodes if rename_node(n) != n}
flow_network = nx.relabel_nodes(flow_network, rename_map)

outflows = nodes = set(
    n
    for n in flow_network.nodes
    if not (n.startswith("CER-") or n.startswith("r-"))
)

outflow_peaks_dict = get_top_peaks_for_outflows(gene2peak, outflows)

plot_gene_peak_circular_bipartite_rotated(
    gene2peak=gene2peak,
    genes=outflows,
    topk=5,
    figsize=(12, 12),
    title="Gene-Peak Circular Network",
    plot_legend=True,
    peak_label_angle=10,
    save_fig=True,
    file_name=data_file+'gene_peak_circular_network.pdf'
)

# motifs enrichment analysis on these peaks using snapatac2
outflows_motifs = snap.tl.motif_enrichment(
    motifs=snap.datasets.cis_bp(unique=True),
    regions=outflow_peaks_dict,
    genome_fasta=snap.genome.mm10
)

# select top significant TFs
outflows_top_TFs = select_significant_top_TFs(outflows_motifs, K=60, padj_cutoff=0.05)

# construct initial network from FlowSig results for specific nodes
network_inflow_Edn1 = specific_inflow_subgraph_extract(flow_network, 'r-Edn1')
network_outflow_Edn1 = specific_outflow_subgraph_extract(flow_network, 'Edn1')

node_name = [n for n in flow_network.nodes if n.startswith("r-")]
network_global = specific_inflow_subgraph_extract(flow_network, node_name=node_name)
convert_format_for_plotting(network_global, 'global_flowsig_inflow_cer_outflow_network')

with open(data_file+"network_inflow_Edn1.pkl", "wb") as f:
    pickle.dump(network_inflow_Edn1, f)

with open(data_file+"network_outflow_Edn1.pkl", "wb") as f:
    pickle.dump(network_outflow_Edn1, f)

with open("contributions_to_target_CERs_top100.pkl", "rb") as f:
    CER_top_genes = pickle.load(f)

# select some CERs having high weight to inflow node
CERs_inflow = [cer for cer in network_inflow_Edn1.nodes if cer.startswith("CER-")]
# construct network from inflow--CERs--TFs
inflow_G = build_CER_TF_outflow_network(outflows_top_TFs, CER_top_genes, CERs_inflow)
# 查看边信息
print(inflow_G.edges(data=True))

# Merge initial network from FlowSig results and inflow_G, to construct multi-layers network inflow--CERs--TFs--outflow
G_merged, TFs, outflow_nodes = network_merge(inflow_G, network_inflow_Edn1)
build_inflow_cer_tf_outflow_table(G_merged, TFs, outflow_nodes, 'inflow-Edn1-inflow_cer_tf_outflow_four_layers_network')



CERs_outflow = [cer for cer in network_outflow_Edn1.nodes if cer.startswith("CER-")]
top_TFs = {'Edn1': outflows_top_TFs['Edn1']}
outflow_G = build_CER_TF_outflow_network(top_TFs, CER_top_genes, CERs_outflow)
# 查看边信息
print(outflow_G.edges(data=True))

G_merged, TFs, outflow_nodes = network_merge(outflow_G, network_outflow_Edn1)
build_inflow_cer_tf_outflow_table(G_merged, TFs, outflow_nodes, 'Edn1-inflow_cer_tf_outflow_four_layers_network')
