import networkx as nx
import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')



data_flie = './GraphDiffusion/MISAR/E18_5-S1/'
flow_network = nx.read_gexf(data_flie+"flowsig_network.gexf")
adata = sc.read_h5ad(data_flie+'adata_flow_network.h5ad')

def convert_data_format(G):
    all_nodes = list(G.nodes)
    node_indices = {n: i for i, n in enumerate(all_nodes)}
    sources = []
    targets = []
    values = []
    for u, v, d in G.edges(data=True):
        sources.append(node_indices[u])
        targets.append(node_indices[v])
        values.append(d['weight'])
    return all_nodes, sources, targets, values


import networkx as nx
from typing import Union, Sequence
import scanpy as sc
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors  # 导入 matplotlib.colors 模块

def plot_intercellular_sankey(adata: sc.AnnData,
                              flow_network: nx.DiGraph,
                              inflow_vars: Union[str, Sequence[str]] = None,
                              module_vars: Union[str, Sequence[str]] = None,
                              outflow_vars: Union[str, Sequence[str]] = None,
                              flowsig_network_key: str = 'flowsig_network'):
    flow_vars = list(flow_network.nodes())
    flow_var_info = adata.uns[flowsig_network_key]['flow_var_info']

    all_inflow_vars = [node for node in flow_vars if flow_var_info.loc[node]['Type'] == 'inflow']
    all_module_vars = [node for node in flow_vars if flow_var_info.loc[node]['Type'] == 'module']
    all_outflow_vars = [node for node in flow_vars if flow_var_info.loc[node]['Type'] == 'outflow']

    if inflow_vars is None:
        inflow_vars = all_inflow_vars
    if module_vars is None:
        module_vars = all_module_vars
    if outflow_vars is None:
        outflow_vars = all_outflow_vars

    selected_nodes = inflow_vars + module_vars + outflow_vars
    subgraph = flow_network.subgraph(selected_nodes)

    node_labels = list(subgraph.nodes())
    node_indices = {node: idx for idx, node in enumerate(node_labels)}

    sources = []
    targets = []
    values = []
    link_colors = ['#BBBBBB'] * len(list(subgraph.edges()))  # Set all link colors to a single color (e.g., gray)

    # Generate a color palette with enough distinct colors
    color_map = plt.cm.get_cmap("tab20c")  # Choose a colormap with enough distinct colors
    node_colors = {node: color_map(i / len(node_labels)) for i, node in enumerate(node_labels)}  # Generate unique colors

    # Convert RGBA to Hex colors using matplotlib.colors.rgb2hex
    node_colors = {node: mcolors.rgb2hex(node_colors[node][:3]) for node in node_colors}

    for source, target, data in subgraph.edges(data=True):
        sources.append(node_indices[source])
        targets.append(node_indices[target])
        values.append(data.get('weight', 1))

    sankey_figure = go.Figure(go.Sankey(
        node=dict(
            pad=80,  # Reduce spacing between nodes
            thickness=25,
            line=dict(color='black', width=1.0),
            label=node_labels,
            color=[node_colors[node] for node in node_labels]
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors  # Set the same color for all links
        )
    ))

    sankey_figure.update_layout(title_text=None, font_size=15)  # Increase font size
    sankey_figure.show(renderer='browser')


# convert format of networks for plotting
def convert_format_for_plotting(flow_network, file_name):
    all_nodes, sources, targets, values = convert_data_format(flow_network)
    node_names = all_nodes
    df = pd.DataFrame({'source': [node_names[i] for i in sources],
                       'target': [node_names[i] for i in targets],
                       'weight': values
                       })
    data = df

    inflow_gem = data[data.source.str.startswith('inflow')][['source', 'target']]

    gem_outflow = data[(data.source.str.startswith('GEM'))
                  & (~data.target.str.startswith('GEM'))][['source', 'target']]

    result = inflow_gem.merge(gem_outflow,
                              left_on='target',
                              right_on='source',
                              how='left')[['source_x', 'target_x', 'target_y']]

    result.columns = ['inflow', 'GEM', 'outflow']
    result.dropna().reset_index(drop=True)
    result.to_excel(data_flie+file_name+'-long.xlsx', index=None)




network1 = nx.DiGraph()
# 第一层连接
in_edges = list(flow_network.out_edges("inflow-Wnt6", data=True))
out_edges = [(source, target, data['weight']) for source, target, data in in_edges]
network1.add_weighted_edges_from(out_edges)
# 第二层连接
in_edges_nodes = [x[1] for x in in_edges]
for node in in_edges_nodes:
    out_edges = list(flow_network.out_edges(node, data=True))
    out_edges = [(source, target, data['weight']) for source, target, data in out_edges]
    network1.add_weighted_edges_from(out_edges)
network1.remove_node("GEM-20")
# 查看节点和边
print("Nodes:", network1.nodes)
print("Edges with weights:")
for u, v, data in network1.edges(data=True):
    print(f"{u} -> {v} with weight: {data['weight']}")
plot_intercellular_sankey(adata, network1)



network2 = nx.DiGraph()
# 第一层连接
in_edges = list(flow_network.in_edges("Wnt6", data=True))
in_edges = [(source, target, data['weight']) for source, target, data in in_edges]
network2.add_weighted_edges_from(in_edges)
# 第二层连接
in_edges_nodes = [x[0] for x in in_edges]
for node in in_edges_nodes:
    in_edges = list(flow_network.in_edges(node, data=True))
    in_edges = [(source, target, data['weight']) for source, target, data in in_edges]
    network2.add_weighted_edges_from(in_edges)
# 查看节点和边
print("Nodes:", network2.nodes)
print("Edges with weights:")
for u, v, data in network2.edges(data=True):
    print(f"{u} -> {v} with weight: {data['weight']}")
network2.remove_node("GEM-20")
plot_intercellular_sankey(adata, network2)

convert_format_for_plotting(flow_network, 'flow_network')
convert_format_for_plotting(network1, 'inflow-Wnt6_network')
convert_format_for_plotting(network2, 'Wnt6_network')









