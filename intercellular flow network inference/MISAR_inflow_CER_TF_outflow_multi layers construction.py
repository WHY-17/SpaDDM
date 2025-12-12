import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
from typing import Union, Sequence
import scanpy as sc
import plotly.graph_objects as go
import matplotlib.colors as mcolors  # 导入 matplotlib.colors 模块


def rename_node(name):
    if name.startswith("GEM-"):
        return name.replace("GEM-", "CER-")
    if name.startswith("inflow-"):
        return name.replace("inflow-", "r-")
    return name


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


# convert format of networks for plotting
def convert_format_for_plotting(flow_network, file_name):
    all_nodes, sources, targets, values = convert_data_format(flow_network)
    node_names = all_nodes
    df = pd.DataFrame({'source': [node_names[i] for i in sources],
                       'target': [node_names[i] for i in targets],
                       'weight': values
                       })
    data = df

    inflow_gem = data[data.source.str.startswith('r-')][['source', 'target']]

    gem_outflow = data[(data.source.str.startswith('CER'))
                  & (~data.target.str.startswith('CER'))][['source', 'target']]

    result = inflow_gem.merge(gem_outflow,
                              left_on='target',
                              right_on='source',
                              how='left')[['source_x', 'target_x', 'target_y']]

    result.columns = ['inflow', 'CER', 'outflow']
    result.dropna().reset_index(drop=True)
    result.to_excel(data_file+file_name+'-long.xlsx', index=None)



def plot_inflow_signaling_network(G, figsize=(12, 8)):
    """
    绘制三层有向网络图，并使用边的权重控制线条粗细。
    """
    # ---- 1. 分类节点 ----
    inflow_nodes = [n for n in G.nodes if n.startswith('r-')]
    CER_nodes = [n for n in G.nodes if n.startswith('CER-')]
    outflow_nodes = [n for n in G.nodes if n not in inflow_nodes + CER_nodes]

    # ---- 2. 节点颜色 ----
    node_color = []
    for n in G.nodes:
        if n in inflow_nodes:
            node_color.append('orange')
        elif n in CER_nodes:
            node_color.append('skyblue')
        else:
            node_color.append('lightgreen')

    # ---- 3. 根据权重计算线宽 ----
    edge_widths = []
    for u, v, data in G.edges(data=True):
        w = data.get("weight", 1.0)
        edge_widths.append(w * 4)  # 可调，5 表示放大权重

    # ---- 4. 三层布局 ----
    shells = [inflow_nodes, CER_nodes, outflow_nodes]
    pos = nx.shell_layout(G, nlist=shells)

    # ---- 5. 绘图 ----
    plt.figure(figsize=figsize)
    nx.draw(
        G, pos,
        with_labels=True,
        node_color=node_color,
        node_size=800,
        width=edge_widths,          # ← 使用权重映射的线宽
        edge_color='gray',
        arrowsize=20,
        font_size=10
    )
    plt.title("Three-layer Network (edge width reflects weight)")
    plt.show()
    return pos



def plot_outflow_signaling_network(G, figsize=(12, 8)):
    """
    绘制三层有向网络图，并使用边的权重控制线条粗细。
    """
    # ---- 1. 分类节点 ----
    inflow_nodes = [n for n in G.nodes if n.startswith('r-')]
    CER_nodes = [n for n in G.nodes if n.startswith('CER-')]
    outflow_nodes = [n for n in G.nodes if n not in inflow_nodes + CER_nodes]

    # ---- 2. 节点颜色 ----
    node_color = []
    for n in G.nodes:
        if n in inflow_nodes:
            node_color.append('orange')
        elif n in CER_nodes:
            node_color.append('skyblue')
        else:
            node_color.append('lightgreen')

    # ---- 3. 根据权重计算线宽 ----
    edge_widths = []
    for u, v, data in G.edges(data=True):
        w = data.get("weight", 1.0)
        edge_widths.append(w * 4)  # 可调，5 表示放大权重

    # ---- 4. 三层布局 ----
    shells = [outflow_nodes, CER_nodes, inflow_nodes]
    pos = nx.shell_layout(G, nlist=shells)

    # ---- 5. 绘图 ----
    plt.figure(figsize=figsize)
    nx.draw(
        G, pos,
        with_labels=True,
        node_color=node_color,
        node_size=800,
        width=edge_widths,          # ← 使用权重映射的线宽
        edge_color='gray',
        arrowsize=20,
        font_size=10
    )
    plt.title("Three-layer Network (edge width reflects weight)")
    plt.show()
    return pos



def specific_inflow_subgraph_extract(flow_network, node_name, global_plot=True):

    # -------------------------------
    # 1. 子图构建
    # -------------------------------
    network = nx.DiGraph()

    # 第一层：node_name → 下游
    first_edges = list(flow_network.out_edges(node_name, data=True))
    network.add_weighted_edges_from([(u, v, d['weight']) for u, v, d in first_edges])

    # 第二层：这些下游节点 → 其下游
    first_layer_nodes = [v for _, v, _ in first_edges]

    for n in first_layer_nodes:
        edges = list(flow_network.out_edges(n, data=True))
        network.add_weighted_edges_from([(u, v, d['weight']) for u, v, d in edges])

    # 全局图（未删节点）可选择先画一遍
    if global_plot:
        plot_inflow_signaling_network(network)

    # -------------------------------
    # 2. 删除低权重边
    # -------------------------------
    """
    edges_to_remove = [
        (u, v) for u, v, data in network.edges(data=True)
        if data["weight"] < 0.5
    ]
    network.remove_edges_from(edges_to_remove)
    """

    # -------------------------------
    # 3. 删除 out-degree=0 或 in-degree=0 的 GEM 节点
    # -------------------------------
    CER_nodes_to_remove_1 = [
        n for n in network.nodes
        if n.startswith("CER-") and network.out_degree(n) == 0
    ]
    print("Removing GEM nodes:", CER_nodes_to_remove_1)
    network.remove_nodes_from(CER_nodes_to_remove_1)

    # -------------------------------
    # 4. 重新分类节点
    # -------------------------------
    inflow_nodes = [n for n in network.nodes if n.startswith("r-")]
    CER_nodes = [n for n in network.nodes if n.startswith("CER-")]
    outflow_nodes = [n for n in network.nodes if n not in inflow_nodes + CER_nodes]

    # 删除与outflow没有连接的GEM
    CER_nodes_to_remove_2 = [
        cer_node for cer_node in CER_nodes
        if not any(network.has_edge(cer_node, outflow_node) for outflow_node in outflow_nodes)
    ]
    print("Removing CER nodes:", CER_nodes_to_remove_2)
    network.remove_nodes_from(CER_nodes_to_remove_2)

    # -------------------------------
    # 5. 删除 in-degree=0 的 outflow 节点
    # -------------------------------
    outflow_nodes_to_remove = [
        n for n in outflow_nodes
        if network.in_degree(n) == 0
    ]
    print("Removing outflow nodes:", outflow_nodes_to_remove)
    network.remove_nodes_from(outflow_nodes_to_remove)

    # -------------------------------
    # 6. 最终绘图
    # -------------------------------
    plot_inflow_signaling_network(network)

    return network


def specific_outflow_subgraph_extract(flow_network, node_name, global_plot=True):

    # -------------------------------
    # 1. 子图构建（反向：追溯来源）
    # -------------------------------
    network = nx.DiGraph()

    # 第一层：上游节点 → node_name
    first_edges = list(flow_network.in_edges(node_name, data=True))
    network.add_weighted_edges_from([(u, v, d['weight']) for u, v, d in first_edges])

    # 第一层节点（它们指向 node_name）
    first_layer_nodes = [u for u, _, _ in first_edges]

    # 第二层：更上游的节点 → 第一层节点
    for n in first_layer_nodes:
        edges = list(flow_network.in_edges(n, data=True))
        network.add_weighted_edges_from([(u, v, d['weight']) for u, v, d in edges])

    # 可选：先画原始子图
    if global_plot:
        plot_outflow_signaling_network(network)

    # -------------------------------
    # 2. 删除低权重边
    # -------------------------------
    """
    edges_to_remove = [(u, v) for u, v, data in network.edges(data=True)
                       if data["weight"] < 0.5]
    network.remove_edges_from(edges_to_remove)
    """

    # -------------------------------
    # 3. 删除与node_name没有连接的 GEM 节点（无下游）
    # -------------------------------
    CER_nodes_to_remove = [
        n for n in network.nodes
        if n.startswith("CER-") and (n, node_name) not in network.edges
    ]
    print("Removing CER nodes:", CER_nodes_to_remove)
    network.remove_nodes_from(CER_nodes_to_remove)

    # -------------------------------
    # 4. 分类节点（基于网络）
    # -------------------------------
    inflow_nodes = [n for n in network.nodes if n.startswith("r-")]
    CER_nodes = [n for n in network.nodes if n.startswith("CER-")]
    outflow_nodes = [n for n in network.nodes if n not in inflow_nodes + CER_nodes]

    # -------------------------------
    # 5. 删除 out-degree=0 的 inflow 节点（无上游）
    # -------------------------------
    inflow_nodes_to_remove = [n for n in inflow_nodes if network.out_degree(n) == 0]
    print("Removing inflow nodes:", inflow_nodes_to_remove)
    network.remove_nodes_from(inflow_nodes_to_remove)

    # -------------------------------
    # 6. 删除与 inflow 没有连接的 CER 节点
    # -------------------------------
    CER_nodes_to_remove_ = [
        n for n in network.nodes
        if n.startswith("CER-") and not any((inflow, n) in network.edges for inflow in inflow_nodes)
    ]
    print("Removing CER nodes:", CER_nodes_to_remove_)
    network.remove_nodes_from(CER_nodes_to_remove_)

    # -------------------------------
    # 7. 最终绘图
    # -------------------------------
    plot_outflow_signaling_network(network)
    return network


def network_merge(G_1, G_2):
    # 找出需要删除的边
    edges_to_remove = [(u, v) for u, v in G_2.edges() if
                       u.startswith("CER-") and not v.startswith("CER-")]
    # 删除这些边
    G_2.remove_edges_from(edges_to_remove)

    # 找出所有TF节点：不是 "CER" 开头，也不是 "inflow" 开头
    tf_nodes = [x for x in G_1.nodes if x not in G_2.nodes]

    # 找出所有outflow节点
    outflow_nodes = [n for n in G_2.nodes() if not n.startswith("r-") if not n.startswith("CER-")]

    G_merged = nx.compose(G_1, G_2)

    return G_merged, tf_nodes, outflow_nodes



def build_inflow_cer_tf_outflow_table(G, tf_nodes, outflow_nodes, file_name):
    # 1. 构建边表
    all_nodes, sources, targets, values = convert_data_format(G)
    df = pd.DataFrame({
        'source': [all_nodes[i] for i in sources],
        'target': [all_nodes[i] for i in targets],
        'weight': values
    })
    # 2. inflow → CER
    inflow_cer = df[
        df.source.str.startswith('r-') &
        df.target.str.startswith('CER')
    ][['source', 'target']]
    # 3. CER → TF
    cer_tf = df[
        df.source.str.startswith('CER') &
        df.target.isin(tf_nodes)
    ][['source', 'target']]
    # 4. TF → outflow
    tf_outflow = df[
        df.source.isin(tf_nodes) &
        df.target.isin(outflow_nodes)
    ][['source', 'target']]
    # 5. 合并 inflow → CER → TF
    step1 = inflow_cer.merge(
        cer_tf,
        left_on='target',    # CER
        right_on='source',   # CER
        how='left'
    )[['source_x', 'target_x', 'target_y']]
    step1.columns = ['inflow', 'CER', 'TF']
    # 6. 合并 CER → TF → outflow
    step2 = step1.merge(
        tf_outflow,
        left_on='TF',
        right_on='source',
        how='left'
    )[['inflow', 'CER', 'TF', 'target']]
    step2.columns = ['inflow', 'CER', 'TF', 'outflow']
    # 去掉缺失值
    step2 = step2.dropna().reset_index(drop=True)
    # 保存 Excel
    step2.to_excel(data_file + file_name + '-long.xlsx', index=None)




data_file = './GraphDiffusion/MISAR/E18_5-S1/'
flow_network = nx.read_gexf(data_file+"flowsig_network.gexf")
adata = sc.read_h5ad(data_file+'adata_flow_network.h5ad')

rename_map = {n: rename_node(n) for n in flow_network.nodes if rename_node(n) != n}
flow_network = nx.relabel_nodes(flow_network, rename_map)
flow_var_info = adata.uns['flowsig_network']['flow_var_info']
flow_var_info.index = [rename_node(n) for n in flow_var_info.index]
flow_vars = adata.uns['flowsig_network']['network']['flow_vars']
adata.uns['flowsig_network']['network']['flow_vars'] = np.array([rename_node(n) for n in flow_vars])


network_inflow_Edn1 = specific_inflow_subgraph_extract(flow_network, 'r-Edn1')
network_outflow_Edn1 = specific_outflow_subgraph_extract(flow_network, 'Edn1')


node_name = [n for n in flow_network.nodes if n.startswith("r-")]
network_global = specific_inflow_subgraph_extract(flow_network, node_name=node_name)
convert_format_for_plotting(network_global, 'global_flowsig_inflow_cer_outflow_network')

import pickle
with open(data_file+"network_inflow_Edn1.pkl", "wb") as f:
    pickle.dump(network_inflow_Edn1, f)

with open(data_file+"network_outflow_Edn1.pkl", "wb") as f:
    pickle.dump(network_outflow_Edn1, f)


with open(data_file + "inflow-Edn1_CER_TF_outflow_network.pkl", "rb") as f:
    G_1 = pickle.load(f)
G_2 = network_inflow_Edn1
G_merged, TFs, outflow_nodes = network_merge(G_1, G_2)
build_inflow_cer_tf_outflow_table(G_merged, TFs, outflow_nodes, 'inflow-Edn1-inflow_cer_tf_outflow_four_layers_network')



with open(data_file + "Edn1_CER_TF_outflow_network.pkl", "rb") as f:
    G_1 = pickle.load(f)
G_2 = network_outflow_Edn1
G_merged, TFs, outflow_nodes = network_merge(G_1, G_2)
build_inflow_cer_tf_outflow_table(G_merged, TFs, outflow_nodes, 'Edn1-inflow_cer_tf_outflow_four_layers_network')
