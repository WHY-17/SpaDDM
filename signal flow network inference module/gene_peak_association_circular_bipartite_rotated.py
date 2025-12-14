import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import pandas as pd



def get_top_peaks_for_outflows(gene2peak, outflows, top_k=5, score_key="score"):
    """
    从 gene2peak 图中为每个 outflow 基因提取与其相连的 top-k peaks，
    按照 edge attribute 中的 score 排序。

    参数
    ----
    gene2peak : networkx.Graph 或 networkx.DiGraph
        代表 gene-peak regulatory links 的图，边上有 score 属性。
    outflows : list
        需要提取的 outflow 基因列表。
    top_k : int, 默认 5
        每个 outflow 选取的 top peaks 数量。
    score_key : str, 默认 "score"
        用于排序的边属性键名。

    返回
    ----
    dict
        {outflow_gene: [peak1, peak2, ...]} 的字典
    """

    outflow_peaks_dict = {}

    for outflow in outflows:
        if outflow not in gene2peak:
            continue

        # 所有相关边：(u, v, attr)
        edges = gene2peak.edges(outflow, data=True)

        # 选 top_k highest score
        edges_top = sorted(
            edges,
            key=lambda x: x[2].get(score_key, 0),
            reverse=True
        )[:top_k]

        # 提取 outflow 相连的峰（目标节点）
        top_peaks = [
            v if u == outflow else u
            for u, v, _ in edges_top
        ]
        top_peaks = pd.Index(sorted(set(top_peaks)))
        outflow_peaks_dict[outflow] = top_peaks
    return outflow_peaks_dict


def plot_gene_peak_circular_bipartite_rotated(
        gene2peak,
        genes,
        topk=5,
        figsize=(12, 12),
        title="Gene-Peak Circular Network",
        plot_legend=True,
        peak_label_angle=20,  # 外圈标签旋转角度
        save_fig=True,
        file_name=None
):
    """
    双层环形图：基因在内圈，peak在外圈，peak标签旋转显示。

    Parameters
    ----------
    gene2peak : nx.DiGraph or nx.Graph
        原始基因-peak网络
    genes : list
        基因列表
    topk : int
        每个基因选取 topk 条边
    figsize : tuple
        图尺寸
    title : str
        图标题
    peak_label_angle : float
        外圈 peak 标签旋转角度（度数）
    """
    # 1️⃣ 提取 topk edges
    edges_sub = []
    peaks_set = set()
    for gene in genes:
        edges = [(u, v, d) for u, v, d in gene2peak.edges(data=True) if u == gene]
        edges_top = sorted(edges, key=lambda x: x[2].get('score', 0), reverse=True)[:topk]
        edges_sub.extend(edges_top)
        peaks_set.update([v for _, v, _ in edges_top])

    # 2️⃣ 构建无向子图
    G_sub = nx.Graph()
    for u, v, d in edges_sub:
        G_sub.add_edge(u, v, **d)

    # 3️⃣ 区分节点
    gene_nodes = [n for n in G_sub.nodes if n in genes]
    peak_nodes = [n for n in G_sub.nodes if n not in genes]

    # 4️⃣ 双层环形布局
    pos = {}
    # 内圈：genes
    n_genes = len(gene_nodes)
    for i, n in enumerate(gene_nodes):
        angle = 2 * np.pi * i / n_genes
        pos[n] = (0.3 * np.cos(angle), 0.3 * np.sin(angle))  # 内圈半径 0.3
    # 外圈：peaks
    n_peaks = len(peak_nodes)
    for i, n in enumerate(peak_nodes):
        angle = 2 * np.pi * i / n_peaks
        pos[n] = (np.cos(angle), np.sin(angle))  # 外圈半径 1

    # 5️⃣ 绘图
    plt.figure(figsize=figsize)

    # 节点
    nx.draw_networkx_nodes(G_sub, pos, nodelist=gene_nodes, node_color="#FF7F50", node_size=300)
    nx.draw_networkx_nodes(G_sub, pos, nodelist=peak_nodes, node_color="#87CEFA", node_size=200)

    # 边
    scores = [d.get('score', 0) for _, _, d in G_sub.edges(data=True)]
    nx.draw_networkx_edges(
        G_sub, pos,
        width=[0.5 + s * 2 for s in scores],
        edge_color=scores,
        edge_cmap=plt.cm.Reds
    )

    # 标签
    # 先绘制 gene 标签（不旋转）
    nx.draw_networkx_labels(G_sub, pos, labels={n: n for n in gene_nodes}, font_size=8)

    # 绘制 peak 标签，旋转 peak_label_angle 度
    for n in peak_nodes:
        x, y = pos[n]
        plt.text(
            x, y, n,
            fontsize=8,
            rotation=peak_label_angle,
            ha='center', va='center'
        )

    # 图例
    legend_elements = [
        Patch(facecolor="#FF7F50", label='Genes'),
        Patch(facecolor="#87CEFA", label='Peaks')
    ]
    if plot_legend:
        plt.legend(handles=legend_elements)

    plt.title(title, fontsize=10)
    plt.axis('off')
    if save_fig:
        plt.savefig(file_name, dpi=300)
    plt.show()

def rename_node(name):
    if name.startswith("GEM-"):
        return name.replace("GEM-", "CER-")
    if name.startswith("inflow-"):
        return name.replace("inflow-", "r-")
    return name


data_file = './GraphDiffusion/Mouse Brain/RNA-ATAC/'
gene2peak = nx.read_graphml(data_file+"mousebrain_gene2peak_network.graphml")
flow_network = nx.read_gexf(data_file+"flowsig_network.gexf")
rename_map = {n: rename_node(n) for n in flow_network.nodes if rename_node(n) != n}
flow_network = nx.relabel_nodes(flow_network, rename_map)

"""
outflows = nodes = set(
    n
    for G in [network_inflow, network_outflow]
    for n in G.nodes
    if not (n.startswith("CER-") or n.startswith("r-"))
)
"""
outflows = nodes = set(
    n
    for n in flow_network.nodes
    if not (n.startswith("CER-") or n.startswith("r-"))
)

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


