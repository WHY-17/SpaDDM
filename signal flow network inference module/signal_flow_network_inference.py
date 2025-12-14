import scanpy as sc
import flowsig as fs
import commot as ct
import networkx as nx
import anndata as ad
import networkx as nx
import scglue
from itertools import chain
import pandas as pd
import numpy as np
import snapatac2 as snap
import polars as pl
from gene_peak_association_circular_bipartite_rotated import *



def preprocessing(adata):
    adata.var_names_make_unique()
    adata.raw = adata
    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.filter_cells(adata, min_genes=10)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata = adata[:, adata.var.highly_variable]
    return adata


def spatial_communication_inference_using_commot(adata, df_ligrec):
    ct.tl.spatial_communication(adata, database_name='cellchat',
                                df_ligrec=df_ligrec, dis_thr=25,
                                heteromeric=True, pathway_sum=True)
    return adata

def global_signal_flow_using_flowsig(adata, commot_output_key):
    fs.pp.construct_flows_from_commot(adata,
                                      commot_output_key=commot_output_key,
                                      gem_expr_key='SpatialDDM_pca',
                                      scale_gem_expr=False,
                                      flowsig_network_key='flowsig_network',
                                      flowsig_expr_key='X_flow')

    fs.pp.construct_spatial_blocks(adata,
                                   n_blocks=20,
                                   use_graph=False,
                                   spatial_block_key="spatial_block",
                                   spatial_key="spatial")

    # Now we are ready to learn the network
    fs.tl.learn_intercellular_flows(adata,
                                    flowsig_key='flowsig_network',
                                    flow_expr_key='X_flow',
                                    use_spatial=True,
                                    block_key='spatial_block',
                                    n_jobs=6,
                                    n_bootstraps=10)

    """
    Now we do post-learning validation to reorient undirected edges from the learnt CPDAG
    so that they flow from inflow to GEM to outflow. After that, we remove low-confidence edges.
    """
    # This part is key for reducing false positives
    fs.tl.apply_biological_flow(adata,
                                flowsig_network_key='flowsig_network',
                                adjacency_key='adjacency',
                                validated_key='adjacency_validated')

    edge_threshold = 0.7
    fs.tl.filter_low_confidence_edges(adata,
                                      edge_threshold=edge_threshold,
                                      flowsig_network_key='flowsig_network',
                                      adjacency_key='adjacency',
                                      filtered_key='adjacency_filtered')

    flow_network = fs.tl.construct_intercellular_flow_network(adata, adjacency_key='adjacency')

    return adata, flow_network



def preprocessing_for_scglue_and_gene_annotation(rna, atac):
    rna.var.index.name = 'genes'
    del rna.var['gene_ids']
    del rna.var['feature_types']
    del rna.var['genome']

    new_varnames = []
    for p in atac.var_names:
        parts = p.split('.')
        # chromosome 名称可能包含一个或多个小数点（如 GL456216.1）
        chrom = ".".join(parts[:-2])
        start = parts[-2]
        end = parts[-1]
        new_varnames.append(f"{chrom}:{start}-{end}")
    atac.var_names = new_varnames

    rna.layers["counts"] = rna.X.copy()
    sc.pp.filter_genes(rna, min_cells=10)
    sc.pp.highly_variable_genes(rna, n_top_genes=3000, flavor="seurat_v3")
    sc.pp.normalize_total(rna)
    sc.pp.log1p(rna)
    sc.pp.scale(rna)
    sc.tl.pca(rna, n_comps=100, svd_solver="auto")
    sc.pp.neighbors(rna, metric="cosine")
    sc.tl.umap(rna)
    sc.pl.umap(rna, color="Combined_Clusters_annotation")

    scglue.data.lsi(atac, n_components=100, n_iter=15)
    sc.pp.neighbors(atac, use_rep="X_lsi", metric="cosine")
    sc.tl.umap(atac)
    sc.pl.umap(atac, color="Combined_Clusters_annotation")

    scglue.data.get_gene_annotation(
        rna, gtf="gencode.vM25.chr_patch_hapl_scaff.annotation.gtf.gz",
        gtf_by="gene_name"
    )
    rna.var.loc[:, ["chrom", "chromStart", "chromEnd"]].head()
    rna = rna[:, rna.var["mgi_id"].notna()].copy()

    split = atac.var_names.str.split(r"[:-]")
    atac.var["chrom"] = split.map(lambda x: x[0])
    atac.var["chromStart"] = split.map(lambda x: x[1]).astype(int)
    atac.var["chromEnd"] = split.map(lambda x: x[2]).astype(int)

    guidance = scglue.genomics.rna_anchored_guidance_graph(rna, atac)
    scglue.graph.check_graph(guidance, [rna, atac])

    rna.write("MISAR-rna-pp.h5ad", compression="gzip")
    atac.write("MISAR-atac-pp.h5ad", compression="gzip")
    nx.write_graphml(guidance, "MISAR-guidance.graphml.gz")
    return rna, atac, guidance


def scglue_model_fit(rna, atac, guidance):
    scglue.models.configure_dataset(
        rna, "NB", use_highly_variable=True,
        use_layer="counts", use_rep="X_pca"
    )
    scglue.models.configure_dataset(
        atac, "NB", use_highly_variable=True,
        use_rep="X_lsi"
    )
    guidance_hvf = guidance.subgraph(chain(
        rna.var.query("highly_variable").index,
        atac.var.query("highly_variable").index
    )).copy()
    glue = scglue.models.fit_SCGLUE(
        {"rna": rna, "atac": atac}, guidance_hvf,
        fit_kws={"directory": "glue"}
    )
    glue.save("MISAR-glue.dill")

    dx = scglue.models.integration_consistency(
        glue, {"rna": rna, "atac": atac}, guidance_hvf
    )

    rna.obsm["X_glue"] = glue.encode_data("rna", rna)
    atac.obsm["X_glue"] = glue.encode_data("atac", atac)

    combined = ad.concat([rna, atac])
    sc.pp.neighbors(combined, use_rep="X_glue", metric="cosine")
    sc.tl.umap(combined)
    sc.pl.umap(combined, color=["Combined_Clusters_annotation"], wspace=0.65)

    feature_embeddings = glue.encode_graph(guidance_hvf)
    feature_embeddings = pd.DataFrame(feature_embeddings, index=glue.vertices)

    rna.varm["X_glue"] = feature_embeddings.reindex(rna.var_names).to_numpy()
    atac.varm["X_glue"] = feature_embeddings.reindex(atac.var_names).to_numpy()

    rna.write("MISAR-rna-emb.h5ad", compression="gzip")
    atac.write("MISAR-atac-emb.h5ad", compression="gzip")
    nx.write_graphml(guidance_hvf, "MISAR-guidance-hvf.graphml.gz")
    return  rna, atac, guidance_hvf


def gene_peak_association(rna, atac, guidance_hvf):
    rna.var["name"] = rna.var_names
    atac.var["name"] = atac.var_names

    genes = rna.var.query("highly_variable").index
    peaks = atac.var.query("highly_variable").index

    features = pd.Index(np.concatenate([rna.var_names, atac.var_names]))
    feature_embeddings = np.concatenate([rna.varm["X_glue"], atac.varm["X_glue"]])

    skeleton = guidance_hvf.edge_subgraph(
        e for e, attr in dict(guidance_hvf.edges).items()
        if attr["type"] == "fwd"
    ).copy()

    reginf = scglue.genomics.regulatory_inference(
        features, feature_embeddings,
        skeleton=skeleton, random_state=0
    )

    gene2peak = reginf.edge_subgraph(
        e for e, attr in dict(reginf.edges).items()
        if attr["qval"] < 0.05
    )

    scglue.genomics.Bed(atac.var).write_bed("peaks.bed", ncols=3)
    scglue.genomics.write_links(
        gene2peak,
        scglue.genomics.Bed(rna.var).strand_specific_start_site(),
        scglue.genomics.Bed(atac.var),
        "gene2peak.links", keep_attrs=["score"]
    )

    nx.write_graphml(gene2peak, "MISAR_E18.5-S1_gene_peak_network.graphml")
    return gene2peak


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


def select_significant_top_TFs(motifs_results, K, padj_cutoff):
    result = {}
    for cluster, df in motifs_results.items():
        # 1. 筛选显著 TF
        df_sig = df.filter(pl.col("adjusted p-value") <= padj_cutoff)

        # 2. 如果一个簇显著 TF 不足 K 个，就自动保留所有现有的
        df_top = (df_sig.sort("log2(fold change)", descending=True).head(K))

        # 保存结果
        result[cluster] = df_top

    for key, df in result.items():
        result[key] = (
            df
            .with_columns(
                # 先全部小写
                pl.col("name").str.to_lowercase().alias("name")
            )
            .with_columns(
                # 大写首字母 + 保留其余
                (pl.col("name").str.slice(0, 1).str.to_uppercase()
                 + pl.col("name").str.slice(1, None)).alias("name")
            )
        )
    return result

def build_CER_TF_outflow_network(outflows_top_TFs, CER_top_genes, selected_CERs):
    """
    构建 TF -> CER 的有向图，边权重为 TF 对 CER 的绝对贡献。

    Parameters
    ----------
    outflows_top_TFs : dict
        每个键是 outflow 名称，对应值是包含 TF 的 DataFrame（Polars 或 Pandas），列名为 "name"。
    CER_top_genes : dict
        每个键是 CER 名称，对应值是包含基因贡献信息的 DataFrame，列名至少包括 "gene" 和 "abs_importance"。
    selected_CERs : list of str
        需要考虑的 CER 键名。

    Returns
    -------
    G : nx.DiGraph
        构建好的有向图，节点是 TF 和 CER，边的权重为 abs_importance。
    """
    G = nx.DiGraph()

    for outflow, tf_df in outflows_top_TFs.items():
        tf_df = tf_df.to_pandas()
        # Polars → Python list
        tfs = tf_df["name"].to_list()
        for tf in tfs:
            for cer in selected_CERs:
                if cer not in CER_top_genes:
                    continue
                cer_df = CER_top_genes[cer]

                # Polars → Pandas
                if hasattr(cer_df, "to_pandas"):
                    cer_pd = cer_df.to_pandas()
                else:
                    cer_pd = cer_df

                # 查找 TF 是否在 CER 中
                row = cer_pd[cer_pd["gene"] == tf]
                if row.empty:
                    continue
                importance = float(row["abs_importance"].iloc[0])
                tf_log_fold_change = float(tf_df[tf_df['name']==tf]['log2(fold change)'])
                # 添加边
                G.add_edge(cer, tf, weight=importance)
                G.add_edge(tf, outflow, weight=tf_log_fold_change)

    return G





