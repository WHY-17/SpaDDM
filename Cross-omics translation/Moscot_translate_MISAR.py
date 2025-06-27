import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

import moscot.plotting as mtp
from moscot import datasets
from moscot.problems.cross_modality import TranslationProblem
import numpy as np
import pandas as pd
import scipy
from sklearn import preprocessing as pp
import anndata as ad
import scanpy as sc



def foscttm(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    d = scipy.spatial.distance_matrix(x, y)
    foscttm_x = (d < np.expand_dims(np.diag(d), axis=1)).mean(axis=1)
    foscttm_y = (d < np.expand_dims(np.diag(d), axis=0)).mean(axis=0)
    fracs = []
    for i in range(len(foscttm_x)):
        fracs.append((foscttm_x[i] + foscttm_y[i]) / 2)
    return np.mean(fracs).round(4)


adata_atac = sc.read_h5ad('Graph_Diffusion/MISAR/E18_5-S1/adata_Peak.h5ad')
adata_rna = sc.read_h5ad('Graph_Diffusion/MISAR/E18_5-S1/adata_RNA.h5ad')
adata_atac.var_names_make_unique()
adata_rna.var_names_make_unique()

# adata_atac.obsm["ATAC_lsi_l2_norm"] = pp.normalize(adata_atac.obsm["X_lsi"], norm="l2")
adata_results = sc.read_h5ad('Graph_Diffusion/MISAR/E18_5-S1/spatialDDM_results.h5ad')
adata_results.obs['SpatialDDM'] = adata_results.obs['SpatialDDM'].astype(str)
adata_rna.obs['SpatialDDM'] = adata_results.obs['SpatialDDM']
adata_atac.obs['SpatialDDM'] = adata_results.obs['SpatialDDM']
adata_rna.obsm['SpatialDDM'] = adata_results.obsm['SpatialDDM']
adata_atac.obsm['SpatialDDM'] = adata_results.obsm['SpatialDDM']


pred_label = list(adata_rna.obs['SpatialDDM'])
mapping = {'1': 'Mesenchyme',
           '2': 'DH',
           '3': 'DPallv',
           '4': 'Subpallium_1',
           '5': 'DPallm',
           '6': 'Subpallium_1',
           '7': 'Cartilage_2',
           '8': 'Cartilage_1',
           '9': 'Muscle',
           '10': 'DPallm',
           '11': 'Cartilage_3',
           '12': 'Cartilage_4',
           '13': 'DPallm',
           '14': 'Thalamus',
           }

annot = [mapping[i] for i in pred_label]
adata_rna.obs['annotation'] = pd.Categorical(annot)
adata_atac.obs['annotation'] = pd.Categorical(annot)
adata_results.obs['annotation'] = pd.Categorical(annot)

adata_results.obs['Combined_Clusters_annotation'] = adata_results.obs['Combined_Clusters_annotation'].replace({
                                                                           'Basal_plate_of_hindbrain': 'BPH',
                                                                           'Diencephalon_and_hindbrain': 'DH'
                                                                           })


adata_rna.obsm['X_pca'] = adata_results.obsm['feat']
adata_rna.obsm['emb_latent'] = adata_results.obsm['emb_latent_omics_1']
adata_rna.obsm['Diff_recon'] = adata_results.obsm['rec_omics_1']
adata_atac.obsm['X_lsi'] = adata_results.obsm['ATAC_X_lsi']
adata_atac.obsm['emb_latent'] = adata_results.obsm['emb_latent_omics_2']
adata_atac.obsm['Diff_recon'] = adata_results.obsm['rec_omics_2']


sc.pp.neighbors(adata_atac, use_rep='X_lsi')
sc.tl.umap(adata_atac)
sc.pp.neighbors(adata_rna, use_rep='X_pca')
sc.tl.umap(adata_rna)
sc.pp.neighbors(adata_results, use_rep='SpatialDDM')
sc.tl.umap(adata_results)

plot_color = ["#458A74", "#F5A216", "#57AF37", "#41B9c1", "#008B8B",
              "#4E5689", "#6A8EC9", "#652884", "#8A7355", "#CC5B45",
              "#848484", "#EC3E31", "#B46DA9", "#F59B79"]
adata_atac.uns['Combined_Clusters_annotation_colors'] = plot_color
adata_rna.uns['Combined_Clusters_annotation_colors'] = plot_color
adata_atac.uns['annotation_colors'] = plot_color
adata_rna.uns['annotation_colors'] = plot_color
adata_results.uns['annotation_colors'] = plot_color
adata_results.uns['Combined_Clusters_annotation_colors'] = plot_color
adata_results.uns['Combined_Clusters_colors'] = plot_color
adata_atac.uns['ATAC_Clusters_colors'] = plot_color
adata_rna.uns['RNA_Clusters_colors'] = plot_color


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))
sc.pl.umap(adata_atac, color='ATAC_Clusters', ax=ax1, show=False, size=30)
ax1.spines['right'].set_visible(False) # 去掉边框
ax1.spines['top'].set_visible(False)   # 去掉边框
ax1.spines['left'].set_visible(False) # 去掉边框
ax1.spines['bottom'].set_visible(False)   # 去掉边框
ax2.legend().remove()
ax1.get_yaxis().set_visible(False)
ax1.get_xaxis().set_visible(False)
ax1.set_title("Chromatin Accessibility (ATAC_Clusters)", fontsize=12)

sc.pl.umap(adata_rna, color='RNA_Clusters', ax=ax2, show=False, size=30)
ax2.spines['right'].set_visible(False) # 去掉边框
ax2.spines['top'].set_visible(False)   # 去掉边框
ax2.spines['left'].set_visible(False) # 去掉边框
ax2.spines['bottom'].set_visible(False)   # 去掉边框
ax2.get_yaxis().set_visible(False)
ax2.get_xaxis().set_visible(False)
ax2.set_title("Gene Expression (RNA_Clusters)", fontsize=12)
plt.tight_layout(pad=3.0)
plt.show()
# ax2.legend().remove()

sc.pl.umap(adata_results, color='Combined_Clusters', ax=ax3, show=False, size=30)
ax3.spines['right'].set_visible(False) # 去掉边框
ax3.spines['top'].set_visible(False)   # 去掉边框
ax3.spines['left'].set_visible(False) # 去掉边框
ax3.spines['bottom'].set_visible(False)   # 去掉边框
ax3.get_yaxis().set_visible(False)
ax3.get_xaxis().set_visible(False)
ax3.set_title("Multi-omics integrating (Combined_Clusters)", fontsize=12)
plt.tight_layout(pad=3.0)
plt.show()
# ax3.legend().remove()
plt.savefig('Graph_Diffusion/MISAR/E18_5-S1/Umap_Clusters.pdf', dpi=300)

"""
fig_legend = plt.figure(figsize=(12, 1.2))  # 宽一点，矮一点
legend_ax = fig_legend.add_subplot(111)
legend_ax.axis("off")  # 不显示轴
# 获取图例句柄与标签
handles, labels = ax1.get_legend_handles_labels()
# 横向图例：设置 ncol 为算法数量
legend_ax.legend(
    handles,
    labels,
    loc="center",
    frameon=False,
    fontsize=10,
    ncol=14,  # 横向排列
    handletextpad=0.8,
    columnspacing=1.5
)
"""

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 2))
sc.pl.embedding(adata_results, basis='spatial', color=["annotation"], ax=ax1, show=False, size=20)
ax1.invert_yaxis()
# ax1.spines['right'].set_visible(False) # 去掉边框
# ax1.spines['top'].set_visible(False)   # 去掉边框
# ax1.spines['left'].set_visible(False) # 去掉边框
# ax1.spines['bottom'].set_visible(False)   # 去掉边框
ax1.get_yaxis().set_visible(False)
ax1.get_xaxis().set_visible(False)
ax1.legend().remove()
ax1.set_title(" ", fontsize=8)

sc.pl.umap(adata_results, color=["annotation"], ax=ax2, show=False, size=10)
# ax2.spines['right'].set_visible(False) # 去掉边框
# ax2.spines['top'].set_visible(False)   # 去掉边框
# ax2.spines['left'].set_visible(False) # 去掉边框
# ax2.spines['bottom'].set_visible(False)   # 去掉边框
ax2.get_yaxis().set_visible(False)
ax2.get_xaxis().set_visible(False)
ax2.legend().remove()
ax2.set_title(" ", fontsize=8)
plt.tight_layout(pad=1.0)
plt.savefig('Graph_Diffusion/MISAR/E18_5-S1/annotation_visualization.pdf', dpi=300)


fig_legend = plt.figure(figsize=(4, 1.5))  # 宽一点，矮一点
legend_ax = fig_legend.add_subplot(111)
legend_ax.axis("off")  # 不显示轴
# 获取图例句柄与标签
handles, labels = ax1.get_legend_handles_labels()
# 横向图例：设置 ncol 为算法数量
legend_ax.legend(
    handles,
    labels,
    loc="center",
    frameon=False,
    fontsize=7,
    ncol=4,  # 横向排列
    handletextpad=0.8,
    columnspacing=1.5
)
plt.tight_layout(pad=1.0)
plt.savefig('Graph_Diffusion/MISAR/E18_5-S1/annotation_visualization_legend.pdf', dpi=300)




fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8,2.0))
sc.pl.umap(adata_atac, color='annotation', ax=ax1, show=False, size=20)
ax1.spines['right'].set_visible(False) # 去掉边框
ax1.spines['top'].set_visible(False)   # 去掉边框
ax1.spines['left'].set_visible(False) # 去掉边框
ax1.spines['bottom'].set_visible(False)   # 去掉边框
ax1.get_yaxis().set_visible(False)
ax1.get_xaxis().set_visible(False)
ax1.legend().remove()
ax1.set_title("Chromatin Accessibility", fontsize=8)

sc.pl.umap(adata_rna, color='annotation', ax=ax2, show=False, size=20)
ax2.spines['right'].set_visible(False) # 去掉边框
ax2.spines['top'].set_visible(False)   # 去掉边框
ax2.spines['left'].set_visible(False) # 去掉边框
ax2.spines['bottom'].set_visible(False)   # 去掉边框
ax2.get_yaxis().set_visible(False)
ax2.get_xaxis().set_visible(False)
ax2.legend().remove()
ax2.set_title("Gene Expression", fontsize=8)


sc.pl.umap(adata_results, color='annotation', ax=ax3, show=False, size=20)
ax3.spines['right'].set_visible(False) # 去掉边框
ax3.spines['top'].set_visible(False)   # 去掉边框
ax3.spines['left'].set_visible(False) # 去掉边框
ax3.spines['bottom'].set_visible(False)   # 去掉边框
ax3.get_yaxis().set_visible(False)
ax3.get_xaxis().set_visible(False)
ax3.legend().remove()
ax3.set_title("Multi-omics integrating", fontsize=8)
plt.tight_layout(pad=0.5)
plt.savefig('Graph_Diffusion/MISAR/E18_5-S1/Umap_annotation-8.pdf', dpi=300)


fig_legend = plt.figure(figsize=(8, 1.5))  # 宽一点，矮一点
legend_ax = fig_legend.add_subplot(111)
legend_ax.axis("off")  # 不显示轴
# 获取图例句柄与标签
handles, labels = ax1.get_legend_handles_labels()
# 横向图例：设置 ncol 为算法数量
legend_ax.legend(
    handles,
    labels,
    loc="center",
    frameon=False,
    fontsize=7,
    ncol=6,  # 横向排列
    handletextpad=0.8,
    columnspacing=1.5
)
plt.savefig('Graph_Diffusion/MISAR/E18_5-S1/Umap_annotation legend-8.pdf', dpi=300)



tp = TranslationProblem(adata_src=adata_atac, adata_tgt=adata_rna)
tp = tp.prepare(src_attr="X_lsi", tgt_attr="X_pca")
tp = tp.solve(alpha=1.0, epsilon=1e-3)

translated = tp.translate(source="src", target="tgt", forward=True)

print(
    "Average FOSCTTM score of translating ATAC onto RNA: ",
    foscttm(adata_rna.obsm["X_pca"], translated),
)

adata = sc.concat(
    [adata_atac, adata_rna],
    join="outer",
    label="batch",
    keys=["ATAC (translated)", "RNA"],
)
adata.obsm["X_translated_1"] = np.concatenate(
    (translated, adata_rna.obsm["X_pca"]), axis=0
)
sc.pp.neighbors(adata, use_rep="X_translated_1")
sc.tl.umap(adata)
adata.uns['annotation_colors'] = plot_color


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 2))
sc.pl.umap(adata, color=["batch"], ax=ax1, show=False, size=7)
ax1.spines['right'].set_visible(False) # 去掉边框
ax1.spines['top'].set_visible(False)   # 去掉边框
ax1.spines['left'].set_visible(False) # 去掉边框
ax1.spines['bottom'].set_visible(False)   # 去掉边框
ax1.get_yaxis().set_visible(False)
ax1.get_xaxis().set_visible(False)
ax1.legend().remove()
ax1.set_title("Modality", fontsize=8)

sc.pl.umap(adata, color=["annotation"], ax=ax2, show=False, size=7)
ax2.spines['right'].set_visible(False) # 去掉边框
ax2.spines['top'].set_visible(False)   # 去掉边框
ax2.spines['left'].set_visible(False) # 去掉边框
ax2.spines['bottom'].set_visible(False)   # 去掉边框
ax2.get_yaxis().set_visible(False)
ax2.get_xaxis().set_visible(False)
ax2.legend().remove()
ax2.set_title("Clustering", fontsize=8)
plt.savefig('Graph_Diffusion/MISAR/E18_5-S1/Umap_alignment.pdf', dpi=300)


ftp = TranslationProblem(adata_src=adata_atac, adata_tgt=adata_rna)
ftp = ftp.prepare(src_attr="emb_latent", tgt_attr="emb_latent")
ftp = ftp.solve(alpha=1.0, epsilon=1e-3)
translated_fused = ftp.translate(source="src", target="tgt", forward=True)
print(
    "Average FOSCTTM score for translating ATAC onto RNA: ",
    foscttm(adata_rna.obsm["emb_latent"], translated_fused),
)


adata.obsm["X_translated_2"] = np.concatenate(
    (translated_fused, adata_rna.obsm["emb_latent"]), axis=0
)
sc.pp.neighbors(adata, use_rep="X_translated_2")
sc.tl.umap(adata)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 2))
sc.pl.umap(adata, color=["batch"], ax=ax1, show=False, size=7)
ax1.spines['right'].set_visible(False) # 去掉边框
ax1.spines['top'].set_visible(False)   # 去掉边框
ax1.spines['left'].set_visible(False) # 去掉边框
ax1.spines['bottom'].set_visible(False)   # 去掉边框
ax1.get_yaxis().set_visible(False)
ax1.get_xaxis().set_visible(False)
ax1.set_title("Modality", fontsize=8)
ax1.legend().remove()


sc.pl.umap(adata, color=["annotation"], ax=ax2, show=False, size=7)
ax2.spines['right'].set_visible(False) # 去掉边框
ax2.spines['top'].set_visible(False)   # 去掉边框
ax2.spines['left'].set_visible(False) # 去掉边框
ax2.spines['bottom'].set_visible(False)   # 去掉边框
ax2.get_yaxis().set_visible(False)
ax2.get_xaxis().set_visible(False)
ax2.set_title("Clustering", fontsize=8)
ax2.legend().remove()
plt.show()
plt.savefig('Graph_Diffusion/MISAR/E18_5-S1/Umap_alignment_emb_latent.pdf', dpi=300)


### 绘制图例###
# 1. 绘制 SpaDDM 的 spatial 图，用于获取 legend（不显示）
ax = sc.pl.embedding(adata, basis='spatial', color=["annotation"], s=10, show=False)
# 2. 兼容性处理：如果是列表，取第一个 Axes
if isinstance(ax, list):
    ax = ax[0]
# 3. 创建一个新的图例画布
fig_legend = plt.figure(figsize=(1.2, 4))  # 根据图例数量自定义尺寸
legend_ax = fig_legend.add_subplot(111)
legend_ax.axis("off")
# 4. 提取图例句柄与标签
handles, labels = ax.get_legend_handles_labels()
# 5. 横向排布图例
legend_ax.legend(
    handles,
    labels,
    loc="center",
    frameon=False,
    fontsize=8,
    ncol=1,           # 图例横向排列列数（根据需要调整）
    handletextpad=0.8,
    columnspacing=1.5
)
plt.show()
plt.savefig('Graph_Diffusion/MISAR/E18_5-S1/Umap_alignment_legend_domains.pdf', dpi=300)


### 绘制图例###
# 1. 绘制 SpaDDM 的 spatial 图，用于获取 legend（不显示）
ax = sc.pl.embedding(adata, basis='spatial', color=["batch"], s=10, show=False)
# 2. 兼容性处理：如果是列表，取第一个 Axes
if isinstance(ax, list):
    ax = ax[0]
# 3. 创建一个新的图例画布
fig_legend = plt.figure(figsize=(4, 1.2))  # 根据图例数量自定义尺寸
legend_ax = fig_legend.add_subplot(111)
legend_ax.axis("off")
# 4. 提取图例句柄与标签
handles, labels = ax.get_legend_handles_labels()
# 5. 横向排布图例
legend_ax.legend(
    handles,
    labels,
    loc="center",
    frameon=False,
    fontsize=8,
    ncol=1,           # 图例横向排列列数（根据需要调整）
    handletextpad=0.8,
    columnspacing=1.5
)
plt.show()
plt.savefig('Graph_Diffusion/MISAR/E18_5-S1/Umap_alignment_legend_omics.pdf', dpi=300)





ftp = TranslationProblem(adata_src=adata_atac, adata_tgt=adata_rna)
ftp = ftp.prepare(src_attr="emb_latent", tgt_attr="emb_latent", joint_attr="SpatialDDM")
ftp = ftp.solve(epsilon=0.5e-2, alpha=0.5)
translated_fused = ftp.translate(source="src", target="tgt", forward=True)
print(
    "Average FOSCTTM score for translating ATAC onto RNA: ",
    foscttm(adata_rna.obsm["emb_latent"], translated_fused),
)

adata = sc.concat(
    [adata_atac, adata_rna],
    join="outer",
    label="batch",
    keys=["ATAC (translated)", "RNA"],
)
adata.obsm["X_translated_2"] = np.concatenate(
    (translated_fused, adata_rna.obsm["emb_latent"]), axis=0
)
sc.pp.neighbors(adata, use_rep="X_translated_2")
sc.tl.umap(adata)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
sc.pl.umap(adata, color=["batch"], ax=ax1, show=False, size=30)
ax1.spines['right'].set_visible(False) # 去掉边框
ax1.spines['top'].set_visible(False)   # 去掉边框
ax1.spines['left'].set_visible(False) # 去掉边框
ax1.spines['bottom'].set_visible(False)   # 去掉边框
ax1.get_yaxis().set_visible(False)
ax1.get_xaxis().set_visible(False)
ax1.legend()
ax1.set_title("Modality", fontsize=15)

sc.pl.umap(adata, color=["annotation"], ax=ax2, show=False, size=30)
ax2.spines['right'].set_visible(False) # 去掉边框
ax2.spines['top'].set_visible(False)   # 去掉边框
ax2.spines['left'].set_visible(False) # 去掉边框
ax2.spines['bottom'].set_visible(False)   # 去掉边框
ax2.get_yaxis().set_visible(False)
ax2.get_xaxis().set_visible(False)
ax2.set_title("Clustering", fontsize=15)
plt.tight_layout(pad=3.0)
plt.show()
plt.savefig('Graph_Diffusion/MISAR/E18_5-S1/Umap_alignment_emb_latent_DDM_joint.pdf', dpi=300)




order = adata_atac.obs["annotation"].cat.categories

cell_transition = ftp.cell_transition(
    source="src",
    target="tgt",
    source_groups={"annotation": order},
    target_groups={"annotation": order},
    forward=True,
    key_added="cell_transition",
)
# mtp.cell_transition(ftp.adata, fontsize=10)

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 示例转移概率矩阵
cell_types = list(cell_transition.index)

# 绘制热图
plt.figure(figsize=(4, 4))
ax = sns.heatmap(cell_transition, annot=True, fmt=".2f", cmap="viridis",
                 xticklabels=cell_types, yticklabels=cell_types,
                 cbar=True, linewidths=0.7, square=True, annot_kws={"size": 6}, cbar_kws={"shrink": 0.6})

# 移动 X 轴标签到顶部
ax.xaxis.set_label_position('top')  # X 轴标签移动到顶部
ax.xaxis.tick_top()  # X 轴刻度标签也移动到顶部
# 旋转 X 轴顶部的标签，使其竖直显示
plt.xticks(rotation=90)  # 90° 旋转 X 轴标签（使其竖直）

# 设置轴标签
ax.set_xlabel("target", fontsize=9)
ax.set_ylabel("source", fontsize=9)
# 设置刻度字体大小
ax.tick_params(axis='x', labelsize=7)
ax.tick_params(axis='y', labelsize=7)
# plt.title("Spatial Domains Transition Probability", pad=20)  # 增加标题与图像间距
plt.show()
plt.savefig('Graph_Diffusion/MISAR/E18_5-S1/Transition_heatmap_emb_E18.5.pdf', dpi=300)