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


adata_adt = sc.read_h5ad('Graph_Diffusion/Human lymph node/adata_ADT.h5ad')
adata_rna = sc.read_h5ad('Graph_Diffusion/Human lymph node/adata_RNA.h5ad')
adata_adt.var_names_make_unique()
adata_rna.var_names_make_unique()

# adata_atac.obsm["ATAC_lsi_l2_norm"] = pp.normalize(adata_atac.obsm["X_lsi"], norm="l2")
adata_results = sc.read_h5ad('Graph_Diffusion/Human lymph node/spatialDDM_results.h5ad')
adata_results.obs['SpatialDDM'] = adata_results.obs['SpatialDDM'].astype(str)
adata_rna.obs['SpatialDDM'] = adata_results.obs['SpatialDDM']
adata_adt.obs['SpatialDDM'] = adata_results.obs['SpatialDDM']
adata_rna.obsm['SpatialDDM'] = adata_results.obsm['SpatialDDM']
adata_adt.obsm['SpatialDDM'] = adata_results.obsm['SpatialDDM']

pred_label = list(adata_rna.obs['SpatialDDM'])
mapping = {'1': 'medulla',
           '2': 'capsule',
           '3': 'cortex',
           '4': 'medulla',
           '5': 'f/ss',
           '6': 'pat',
           }
annot = [mapping[i] for i in pred_label]
adata_rna.obs['annotation'] = pd.Categorical(annot)
adata_adt.obs['annotation'] = pd.Categorical(annot)
adata_results.obs['annotation'] = pd.Categorical(annot)

adata_rna.obsm['X_pca'] = adata_results.obsm['feat']
adata_rna.obsm['emb_latent'] = adata_results.obsm['emb_latent_omics_1']
adata_rna.obsm['Diff_recon'] = adata_results.obsm['rec_omics_1']
adata_adt.obsm['X_pca'] = adata_results.obsm['Protin_X_pca']
adata_adt.obsm['emb_latent'] = adata_results.obsm['emb_latent_omics_2']
adata_adt.obsm['Diff_recon'] = adata_results.obsm['rec_omics_2']


sc.pp.neighbors(adata_adt, use_rep='X_pca')
sc.tl.umap(adata_adt)
sc.pp.neighbors(adata_rna, use_rep='X_pca')
sc.tl.umap(adata_rna)
sc.pp.neighbors(adata_results, use_rep='SpatialDDM')
sc.tl.umap(adata_results)

plot_color = [ "#6A8EC9", "#652884", "#8A7355", "#CC5B45",
              "#848484", "#EC3E31", "#B46DA9", "#F59B79"]

adata_adt.uns['annotation_colors'] = plot_color
adata_rna.uns['annotation_colors'] = plot_color
adata_results.uns['annotation_colors'] = plot_color



fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))
sc.pl.umap(adata_adt, color="annotation", ax=ax1, show=False, size=30)
ax1.spines['right'].set_visible(False) # 去掉边框
ax1.spines['top'].set_visible(False)   # 去掉边框
ax1.spines['left'].set_visible(False) # 去掉边框
ax1.spines['bottom'].set_visible(False)   # 去掉边框
ax1.get_yaxis().set_visible(False)
ax1.get_xaxis().set_visible(False)
ax1.legend().remove()
ax1.set_title("Protein", fontsize=10)

sc.pl.umap(adata_rna, color="annotation", ax=ax2, show=False, size=30)
ax2.spines['right'].set_visible(False) # 去掉边框
ax2.spines['top'].set_visible(False)   # 去掉边框
ax2.spines['left'].set_visible(False) # 去掉边框
ax2.spines['bottom'].set_visible(False)   # 去掉边框
ax2.get_yaxis().set_visible(False)
ax2.get_xaxis().set_visible(False)
ax2.set_title("Gene Expression", fontsize=10)
plt.show()
ax2.legend().remove()

sc.pl.umap(adata_results, color='annotation', ax=ax3, show=False, size=30)
ax3.spines['right'].set_visible(False) # 去掉边框
ax3.spines['top'].set_visible(False)   # 去掉边框
ax3.spines['left'].set_visible(False) # 去掉边框
ax3.spines['bottom'].set_visible(False)   # 去掉边框
ax3.get_yaxis().set_visible(False)
ax3.get_xaxis().set_visible(False)
ax3.set_title("Multi-omics integrating", fontsize=10)
plt.show()
ax3.legend().remove()
plt.savefig('Graph_Diffusion/Human lymph node/A1_umap_annotation.pdf', dpi=300)


fig_legend = plt.figure(figsize=(10, 1.2))  # 宽一点，矮一点
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
    ncol=5,  # 横向排列
    handletextpad=0.8,
    columnspacing=1.5
)
plt.savefig('Graph_Diffusion/Human lymph node/A1_umap_annotation_legend.pdf', dpi=300)


tp = TranslationProblem(adata_src=adata_adt, adata_tgt=adata_rna)
tp = tp.prepare(src_attr="X_pca", tgt_attr="X_pca")
tp = tp.solve(alpha=1.0, epsilon=1e-3)

translated = tp.translate(source="src", target="tgt", forward=True)

print(
    "Average FOSCTTM score of translating ATAC onto RNA: ",
    foscttm(adata_rna.obsm["X_pca"], translated),
)

adata = sc.concat(
    [adata_adt, adata_rna],
    join="outer",
    label="batch",
    keys=["Protein (translated)", "RNA"],
)
adata.obsm["X_translated_1"] = np.concatenate(
    (translated, adata_rna.obsm["X_pca"]), axis=0
)
sc.pp.neighbors(adata, use_rep="X_translated_1")
sc.tl.umap(adata)
adata.uns['annotation_colors'] = plot_color

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 1.5))
sc.pl.umap(adata, color=["batch"], ax=ax1, show=False, size=6)
ax1.spines['right'].set_visible(False) # 去掉边框
ax1.spines['top'].set_visible(False)   # 去掉边框
ax1.spines['left'].set_visible(False) # 去掉边框
ax1.spines['bottom'].set_visible(False)   # 去掉边框
ax1.get_yaxis().set_visible(False)
ax1.get_xaxis().set_visible(False)
ax1.legend().remove()
ax1.set_title("Modality", fontsize=8)

sc.pl.umap(adata, color=["annotation"], ax=ax2, show=False, size=6)
ax2.spines['right'].set_visible(False) # 去掉边框
ax2.spines['top'].set_visible(False)   # 去掉边框
ax2.spines['left'].set_visible(False) # 去掉边框
ax2.spines['bottom'].set_visible(False)   # 去掉边框
ax2.get_yaxis().set_visible(False)
ax2.get_xaxis().set_visible(False)
ax2.legend().remove()
ax2.set_title("Clustering", fontsize=8)
plt.tight_layout(pad=0.2)
plt.savefig('Graph_Diffusion/Human lymph node/A1_umap_alignment.pdf', dpi=300)


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
plt.savefig('Graph_Diffusion/Human lymph node/Umap_alignment_legend_domains.pdf', dpi=300)


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
plt.savefig('Graph_Diffusion/Human lymph node/Umap_alignment_legend_omics.pdf', dpi=300)





tp = TranslationProblem(adata_src=adata_adt, adata_tgt=adata_rna)
tp = tp.prepare(src_attr="emb_latent", tgt_attr="emb_latent")
tp = tp.solve(alpha=1.0, epsilon=1e-3)

translated = tp.translate(source="src", target="tgt", forward=True)
print(
    "Average FOSCTTM score of translating ATAC onto RNA: ",
    foscttm(adata_rna.obsm["emb_latent"], translated),
)
adata = sc.concat(
    [adata_adt, adata_rna],
    join="outer",
    label="batch",
    keys=["Protein (translated)", "RNA"],
)
adata.obsm["X_translated_1"] = np.concatenate(
    (translated, adata_rna.obsm["emb_latent"]), axis=0
)
sc.pp.neighbors(adata, use_rep="X_translated_1")
sc.tl.umap(adata)

adata.uns['annotation_colors'] = plot_color

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 1.5))
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
plt.tight_layout(pad=0.2)
plt.savefig('Graph_Diffusion/Human lymph node/A1_umap_alignment_emb.pdf', dpi=300)




ftp = TranslationProblem(adata_src=adata_adt, adata_tgt=adata_rna)
ftp = ftp.prepare(src_attr="emb_latent", tgt_attr="emb_latent", joint_attr="SpatialDDM")
ftp = ftp.solve(epsilon=0.5e-2, alpha=0.5)
translated_fused = ftp.translate(source="src", target="tgt", forward=True)
print(
    "Average FOSCTTM score for translating ATAC onto RNA: ",
    foscttm(adata_rna.obsm["emb_latent"], translated_fused),
)
adata = sc.concat(
    [adata_adt, adata_rna],
    join="outer",
    label="batch",
    keys=["Protein (translated)", "RNA"],
)
adata.obsm["X_translated_2"] = np.concatenate(
    (translated_fused, adata_rna.obsm["emb_latent"]), axis=0
)
sc.pp.neighbors(adata, use_rep="X_translated_2")
sc.tl.umap(adata)
adata.uns['annotation_colors'] = plot_color

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
plt.savefig('Graph_Diffusion/Human lymph node/A1_umap_alignment_emb_joint_DDM.pdf', dpi=300)






order = adata_adt.obs["annotation"].cat.categories

cell_transition = tp.cell_transition(
    source="src",
    target="tgt",
    source_groups={"annotation": order},
    target_groups={"annotation": order},
    forward=True,
    key_added="cell_transition",
)
# mtp.cell_transition(tp.adata, fontsize=10)



import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 示例转移概率矩阵
cell_types = list(cell_transition.index)

# 绘制热图
plt.figure(figsize=(3, 3))
ax = sns.heatmap(cell_transition, annot=True, fmt=".2f", cmap="viridis",
                 xticklabels=cell_types, yticklabels=cell_types,
                 cbar=True, linewidths=0.7, square=True, annot_kws={"size": 8}, cbar_kws={"shrink": 0.6})

# 移动 X 轴标签到顶部
ax.xaxis.set_label_position('top')  # X 轴标签移动到顶部
ax.xaxis.tick_top()  # X 轴刻度标签也移动到顶部
# 旋转 X 轴顶部的标签，使其竖直显示
# plt.xticks(rotation=90)  # 90° 旋转 X 轴标签（使其竖直）
# plt.yticks(rotation=90)  # 90° 旋转 X 轴标签（使其竖直）

# 设置轴标签
ax.set_xlabel("target", fontsize=9)
ax.set_ylabel("source", fontsize=9)
# 设置刻度字体大小
ax.tick_params(axis='x', labelsize=7)
ax.tick_params(axis='y', labelsize=7)
# plt.title("Spatial Domains Transition Probability", pad=20)  # 增加标题与图像间距
plt.show()
plt.savefig('Graph_Diffusion/Human lymph node/Transition_heatmap_A1.pdf', dpi=300)
