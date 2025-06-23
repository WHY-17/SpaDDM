library(scAI)
library(dplyr)
library(cowplot)
library(ggplot2)
require(Matrix)
library(Signac)
library(anndata)
library(reticulate)

memory.limit()  # 查看当前的内存限制  
memory.limit(size = 36000)  # 将内存限制增加到16GB  


file_fold <- "Data/Dataset7_Mouse_Brain_ATAC/"
use_python("D:/Anaconda/envs/ST/python.exe", required = TRUE)
scanpy <- import("scanpy")

adata_rna <- scanpy$read_h5ad(paste0(file_fold, "adata_RNA.h5ad"))
adata_rna$var_names_make_unique()
adata_peak <- scanpy$read_h5ad(paste0(file_fold, "adata_Peaks.h5ad"))
adata_peak$var_names_make_unique()

# 提取基因表达矩阵 (X)，转换为 R 矩阵
M <- t(as.matrix(py_to_r(adata_rna$X)))
# 提取细胞元数据 (obs) 和基因元数据 (var)
cell_metadata_rna <- py_to_r(adata_rna$obs) # obs 表格
gene_metadata_rna <- py_to_r(adata_rna$var) # var 表格
row.names(M) <- row.names(gene_metadata_rna)
colnames(M) <- row.names(cell_metadata_rna)


atac_counts <- t(as.matrix(py_to_r(adata_peak$X)))
cell_metadata_peak <- py_to_r(adata_peak$obs) # obs 表格
gene_metadata_peak <- py_to_r(adata_peak$var) # var 表格
row.names(atac_counts) <- row.names(gene_metadata_peak)
colnames(atac_counts) <- row.names(cell_metadata_peak)



X <- list()
X$RNA <- M
X$ATAC <- atac_counts


start_time <- Sys.time()

scAI_outs <- create_scAIobject(raw.data = X)
scAI_outs <- preprocessing(scAI_outs, assay = list("RNA", "ATAC"), minFeatures = 1, minCells = 1)

scAI_outs <- run_scAI(scAI_outs, K = 20, nrun = 1, do.fast = T)

write.table(scAI_outs@fit$H, paste0(file_fold,'scAI.csv'), sep = ",", row.names = FALSE, col.names = FALSE)




