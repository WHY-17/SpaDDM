library(dplyr)
library(Seurat)
library(patchwork)
library(Signac)
library(SeuratDisk)
library(anndata)
library(reticulate)



use_python("D:/Anaconda/envs/ST/python.exe", required = TRUE)
scanpy <- import("scanpy")
file_fold <- "Data_SpatialGlue/Dataset7_Mouse_Brain_ATAC/"

P22mousebrain <- readRDS(paste0(file_fold, 'P22mousebrain_spatial_RNA_ATAC.rds'))

DefaultAssay(P22mousebrain) -> "Spatial"
P22mousebrain <- FindVariableFeatures(P22mousebrain, selection.method = "vst", nfeatures = 3000)
P22mousebrain <- SCTransform(P22mousebrain, assay = "Spatial", verbose = FALSE)
P22mousebrain <- RunPCA(P22mousebrain, assay = "SCT", verbose = FALSE)
P22mousebrain <- FindNeighbors(P22mousebrain, reduction = "pca", dims = 1:10, assay = "Spatial")
P22mousebrain <- FindClusters(P22mousebrain, verbose = FALSE, resolution = 1.35)   
P22mousebrain <- RunUMAP(P22mousebrain, reduction = "pca", dims = 1:10, assay = "Spatial")
SpatialDimPlot(P22mousebrain)
DimPlot(P22mousebrain, label = T)
folder_path <- paste0(file_fold, 'Seurat')
# 检查文件夹是否存在  
if (!dir.exists(folder_path)) {  
  # 如果文件夹不存在，则创建文件夹  
  dir.create(folder_path, recursive = TRUE)  # recursive=TRUE 可以创建多层文件夹  
  cat("文件夹已创建：", folder_path, "\n")  
} else {  
  cat("文件夹已存在：", folder_path, "\n")  
}  
write.csv(P22mousebrain@meta.data, paste0(file_fold, 'Seurat/rna_meta.csv'))

DefaultAssay(P22mousebrain) <- "peaks"
P22mousebrain <- RunTFIDF(P22mousebrain, assay = "peaks")  #normalization
P22mousebrain <- FindTopFeatures(P22mousebrain, min.cutoff = 'q0', assay = "peaks")
P22mousebrain <- RunSVD(P22mousebrain, assay = "peaks")

P22mousebrain <- RunUMAP(object = P22mousebrain, reduction = 'lsi', dims = 2:10)
P22mousebrain <- FindNeighbors(object = P22mousebrain, reduction = 'lsi', dims = 2:10)
P22mousebrain <- FindClusters(object = P22mousebrain, verbose = FALSE, resolution = 0.8) 
DimPlot(object = P22mousebrain, label = TRUE) 
SpatialDimPlot(P22mousebrain)
write.csv(P22mousebrain@meta.data, paste0(file_fold, 'Seurat/atac_meta.csv'))

P22mousebrain <- FindMultiModalNeighbors(P22mousebrain, reduction.list = list("pca", "lsi"), dims.list = list(1:10, 2:10))
P22mousebrain <- RunUMAP(P22mousebrain, nn.name = "weighted.nn", reduction.name = "wnn.umap", reduction.key = "wnnUMAP_")
# 确定最优resolution
n_cluster <- 18
seurat_object <- P22mousebrain
for (res in seq(0.1, 2, by = 0.1)) {
  seurat_object <- FindClusters(seurat_object, graph.name = "wsnn", algorithm = 3, resolution = res)
  num_clusters <- length(unique(seurat_object$seurat_clusters))
  cat("Resolution:", res, " Number of clusters:", num_clusters, "\n")
  if (num_clusters == n_cluster){
    break
  }
}
P22mousebrain <- FindClusters(P22mousebrain, graph.name = "wsnn", verbose = FALSE, resolution = res)
DimPlot(P22mousebrain , reduction = "wnn.umap", label = T) #+ ggtitle("WNN")
write.csv(P22mousebrain@meta.data, paste0(file_fold, 'Seurat/wnn_meta.csv'))


SaveH5Seurat(P22mousebrain, filename = paste0(file_fold, "Seurat_results.h5Seurat"))
Convert(paste0(file_fold, "Seurat_results.h5Seurat"), dest = "h5ad")



#p1 <- DimPlot(P22mousebrain, reduction = "umap.rna", label = T) + ggtitle("RNA")
#p2 <- DimPlot(P22mousebrain , reduction = "umap.atac", label = T) + ggtitle("ATAC")
#p3 <- DimPlot(P22mousebrain , reduction = "wnn.umap", label = T) + ggtitle("WNN")
#p1 + p2 + p3 & NoLegend() & theme(plot.title = element_text(hjust = 0.5))

