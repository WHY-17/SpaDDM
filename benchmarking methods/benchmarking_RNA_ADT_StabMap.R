library(StabMap)
library(SingleCellMultiModal)
library(scran)
library(Seurat)
library(SeuratDisk)
library(MultiAssayExperiment)
library(Matrix)
library(rhdf5)
library(scater)
library(SummarizedExperiment)
library(zellkonverter)
library(scuttle)


data_name <- 'Dataset13_Simulation1'
genes <- read.csv(paste(data_name,'genes.csv', sep='_'))
genes <- genes[which(genes$highly_variable=='True'),]
genes <- genes$X

file_fold <- "Data_SpatialGlue/Dataset13_Simulation1/"
RNA <- readH5AD(paste0(file_fold, "adata_RNA.h5ad"))
ADT <- readH5AD(paste0(file_fold, "adata_ADT.h5ad"))


###################### Reprocessing ###################################################

rna_data <- RNA
adt_data <- ADT
mae <- MultiAssayExperiment::MultiAssayExperiment(experiments = list(rna = rna_data, atac = adt_data))

sce.rna <- experiments(mae)[["rna"]]
# Normalization
names(assays(sce.rna)) <- 'counts'
sce.rna <- logNormCounts(sce.rna)
zero_count_cols <- which(colSums(counts(sce.rna)) == 0)

# Replace the first row of counts for these columns with a very small number
if (length(zero_count_cols) > 0) {
  counts(sce.rna)[1, zero_count_cols] <- 0.000000001
}

# Feature selection
hvgs <- genes
length(hvgs)
hvgs <- toupper(hvgs)
rownames(sce.rna) <- toupper(rownames(sce.rna))
valid_hvgs <- intersect(hvgs, rownames(sce.rna))
length(valid_hvgs)
sce.rna <- sce.rna[valid_hvgs,]

sce.atac <- experiments(mae)[["atac"]]
names(assays(sce.atac)) <- 'counts'
zero_count_cols <- which(colSums(counts(sce.atac)) == 0)

# Replace the first row of counts for these columns with a very small number
if (length(zero_count_cols) > 0) {
  counts(sce.atac)[1, zero_count_cols] <- 0.000000001
}
sce.atac <- logNormCounts(sce.atac)
decomp <- modelGeneVar(sce.atac)

# for 7-10
# hvgs <- rownames(decomp)[decomp$mean>1
#                         | decomp$p.value <= 0.0000001]
# length(hvgs)
# sce.atac <- sce.atac[hvgs,]

# find the intersection
sce.atac <- sce.atac[,intersect(colnames(sce.rna),colnames(sce.atac))]
sce.rna <- sce.rna[,intersect(colnames(sce.rna),colnames(sce.atac))]
logcounts_all = rbind(logcounts(sce.rna), logcounts(sce.atac))
dim(logcounts_all)
assayType = ifelse(rownames(logcounts_all) %in% rownames(sce.rna), "rna", "atac")
table(assayType)



################ Indirect mosaic data integration with StabMap #####################################

dataTypeIndirect = setNames(sample(c("RNA", "Multiome", "ATAC"), ncol(logcounts_all),
                                   prob = c(0.3,0.3, 0.3), replace = TRUE),
                            colnames(logcounts_all))
table(dataTypeIndirect)
assay_list_indirect = list(
  RNA = logcounts_all[assayType %in% c("rna"), dataTypeIndirect %in% c("RNA")],
  Multiome = logcounts_all[assayType %in% c("rna", "atac"), dataTypeIndirect %in% c("Multiome")],
  ATAC = logcounts_all[assayType %in% c("atac"), dataTypeIndirect %in% c("ATAC")]
)

lapply(assay_list_indirect, dim)
lapply(assay_list_indirect, class)

mdt_indirect = mosaicDataTopology(assay_list_indirect)
mdt_indirect

stab_indirect = stabMap(assay_list_indirect,
                        reference_list = c("Multiome"),
                        plot = FALSE,
                        maxFeatures = 5000)
dim(stab_indirect)
stab_indirect[1:5,1:5]

folder_path <- paste0(file_fold, 'StapMap')
# 检查文件夹是否存在  
if (!dir.exists(folder_path)) {  
  # 如果文件夹不存在，则创建文件夹  
  dir.create(folder_path, recursive = TRUE)  # recursive=TRUE 可以创建多层文件夹  
  cat("文件夹已创建：", folder_path, "\n")  
} else {  
  cat("文件夹已存在：", folder_path, "\n")  
}  
write.csv(stab_indirect, paste0(file_fold, 'StapMap/d14_stabmap.csv'))



############################## Seurat clustering #######################################

minVal <- min(stab_indirect)
if (minVal < 0) {
  stab_indirect <- stab_indirect - minVal + 0.1
}
seurat_obj <- CreateSeuratObject(counts = t(stab_indirect))
seurat_obj <- NormalizeData(seurat_obj)
seurat_obj <- ScaleData(seurat_obj)
your_feature_list <- rownames(seurat_obj)
seurat_obj <- RunPCA(seurat_obj, features = your_feature_list)
seurat_obj <- FindNeighbors(seurat_obj)

# 确定最优resolution
n_cluster <- 5
seurat_object <- seurat_obj
for (res in seq(0.1, 2, by = 0.01)) {
  seurat_object <- FindClusters(seurat_object, resolution=res) #change the resolution
  num_clusters <- length(unique(seurat_object$seurat_clusters))
  cat("Resolution:", res, " Number of clusters:", num_clusters, "\n")
  if (num_clusters == n_cluster){
    break
  }
}

seurat_obj <- FindClusters(seurat_obj, resolution=res) #change the resolution
Idents(seurat_obj) <- seurat_obj$seurat_clusters
summary(as.factor(seurat_obj$seurat_clusters))

meta_data <- seurat_obj@meta.data
write.csv(meta_data, paste0(file_fold, "StapMap_clustering_results.csv"))

