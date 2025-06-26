# SpaDDM: Integrating Spatial Multi-omics with Directional Graph Diffusion Models to Dissect Spatial Patterning and Signaling
Here, we present SpaDDM, a spatial multi-omics integration framework based on directional diffusion models (DDMs), which supports spatial pattern identification, cross-omics alignment, and inter-and intracellular signaling flow analysis.
![SpaDDM workflow](https://github.com/WHY-17/SpaDDM/blob/main/SpaDDM%20framework.jpg)
# Overiew
The SpaDDM model is designed for the integration analysis of spatial multi-omics data, including spatial domain identification, spatial signal flow inference, and cross-omics data translation, among other downstream tasks. SpaDDM is a deep learning framework based on graph diffusion models, with inputs consisting of omics features and spatial coordinates of cells or spots. It is a flexible computational framework that is not limited by the spatial resolution or platform technology of omics data. 
# Requirements
You'll need to install the following packages in order to run the codes.
<pre lang="markdown"> 
  python = 3.8.16
  anndata = 0.9.2
  cosg = 1.0.3
  dgl = 1.1.2+cu116
  h5py = 3.11.0
  igraph = 0.11.8
  imageio = 2.35.1
  networkx = 3.1
  matplotlib = 3.7.5
  numpy = 1.24.4
  pillow = 10.4.0
  rpy2 = 3.5.16
  scanpy = 1.9.8
  torch = 1.13.1+cu116
  torch-cluster = 1.6.1+pt113cu116
  torch-geometric = 2.6.1
  torch-scatter = 2.1.1+pt113cu116
  torch-sparse = 0.6.17+pt113cu116
  torch-spline-conv = 1.2.2+pt113cu116
  torchsummary = 1.5.1
  torchvision = 0.14.1  </pre>
Moreover, you need install R-4.3.3 and R package mclust.
# Tutorial
## For spatial multi-omics clustering and low dimensional coembedded representation learning: 
For the step-by-step tutorial, please refer to 
# Datasets
All spatial multi-omics datasets analyzed in this study are publicly available. The spatial epigenome-transcriptome P22 mouse brain data can be downloaded from (https://cells.ucsc.edu/?ds=brain-spatial-omics) or from AtlasXplore (https://web.atlasxomics.com/visualization/Fan). The Stereo-CITE-seq mouse thymus data can also be obtained from AtlasXplore (https://web.atlasxomics.com/visualization/Fan). SPOTS mouse spleen data was obtained from the Gene Expression Omnibus (GEO) repository under accession no. GSE198353. 10x Visium human lymph node data was acquired from the GEO repository under accession no. GSE263617. The mouse embryonic brain datasets acquired using MISAR-seq were downloaded from https://www.biosino.org/node/project/detail/OEP003285. All datasets used in this study, along with the benchmarking results including spatial clustering results, low dimensional representations and UMAP results of all compared methods, are available at https://zenodo.org/uploads/15681100.
# Benchmarking
In the paper, we compared SpaDDM with 7 state-of-the-art single-cell or spatial multi-omics integration methods, including Seurat, totalVI, MultiVI, scMM, scAI, StabMap and SpatialGlue. Pipelines covering the benchmarking analysis in this paper are available at https://github.com/WHY-17/SpaDDM/tree/main/benchmarking%20methods.

