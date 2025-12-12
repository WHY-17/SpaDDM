library(tidyverse)
library(org.Hs.eg.db) 
library(org.Mm.eg.db)
library(clusterProfiler)
library(tibble) 
library(ggplot2)
library(reticulate)
options(timeout = 300)


file_fold <- "E18.5-S1/"
use_python("D:/Anaconda/envs/ST/python.exe", required = TRUE)
pickle <- import("pickle")
py <- import_builtins()


py_file <- py$open(paste0(file_fold, "contributions_to_target_CERs_top100.pkl"), "rb")
top_genes <- pickle$load(py_file)

top_genes_r <- py_to_r(top_genes)
all_genes <- unname(unlist(lapply(top_genes_r, function(df) df$gene)))
all_genes <- unique(all_genes)

genelist <- bitr(all_genes,
                 fromType = "SYMBOL", 
                 toType = "ENTREZID", 
                 OrgDb = 'org.Mm.eg.db')

# 进行GO富集分析
# go <- enrichGO(
#   gene = genelist$ENTREZID, 
#   OrgDb = org.Mm.eg.db,      # 小鼠数据库
#   ont = "ALL",               # 可以是 "BP", "CC", "MF" 或 "ALL"
#   pAdjustMethod = "BH", 
#   pvalueCutoff = 0.05, 
#   qvalueCutoff = 0.05, 
#   readable = TRUE
# )


# # 提取富集分析结果
# go_res <- go@result
# # 将GO富集结果保存为CSV文件
# write.csv(go_res, file = paste0(file_fold, "GO.csv"))

# 进行KEGG富集分析
# kegg <- enrichKEGG(gene = genelist$ENTREZID,
#                    organism = 'mmu',      
#                    pAdjustMethod = "BH", 
#                    pvalueCutoff = 0.05,   
#                    qvalueCutoff = 0.05,
#                    )    
#                    
# # 将KEGG结果中的基因ID转换为基因符号，便于阅读
# kegg <- setReadable(kegg, OrgDb = org.Mm.eg.db, keyType = 'ENTREZID')
# kegg_res <- kegg@result
# # 将KEGG富集结果保存为CSV文件
# write.csv(kegg_res, file = paste0(file_fold, "KEGG.csv"))
go_res <- read.csv(paste0(file_fold, "GO.csv"))
kegg_res <- read.csv(paste0(file_fold, "KEGG.csv"))
kegg_res$Description <- trimws(gsub("- Mus musculus \\(house mouse\\)$", "", kegg_res$Description))

go <- new("enrichResult",
          result = go_res,
          pvalueCutoff = 0.05,
          pAdjustMethod = "BH",
          qvalueCutoff = 0.05,
          organism = "UNKNOWN",
          keytype = "UNKNOWN")

kegg <- new("enrichResult",
            result = kegg_res,
            pvalueCutoff = 0.05,
            pAdjustMethod = "BH",
            qvalueCutoff = 0.05,
            organism = "UNKNOWN",
            keytype = "UNKNOWN")

p <- barplot(go, 
        drop = TRUE, 
        showCategory = 10, 
        split = "ONTOLOGY",
        label_format = 100,
        color = "pvalue") + facet_grid(ONTOLOGY~., scale = 'free') + ggtitle("GO enrichment analysis") + 
  theme(
    plot.title = element_text(size = 8, face = "bold"),
    axis.text.x = element_text(size = 8),     # x轴刻度字体
    axis.text.y = element_text(size = 8),     # y轴刻度字体
    axis.title = element_text(size = 8),      # x、y轴标题字体
    legend.text = element_text(size = 8),     # 图例文字
    legend.title = element_text(size = 8),    # 图例标题
    strip.text = element_text(size = 8)       # facet 标题（本体名）
  )
ggsave("go_enrichment analysis.pdf", p, dpi=300, width = 8, height = 5.5)




## 1. KEGG富集分析柱状图
p <- barplot(kegg, drop = TRUE, showCategory = 15, 
        label_format = 50, color = "pvalue") + ggtitle("KEGG enrichment analysis") + 
  theme(
    plot.title = element_text(size = 8, face = "bold"),
    axis.text.x = element_text(size = 8),     # x轴刻度字体
    axis.text.y = element_text(size = 8),     # y轴刻度字体
    axis.title = element_text(size = 8),      # x、y轴标题字体
    legend.text = element_text(size = 8),     # 图例文字
    legend.title = element_text(size = 8),    # 图例标题
    strip.text = element_text(size = 8)       # facet 标题（本体名）
  )
ggsave("kegg_enrichment analysis.pdf", p, dpi=300, width = 8, height = 5.5)


