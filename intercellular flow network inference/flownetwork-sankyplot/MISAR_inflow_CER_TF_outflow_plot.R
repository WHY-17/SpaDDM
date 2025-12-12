library(ggsankey)
library(tidyverse)
library(readxl)
library(networkD3)
library(htmlwidgets)
library(ggsankey)
library(readxl)
library(tidyverse)



d <- read_excel("E18.5-S1/inflow-Edn1-inflow_cer_tf_outflow_four_layers_network-long.xlsx")


df <- d %>%
  make_long(inflow, CER, TF, outflow)


n_colors <- length(unique(df$node))
my_colors <- colorRampPalette(RColorBrewer::brewer.pal(12, "Set3"))(n_colors)


ggplot(df, aes(
  x = x,
  next_x = next_x,
  node = node,
  next_node = next_node,
  fill = factor(node),
  label = node
)) +
  geom_sankey(flow.alpha = .4, node.color = "gray10") +
  geom_sankey_label(size = 2.6, color = "black", hjust = 0) +
  scale_fill_manual(values = my_colors) +
  
  theme_minimal(base_size = 10) +
  theme(
    # —— 删除所有背景 ——
    panel.background = element_blank(),
    plot.background = element_blank(),
    legend.background = element_blank(),
    panel.grid = element_blank(),
    
    # —— 删除坐标轴所有内容 ——
    axis.title = element_blank(),
    axis.text = element_blank(),
    axis.ticks = element_blank(),
    
    # —— 不显示图例 ——
    legend.position = "none"
  )

