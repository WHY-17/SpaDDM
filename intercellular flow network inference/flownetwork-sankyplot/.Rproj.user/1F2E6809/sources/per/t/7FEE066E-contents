library(ggsankey)
library(tidyverse)
library(readxl)
library(networkD3)
library(htmlwidgets)

#Create data which can be used for Sankey
links <- read_excel("Human lymph node/A1/flow_network.xlsx")
# From these flows we need to create a node data frame: it lists every entities involved in the flow
nodes <- data.frame(name = unique(c(links$source, links$target)))
nodes <- na.omit(nodes)

# With networkD3, connection must be provided using id, not using real name like in the links dataframe. So we need to reformat it.
# 创建映射表
links$source <- match(links$source, nodes$name)-1
links$target <- match(links$target, nodes$name)-1
links <- na.omit(links)


# Plot
# 假设你已经有 links 和 nodes 数据
p <- sankeyNetwork(Links = links, Nodes = nodes,
                   Source = "source", Target = "target", Value = "weight",
                   NodeID = "name", fontSize = 12, nodeWidth = 20)

# 保存为临时 HTML
saveWidget(p, "Human lymph node/A1/sankey_temp.html", selfcontained = TRUE)

# 截图为 PNG，设置分辨率为 300dpi（即高分辨率）
webshot("sankey_temp.html", file = "sankey_output.png",
        vwidth = 2400, vheight = 1800, zoom = 2)

# # 找到并设置Google路径
# find_chrome <- function() {
#   possible_paths <- c(
#     "C:/Program Files/Google/Chrome/Application/chrome.exe",
#     "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
#     paste0(Sys.getenv("LOCALAPPDATA"), "/Google/Chrome/Application/chrome.exe")
#   )
#   for (path in possible_paths) {
#     if (file.exists(path)) return(path)
#   }
#   return(NULL)
# }
# 
# chrome_path <- find_chrome()
# if (!is.null(chrome_path)) {
#   Sys.setenv(CHROMOTE_CHROME = chrome_path)
#   message("✅ 找到了 Chrome：", chrome_path)
# } else {
#   message("❌ 没有找到 Chrome，请确认是否安装了 Google Chrome。")
# }



# # And with a different font
# sankeyNetwork(Links = links, Nodes = nodes, Source = "source",
#               Target = "target", Value = "value", NodeID = "name",
#               fontSize = 15, nodeWidth = 30, fontFamily = "monospace")







library(ggsankey)
library(readxl)
library(tidyverse)


d <- read_excel("Human lymph node/A1/inflow-ANGPTL4-long.xlsx")

# 转为桑基图长格式
df <- d %>%
  make_long(inflow, CER, outflow)



ggplot(df, aes(x = x, next_x = next_x, node = node, next_node = next_node, fill = factor(node), label = node)) +
  geom_sankey(flow.alpha = .4,
              node.color = "gray30") +
  geom_sankey_label(size = 3.2, color = "black", hjust = 0) +
  # scale_fill_brewer(palette = "Set3", drop = FALSE) +
  # scale_fill_viridis_d(option = "I", drop = FALSE)+
  scale_fill_manual(values = c(
    "ANGPTL4" = "#1F77B4",  # 蓝色，保持一致
    "CER-18"         = "#FF7F0E",  # 橙色
    "C3"             = "#2CA02C",  # 绿色
    "CER-10"         = "#D62728",  # 红色
    "CCL14"          = "#9467BD",  # 紫色
    "CCL17"          = "#8C564B",  # 棕红色
    "CCL21"          = "#E377C2",  # 粉紫色
    "ADIPOQ"         = "#17BECF",  # 青蓝色，替换原灰
    "CCL5"           = "#BCBD22",  # 黄绿
    "ANGPTL4"        = "#AEC7E8",  # 淡蓝，较柔和
    "ANXA1"          = "#FFBB78",  # 浅橙，替换灰色系
    "CER-4"          = "#98DF8A",  # 浅绿
    "CCL22"          = "#C5B0D5",  # 紫灰色（仍保留有色彩）
    "WNT11"          = "#F7B6D2",  # 粉色
    "CCL19"          = "#FF9896",  # 粉红红
    "ANGPTL2"        = "#C49C94",  # 棕褐色
    "CER-1"          = "#9EDAE5",  # 蓝绿
    "GRN"            = "#B2DF8A",  # 柔和绿
    "IL16"           = "#FDBF6F",  # 柔和橙
    "CXCL2"          = "#FFCC99",  # 替换灰色，淡橙色
    "CER-6"          = "#A6CEE3",  # 替换灰，淡蓝色
    "FLT3LG"         = "#8DD3C7"   # 蓝绿色系
  ))+ theme_minimal(base_size = 12) +
  theme(
    panel.background = element_blank(),     # 去除面板背景
    plot.background = element_blank(),      # 去除整个图的背景
    legend.background = element_blank(),    # 去除图例背景
    panel.grid = element_blank(),           # 去除网格线
    axis.title = element_blank(),           # 不显示轴标题
    axis.text = element_blank(),            # 不显示轴标签
    axis.ticks = element_blank(),           # 不显示刻度线
    legend.title = element_text(size = 10),
    legend.text = element_text(size = 9),
    legend.key = element_blank()            # 去除图例项的灰色方块背景
  )




d <- read_excel("Human lymph node/A1/ANGPTL4-long.xlsx")

# 转为桑基图长格式
df <- d %>%
  make_long(inflow, CER, outflow)



ggplot(df, aes(x = x, next_x = next_x, node = node, next_node = next_node, fill = factor(node), label = node)) +
  geom_sankey(flow.alpha = .4,
              node.color = "gray30") +
  geom_sankey_label(size = 3.2, color = "black", hjust = 0) +
  # scale_fill_brewer(palette = "Set3", drop = FALSE) +
  # scale_fill_viridis_d(option = "I", drop = FALSE)+
  scale_fill_manual(values = c(
    # inflow nodes — 冷色调（蓝绿紫青）
    "ANGPTL4"   = "#1F77B4",  # 蓝
    "GDF15"     = "#17BECF",  # 青蓝
    "C3"        = "#2CA02C",  # 绿
    "PDGFD"     = "#9467BD",  # 紫
    "CCL5"      = "#E377C2",  # 粉紫
    "TNFSF12"   = "#7FC97F",  # 亮绿
    "CXCL12"    = "#BCBD22",  # 黄绿
    "CCL8"      = "#8DA0CB",  # 浅蓝紫
    "TNFSF13B"  = "#A6CEE3",  # 冰蓝
    "IL7"       = "#B3DE69",  # 草绿
    
    # other nodes — 暖色、柔和、黄橙红棕紫
    "ANGPTL4"          = "#FF7F0E",  # 橙
    "CER-1"            = "#FDBF6F",  # 橙黄
    "CER-2"            = "#FFBB78",  # 柔橙
    "CER-3"            = "#F781BF",  # 粉
    "CER-4"            = "#D62728",  # 红
    "CER-6"            = "#C49C94",  # 棕
    "CER-7"            = "#C5B0D5",  # 紫灰
    "CER-8"            = "#FCCDE5",  # 淡粉
    "CER-10"           = "#FF9896",  # 粉红
    "CER-12"           = "#B15928",  # 深棕
    "CER-13"           = "#FB8072",  # 柔红
    "CER-17"           = "#D95F02"   # 暖橙红
  ))+ theme_minimal(base_size = 12) +
  theme(
    panel.background = element_blank(),     # 去除面板背景
    plot.background = element_blank(),      # 去除整个图的背景
    legend.background = element_blank(),    # 去除图例背景
    panel.grid = element_blank(),           # 去除网格线
    axis.title = element_blank(),           # 不显示轴标题
    axis.text = element_blank(),            # 不显示轴标签
    axis.ticks = element_blank(),           # 不显示刻度线
    legend.title = element_text(size = 10),
    legend.text = element_text(size = 9),
    legend.key = element_blank()            # 去除图例项的灰色方块背景
  )






#Create data which can be used for Sankey
links <- read_excel("Human lymph node/D1/flow-network.xlsx")
# From these flows we need to create a node data frame: it lists every entities involved in the flow
nodes <- data.frame(name = unique(c(links$source, links$target)))
nodes <- na.omit(nodes)

# With networkD3, connection must be provided using id, not using real name like in the links dataframe. So we need to reformat it.
# 创建映射表
links$source <- match(links$source, nodes$name)-1
links$target <- match(links$target, nodes$name)-1
links <- na.omit(links)


# Plot
# 假设你已经有 links 和 nodes 数据
p <- sankeyNetwork(Links = links, Nodes = nodes,
                   Source = "source", Target = "target", Value = "weight",
                   NodeID = "name", fontSize = 12, nodeWidth = 20)

# 保存为临时 HTML
saveWidget(p, "Human lymph node/D1/sankey_temp.html", selfcontained = TRUE)


d <- read_excel("Human lymph node/D1/inflow-CXCL12-long.xlsx")

# 转为桑基图长格式
df <- d %>%
  make_long(inflow, CER, outflow)



ggplot(df, aes(x = x, next_x = next_x, node = node, next_node = next_node, fill = factor(node), label = node)) +
  geom_sankey(flow.alpha = .4,
              node.color = "gray30") +
  geom_sankey_label(size = 3.2, color = "black", hjust = 0) +
  # scale_fill_brewer(palette = "Set3", drop = FALSE) +
  # scale_fill_viridis_d(option = "I", drop = FALSE)+
  scale_fill_manual(values = c(
    "CXCL12" = "#1F77B4", "CER-2" = "#FF7F0E", "TGFB1" = "#2CA02C",
    "CSF1" = "#D62728", "LGALS9" = "#9467BD", "CXCL13" = "#8C564B",
    "CXCL12" = "#E377C2", "TNFSF10" = "#7F7F7F", "VEGFB" = "#BCBD22",
    "TNFSF13" = "#AEC7E8", "CER-5" = "#98DF8A", "CCL19" = "#FF9896",
    "CER-9" = "#C5B0D5", "IGF1" = "#C49C94"
  ))+ theme_minimal(base_size = 12) +
  theme(
    panel.background = element_blank(),     # 去除面板背景
    plot.background = element_blank(),      # 去除整个图的背景
    legend.background = element_blank(),    # 去除图例背景
    panel.grid = element_blank(),           # 去除网格线
    axis.title = element_blank(),           # 不显示轴标题
    axis.text = element_blank(),            # 不显示轴标签
    axis.ticks = element_blank(),           # 不显示刻度线
    legend.title = element_text(size = 10),
    legend.text = element_text(size = 9),
    legend.key = element_blank()            # 去除图例项的灰色方块背景
  )







d <- read_excel("Human lymph node/D1/CXCL12-long.xlsx")

# 转为桑基图长格式
df <- d %>%
  make_long(inflow, CER, outflow)

ggplot(df, aes(x = x, next_x = next_x, node = node, next_node = next_node, fill = factor(node), label = node)) +
  geom_sankey(flow.alpha = .4,
              node.color = "gray30") +
  geom_sankey_label(size = 3.2, color = "black", hjust = 0) +
  # scale_fill_brewer(palette = "Set3", drop = FALSE) +
  # scale_fill_viridis_d(option = "I", drop = FALSE)+
  scale_fill_manual(values = c(
    "MIF" = "#1F77B4", "CER-3" = "#FF7F0E", "CXCL12" = "#2CA02C",
    "VEGFC" = "#D62728", "CER-2" = "#9467BD", "CXCL12" = "#8C564B",
    "IL7" = "#E377C2", "CER-1" = "#7F7F7F", "TGFB1" = "#BCBD22",
    "TNFSF12" = "#AEC7E8", "IL34" = "#98DF8A", "TNFSF13" = "#FF9896",
    "CXCL13" = "#C5B0D5", "TNFSF10" = "#C49C94", "TNFSF13B" = "#D48265",
    "CCL21" = "#C7C7C7", "CER-4" = "#1F77B4", "CCL19" = "#FF7F0E",
    "PDGFB" = "#2CA02C"
  ))+ theme_minimal(base_size = 12) +
  theme(
    panel.background = element_blank(),     # 去除面板背景
    plot.background = element_blank(),      # 去除整个图的背景
    legend.background = element_blank(),    # 去除图例背景
    panel.grid = element_blank(),           # 去除网格线
    axis.title = element_blank(),           # 不显示轴标题
    axis.text = element_blank(),            # 不显示轴标签
    axis.ticks = element_blank(),           # 不显示刻度线
    legend.title = element_text(size = 10),
    legend.text = element_text(size = 9),
    legend.key = element_blank()            # 去除图例项的灰色方块背景
  )

