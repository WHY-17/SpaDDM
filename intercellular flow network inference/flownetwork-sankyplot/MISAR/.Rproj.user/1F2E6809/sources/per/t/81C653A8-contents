library(ggsankey)
library(tidyverse)
library(readxl)
library(networkD3)
library(htmlwidgets)

#Create data which can be used for Sankey
links <- read_excel("MISAR/networkx.xlsx")
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
                   Source = "source", Target = "target", Value = "value",
                   NodeID = "name", fontSize = 15, nodeWidth = 20)

# 保存为临时 HTML
saveWidget(p, "sankey_temp.html", selfcontained = TRUE)


# # 截图为 PNG，设置分辨率为 300dpi（即高分辨率）
# webshot("sankey_temp.html", file = "sankey_output.png",
#         vwidth = 2400, vheight = 1800, zoom = 2)

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



d <- read_excel("MISAR/inflow-Wnt6-long.xlsx")

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
    "Wnt6" = "#1F77B4", "CER-4" = "#FF7F0E", "Wnt2" = "#2CA02C",
    "Postn" = "#D62728", "Penk" = "#9467BD", "Igf1" = "#8C564B",
    "Ccl1" = "#E377C2", "Igf2" = "#7F7F7F", "CER-19" = "#BCBD22",
    "Wnt6" = "#17BECF", # 假设这是第二个Wnt6，用不同颜色表示
    "Plg" = "#AEC7E8", "Agt" = "#FFBB78", "Wnt7b" = "#98DF8A",
    "CER-12" = "#FF9896", "Nampt" = "#C5B0D5", "CER-17" = "#C49C94",
    "Tgfb1" = "#D48265", # 与Igf1不同颜色以示区分
    "Tgfb3" = "#C7C7C7"
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





# Alluvial plots are very similiar to sankey plots but have no spaces between nodes and start at y = 0 instead being centered around the x-axis.
# p3 <- ggplot(df, aes(x = x, next_x = next_x, node = node, next_node = next_node, fill = factor(node), label = node)) +
#   geom_alluvial(flow.alpha = 0.6, width = 0.3) +
#   geom_alluvial_text(size = 3, color = "white") +
#   scale_fill_viridis_d(drop = FALSE) +
#   theme_alluvial(base_size = 16) +
#   labs(x = NULL) +
#   theme(legend.position = "none",
#         plot.title = element_text(hjust = .5))
# # ggtitle("Sankey diagram using ggsankey")
# ggsave('p3.pdf', p3, width = 8, height = 6, dpi=300)


d <- read_excel("MISAR/Wnt6-long.xlsx")

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
    "Igf1"     = "#8C564B",  # 用 Igf1 的棕色
    "CER-11"          = "#C49C94",  # 类似 GEM-17 的颜色
    "Wnt6"            = "#17BECF",  # 保留原来的 Wnt6 色
    "Sema3d"   = "#7F7F7F",  # 类似 Igf2 的灰色
    "CER-1"           = "#BCBD22",  # 和 GEM-19 相近
    "Bmp6"     = "#AEC7E8",  # 使用 Plg 的蓝色
    "Bmp7"     = "#FFBB78",  # 使用 Agt 的橘色
    "Bmp4"     = "#FF9896",  # 使用 GEM-12 的红粉色
    "CER-19"          = "#BCBD22",  # 原有色保留
    "Wnt5a"    = "#2CA02C",  # 使用 Wnt2 的绿色
    "Wnt6"     = "#1F77B4"   # 和 inflow-Wnt6 保持一致
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

# fill = "gray80"
