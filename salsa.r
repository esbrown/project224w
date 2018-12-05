library(igraph)
library(centiserve)
library(linkcomm)
data = read.csv(file.choose(), header=FALSE)[,c(1:2)]
graph = graph_from_edgelist(as.matrix(data), directed=TRUE)

linscores = lincent(graph)
clusterrankscores = clusterrank(graph)
leveragescores = leverage(graph)

write.csv(leveragescores, file='leveragescores.csv')

plot(linscores, viewcounts$count)
print(linscores)

