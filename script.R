setwd("/home/zach/sync_docs/ky")
library(readxl)
library("tidyverse")
fname <- "Backward SC 2020-04-13 Tiers 1,2,3,4,5 Supplier Only In one Column.xlsx" #nolint
t <- read_excel(fname, sheet = "Tier 1-5")
t <- filter(t,t$Tier <= 4)
library(igraph)
g <- graph_from_data_frame(t[1:2])

x <- components(g, "weak")
h <- induced_subgraph(g, x$membership == which.max(x$csize))

fracturing <- function(g) {
  n <- 100
  nodecount <- vcount(g)
  p <- 1:n / (2 * n)
  for (i in seq_len(n)) {
    x[i] <- components(induced_subgraph(g, runif(nodecount) > p[i]))$no
  }
  x <- list("p"=p,"x"=x)
}
x <- fracturing(g)


tt <- filter(t,t$"Source Country" !="China" & t$"Target Country" != "China"
             | t$"Source Country" =="China" & t$"Target Country" == "China")
g1 <- graph_from_data_frame(tt[1:2])

x1 <- fracturing(g1)

plot(x$p, x$x,
     xlab="Firm failure rate",
     ylab="Graph component count",
     ylim=c(0,1.3*max(max(unlist(x1$x)),max(unlist(x$x))))
)
points(x1$p,unlist(x1$x) + vcount(g) - vcount(g1),col=2)
#points(x1$p,unlist(x1$x),col=3)
legend("bottomright",c("Global","China separated"),fill=c('black','red'))
print(vcount(g)-vcount(g1))