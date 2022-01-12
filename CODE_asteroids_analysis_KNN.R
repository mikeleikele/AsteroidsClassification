setwd("~/Github/AsteroidsClassification")
library(cluster)

#load dataset RObject as asteroids_data
load("DATA_asteroids_dataset.RData")


set.seed(22)
fit = kmeans(asteroids_data, 4)
fit


kms = silhouette(fit$cluster,dist(customer))
plot(kms)