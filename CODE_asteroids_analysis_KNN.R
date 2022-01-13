setwd("~/Github/AsteroidsClassification")
install.packages('fpc')
install.packages('cluster')
install.packages('seriation')

library(cluster)

#load dataset RObject as asteroids_data
load("DATA_asteroids_dataset.RData")

numeric_covariates = asteroids_data[,c("Orbit.Axis..AU." , "Orbit.Eccentricity" , "Orbit.Inclination..deg." , "Perihelion.Argument..deg.",
                                       "Node.Longitude..deg." , "Mean.Anomoly..deg." , "Perihelion.Distance..AU." , "Aphelion.Distance..AU.",
                                       "Orbital.Period..yr.", "Minimum.Orbit.Intersection.Distance..AU.",  "Asteroid.Magnitude"
                                      )]

set.seed(22)


fit = kmeans(numeric_covariates, 4)
fit

fit$cluster


kms = silhouette(fit$cluster,dist(numeric_covariates))
kms
plot(kms)

silhouette_score <- function(k){
  km <- kmeans(df, centers = k, nstart=25)
  ss <- silhouette(km$cluster, dist(df))
  mean(ss[, 3])
}
k <- 2:10
avg_sil <- sapply(k, silhouette_score)
plot(k, type='b', avg_sil, xlab='Number of clusters', ylab='Average Silhouette Scores', frame=FALSE)



### dissimilarity clusters


library('seriation')

dissplot(dist(numeric_covariates), labels=fit$cluster, options=list(main='Kmeans Clustering with K=4'))


### fine tuninig
fit$cluster


### cluster evaluation

cluser.evaluation(ground_truth, fit$cluser)
         
         
### varying of the silhouette score at the number of clusters

library(fpc)
nk = 2:4
set.seed(22)
Sw < sapply(nk, function(k) { cluster.stats(dist(numeric_covariates),kmeans(numeric_covariates, centers=k)$cluster)$avg.silwidth})
SW
plot(SW, type='I', xlab='number of clusters', ylab='average silhoutte')