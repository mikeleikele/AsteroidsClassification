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

### barplot cluster frequency

img_name_plot <- paste("IMG_asteroids_model_KNN_", "barplot_k_4", ".png", sep = "")
png(img_name_plot,res = 800, height = 10, width = 15, unit='in')

barplot(table(fit$cluster), main = "Frequency by cluster",
        col = rainbow(4))

dev.off()

img_name_plot <- paste("IMG_asteroids_model_KNN_", "barplot_k_6", ".png", sep = "")
png(img_name_plot, res = 800, height = 10, width = 15, unit='in')


fit6 = kmeans(numeric_covariates, 6)
fit6


barplot(table(fit6$cluster), main = "Frequency by cluster",
        col = rainbow(6))

dev.off()


### barplot Classification class

barplot(table(asteroids_data$Classification), main= "Frequency Asteroids Classes", col=rainbow(4))

kms = silhouette(fit$cluster,dist(numeric_covariates))
kms
plot(kms)

img_name_plot <- paste("IMG_asteroids_model_KNN_", "silouhette_score_per_k", ".png", sep = "")
png(img_name_plot,res = 800, height = 10, width = 15, unit='in')

silhouette_score <- function(k){
  km <- kmeans(numeric_covariates, centers = k, nstart=25)
  ss <- silhouette(km$cluster, dist(numeric_covariates))
  mean(ss[, 3])
}
k <- 2:10
avg_sil <- sapply(k, silhouette_score)
plot(k, type='b', avg_sil, xlab='Number of clusters', ylab='Average Silhouette Scores', frame=FALSE)

dev.off()


### dissimilarity clusters


library('seriation')

dissplot(dist(numeric_covariates), labels=fit6$cluster, options=list(main='Kmeans Clustering with K=6'))


### fine tuninig
fit$cluster


### cluster evaluation

cluser.evaluation(ground_truth, fit$cluser)
         
         
### measuring the silhouette score at the number of clusters

library(fpc)
nk = 2:4
set.seed(22)


Sw < sapply(nk, function(k) { cluster.stats(dist(numeric_covariates),kmeans(numeric_covariates, centers=k)$cluster)$avg.silwidth})
SW
plot(SW, type='I', xlab='number of clusters', ylab='average silhoutte')