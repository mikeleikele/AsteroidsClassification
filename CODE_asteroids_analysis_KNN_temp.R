setwd("~/Github/AsteroidsClassification")

library(plyr)
library(gridExtra)
library(neuralnet)
library(caret)
library(tidyverse)
#t-distribution
confidence_interval <- function(vector, interval) {
  # Standard deviation of sample
  vec_sd <- sd(vector)
  # Sample size
  n <- length(vector)
  # Mean of sample
  vec_mean <- mean(vector, na.rm = TRUE)
  # Error according to t distribution
  error <- qt((interval + 1)/2, df = n - 1) * vec_sd / sqrt(n)
  # Confidence interval as a vector
  result <- c("err" = error, "mean" = vec_mean)
  return(result)
}


#load dataset RObject as asteroids_split
load("DATA_asteroids_dataset_split_0.7.RData")

asteroids_split$train$Hazardous.int = as.factor(asteroids_split$train$Hazardous)

#logistic binary or sigm, tanh
#funz loss/obiettivo error quadr, cross entropy
#multiclass relu
network = neuralnet(Hazardous ~ Orbit.Axis..AU. + Orbit.Eccentricity + Orbit.Inclination..deg. + Perihelion.Argument..deg. + Node.Longitude..deg. + Mean.Anomoly..deg. + Perihelion.Distance..AU. + Aphelion.Distance..AU. + Orbital.Period..yr. + Minimum.Orbit.Intersection.Distance..AU. + Asteroid.Magnitude,
                    asteroids_split$train[1:500,], hidden=7,
                    linear.output = FALSE)

net.predict = predict(network, asteroids_split$test[1:100,])

table(asteroids_split$test[1:100,]$Hazardous, net.predict[, 1] > 0.5)


plot(network)


net.prediction = c("a","b","c")[apply(net.predict, 1, which.max)]
# Binary classification
nn <- neuralnet(Species == "setosa" ~ Petal.Length + Petal.Width, iris_train, linear.output = FALSE)
pred <- predict(nn, iris_test)
table(iris_test$Species == "setosa", pred[, 1] > 0.5)
# Multiclass classification
nn <- neuralnet((Species == "setosa") + (Species == "versicolor") + (Species == "virginica")
                ~ Petal.Length + Petal.Width, iris_train, linear.output = FALSE)
pred <- predict(nn, iris_test)
table(iris_test$Species, apply(pred, 1, which.max))



#pesi generalizzati
#quanto attributo spiega il target - rispetto alla rnn ottenuta
#0 spiegano poco, 1 molto
par(mfrow=c(2,2))
  gwplot(network,selected.covariate="Orbit.Axis..AU.")
  gwplot(network,selected.covariate="Orbit.Eccentricity")
  gwplot(network,selected.covariate="Orbit.Inclination..deg.")
  gwplot(network,selected.covariate="Perihelion.Argument..deg.")

par(mfrow=c(2,2))  
  gwplot(network,selected.covariate="Node.Longitude..deg.")
  gwplot(network,selected.covariate="Mean.Anomoly..deg.")
  gwplot(network,selected.covariate="Orbit.Inclination..deg.")
  gwplot(network,selected.covariate="Perihelion.Distance..AU.")

par(mfrow=c(2,2))  
  gwplot(network,selected.covariate="Aphelion.Distance..AU.")
  gwplot(network,selected.covariate="Orbital.Period..yr.")
  gwplot(network,selected.covariate="Orbit.Inclination..deg.")
  gwplot(network,selected.covariate="Minimum.Orbit.Intersection.Distance..AU.")
par(mfrow=c(2,2))
  gwplot(network,selected.covariate="Asteroid.Magnitude")
  