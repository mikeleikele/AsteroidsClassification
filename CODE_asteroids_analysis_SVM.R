setwd("~/Github/AsteroidsClassification")

library(plyr)
library(gridExtra)
library(e1071)

#load dataset RObject as asteroids_split
load("DATA_asteroids_dataset_split_0.7.RData")

asteroids_split$train$Hazardous.int = as.integer(asteroids_split$train$Hazardous)
i = 1
folds.number = 10

folds <- split(asteroids_split$train, cut(sample(1:nrow(asteroids_split$train)), folds.number))
rm(folds.number)

svm.Hazardous.Linear.C1.stats <- NULL
svm.Hazardous.Linear.C1000.stats <- NULL
svm.Hazardous.Linear.C05.stats <- NULL

svm.Hazardous.Radial.C1.stats <- NULL
svm.Hazardous.Radial.C1000.stats <- NULL
svm.Hazardous.Radial.C05.stats <- NULL


for (i in 1:length(folds)) {
  fold.valid <- ldply(folds[i], data.frame)
  fold.valid <- fold.valid[1:10 , !names(fold.valid) %in% c(".id")]
  fold.train <- ldply(folds[-i], data.frame)
  fold.train <- fold.train[1:10, !names(fold.train) %in% c(".id")]
  print(i)
  #svm.Hazardous.Linear.CTuned = tune.svm(Hazardous.int ~ Orbit.Axis..AU. + Orbit.Eccentricity + Orbit.Inclination..deg. + Perihelion.Argument..deg. + Node.Longitude..deg. + Mean.Anomoly..deg. + Perihelion.Distance..AU. + Aphelion.Distance..AU. + Orbital.Period..yr. + Minimum.Orbit.Intersection.Distance..AU. + Asteroid.Magnitude,
  #                 data = fold.train[1:100,],
  #                 kernel='linear', 
  #                 cost=c(1,5))
  hyper.cost <- 1
  
  print(paste(as.character(i), hyper.kernel , as.character(hyper.cost), sep = " "))
  svm.Hazardous.Linear.C1 = svm(Hazardous.int ~ Orbit.Axis..AU. + Orbit.Eccentricity + Orbit.Inclination..deg. + Perihelion.Argument..deg. + Node.Longitude..deg. + Mean.Anomoly..deg. + Perihelion.Distance..AU. + Aphelion.Distance..AU. + Orbital.Period..yr. + Minimum.Orbit.Intersection.Distance..AU. + Asteroid.Magnitude,
    data=fold.train, kernel=hyper.kernel, cost=hyper.cost)
  svm.Hazardous.Linear.C1.pred = predict(svm.Hazardous.Linear.C1, fold.valid)
  svm.Hazardous.Linear.C1.confusion_matrix = table(fold.valid$Hazardous, svm.Hazardous.Linear.C1.pred)
  
  hyper.cost <- 1000
  print(paste(as.character(i), hyper.kernel , as.character(hyper.cost), sep = " "))
  svm.Hazardous.Linear.C1000 = svm(Hazardous.int ~ Orbit.Axis..AU. + Orbit.Eccentricity + Orbit.Inclination..deg. + Perihelion.Argument..deg. + Node.Longitude..deg. + Mean.Anomoly..deg. + Perihelion.Distance..AU. + Aphelion.Distance..AU. + Orbital.Period..yr. + Minimum.Orbit.Intersection.Distance..AU. + Asteroid.Magnitude,
    data=fold.train, kernel=hyper.kernel, cost=hyper.cost)
  svm.Hazardous.Linear.C1000.pred = predict(svm.Hazardous.Linear.C1000, fold.valid)
  svm.Hazardous.Linear.C1000.confusion_matrix = table(fold.valid$Hazardous, svm.Hazardous.Linear.C1000.pred)
  
  hyper.cost <- 0.5
  print(paste(as.character(i), hyper.kernel , as.character(hyper.cost), sep = " "))
  svm.Hazardous.Linear.C05 = svm(Hazardous.int ~ Orbit.Axis..AU. + Orbit.Eccentricity + Orbit.Inclination..deg. + Perihelion.Argument..deg. + Node.Longitude..deg. + Mean.Anomoly..deg. + Perihelion.Distance..AU. + Aphelion.Distance..AU. + Orbital.Period..yr. + Minimum.Orbit.Intersection.Distance..AU. + Asteroid.Magnitude,
                                data=fold.train, kernel=hyper.kernel, cost=hyper.cost)
  svm.Hazardous.Linear.C05.pred = predict(svm.Hazardous.Linear.C05, fold.valid)
  svm.Hazardous.Linear.C05.confusion_matrix = table(fold.valid$Hazardous, svm.Hazardous.Linear.C05.pred)
  
  if (is.null(svm.Hazardous.Linear.C1.stats)){
    svm.Hazardous.Linear.C05.stats = svm.Hazardous.Linear.C05.confusion_matrix
  }else{
    svm.Hazardous.Linear.C05.stats <- svm.Hazardous.Linear.C05.stats + svm.Hazardous.Linear.C05.confusion_matrix
  }
  
  if (is.null(svm.Hazardous.Linear.C1.stats)){
    svm.Hazardous.Linear.C1.stats = svm.Hazardous.Linear.C1.confusion_matrix
  }else{
    svm.Hazardous.Linear.C1.stats <- svm.Hazardous.Linear.C1.stats + svm.Hazardous.Linear.C1.confusion_matrix
  }
  
  if (is.null(svm.Hazardous.Linear.C1000.stats)){
    svm.Hazardous.Linear.C1000.stats = svm.Hazardous.Linear.C1000.confusion_matrix
  }else{
    svm.Hazardous.Linear.C1000.stats <- svm.Hazardous.Linear.C1000.stats + svm.Hazardous.Linear.C1000.confusion_matrix
  }
  rm(fold.train, fold.valid)
  rm(svm.Hazardous.Linear.C05.pred,svm.Hazardous.Linear.C1.pred,svm.Hazardous.Linear.C1000.pred)
  rm(svm.Hazardous.Linear.C1.confusion_matrix,svm.Hazardous.Linear.C1000.confusion_matrix,svm.Hazardous.Linear.C05.confusion_matrix)
}

svm.Hazardous.Linear.C05.stats <- svm.Hazardous.Linear.C05.stats / length(folds)
svm.Hazardous.Linear.C1.stats <- svm.Hazardous.Linear.C1.stats / length(folds)
svm.Hazardous.Linear.C1000.stats <- svm.Hazardous.Linear.C1000.stats / length(folds)

img_name_plot <- paste("IMG_asteroids_model_SVM_", "Hazardous_Confusion_", hyper.kernel,"_05" ,".png", sep = "")
  png(img_name_plot)
  grid.table(svm.Hazardous.Linear.C05.stats)
  dev.off()
img_name_plot <- paste("IMG_asteroids_model_SVM_", "Hazardous_Confusion_", hyper.kernel,"_1" ,".png", sep = "")
  png(img_name_plot)
  grid.table(svm.Hazardous.Linear.C1.stats)
  dev.off()
img_name_plot <- paste("IMG_asteroids_model_SVM_", "Hazardous_Confusion_", hyper.kernel,"_1000" ,".png", sep = "")
  png(img_name_plot)
  grid.table(svm.Hazardous.Linear.C1000.stats)
  dev.off()
rm(hyper.cost, hyper.cost, img_name_plot)
rm(svm.Hazardous.Linear.C05.stats,svm.Hazardous.Linear.C1.stats,svm.Hazardous.Linear.C1000.stats)
rm(svm.Hazardous.Linear.C05,svm.Hazardous.Linear.C1,svm.Hazardous.Linear.C1000)
#Hazardous class
#a small cost creates a large margin (a soft margin) and allows more misclassifications;
#a large cost creates a narrow margin (a hard margin) and allows few misclassifications
#this is a soft margin svm



print(svm..Hazardous.Linear.C1)

svm.table=table(svm.pred, testset$Species)





#Classification type asteroids

#Amor Asteroid - Apollo Asteroid hyperpiano

astroids.subset = subset(asteroids_split$train, select=c("Perihelion.Distance..AU.", "Orbit.Axis..AU.", "Classification"),
  Classification %in% c("Amor Asteroid","Apollo Asteroid"))
astroids.subset.plt = astroids.subset[1:1000,]

#Linear cost 1
svm.Classification.Plot.Linear.C1 = svm(Classification ~ ., data=astroids.subset, 
  kernel=hyper.kernel, cost=hyper.cost, scale=TRUE)
img_name_plot <- paste("IMG_asteroids_model_SVM_","Classification_",hyper.kernel,"_C",as.character(hyper.cost),"_Amor_Apollo_support", ".png", sep = "")
png(img_name_plot)
  plot(x=astroids.subset.plt$Perihelion.Distance..AU.,y=astroids.subset.plt$Orbit.Axis..AU.,  col=astroids.subset.plt$Classification, pch=19)
  points(astroids.subset.plt[svm.Classification.Plot.Linear.C1$index,c(1,2)],col="blue",cex=2)
  w = t(svm.Classification.Plot.Linear.C1$coefs) %*% svm.Classification.Plot.Linear.C1$SV
  c = -svm.Classification.Plot.Linear.C1$rho
  abline (a=-c/w[1,2],b=-w[1,1]/w[1,2], col="red", lty=5)
  abline(a=(-c-1)/w[1,2], b=-w[1,1]/w[1,2], col="orange", lty=3)
  abline(a=(-c+1)/w[1,2], b=-w[1,1]/w[1,2], col="orange", lty=3)
  dev.off()
rm(w, c)
rm(hyper.cost,hyper.kernel,svm.Classification.Plot.Linear.C1)
#Linear cost 1.000
hyper.cost <- 1000
hyper.kernel <- 'linear'
svm.Classification.Plot.Linear.C1 = svm(Classification ~ ., data=astroids.subset, 
  kernel=hyper.kernel, cost=hyper.cost, scale=FALSE)
img_name_plot <- paste("IMG_asteroids_model_SVM_","Classification_",hyper.kernel,"_C",as.character(hyper.cost),"_Amor_Apollo_support", ".png", sep = "")
png(img_name_plot)
  plot(x=astroids.subset.plt$Perihelion.Distance..AU.,y=astroids.subset.plt$Orbit.Axis..AU.,  col=astroids.subset.plt$Classification, pch=19)
  points(astroids.subset.plt[svm.Classification.Plot.Linear.C1$index,c(1,2)],col="blue",cex=2)
  w = t(svm.Classification.Plot.Linear.C1$coefs) %*% svm.Classification.Plot.Linear.C1$SV
  c = -svm.Classification.Plot.Linear.C1$rho
  abline (a=-c/w[1,2],b=-w[1,1]/w[1,2], col="red", lty=5)
  abline(a=(-c-1)/w[1,2], b=-w[1,1]/w[1,2], col="orange", lty=3)
  abline(a=(-c+1)/w[1,2], b=-w[1,1]/w[1,2], col="orange", lty=3)
  dev.off()
rm(w, c)
rm(hyper.cost,hyper.kernel,svm.Classification.Plot.Linear.C1)
#Linear cost 1.000
hyper.cost <- 10000
hyper.kernel <- 'linear'
svm.Classification.Plot.Linear.C1 = svm(Classification ~ ., data=astroids.subset, 
                                        kernel=hyper.kernel, cost=hyper.cost, scale=FALSE)
img_name_plot <- paste("IMG_asteroids_model_SVM_","Classification_",hyper.kernel,"_C",as.character(hyper.cost),"_Amor_Apollo_support", ".png", sep = "")
png(img_name_plot)
  plot(x=astroids.subset.plt$Perihelion.Distance..AU.,y=astroids.subset.plt$Orbit.Axis..AU.,  col=astroids.subset.plt$Classification, pch=19)
  points(astroids.subset.plt[svm.Classification.Plot.Linear.C1$index,c(1,2)],col="blue",cex=2)
  w = t(svm.Classification.Plot.Linear.C1$coefs) %*% svm.Classification.Plot.Linear.C1$SV
  c = -svm.Classification.Plot.Linear.C1$rho
  abline (a=-c/w[1,2],b=-w[1,1]/w[1,2], col="red", lty=5)
  abline(a=(-c-1)/w[1,2], b=-w[1,1]/w[1,2], col="orange", lty=3)
  abline(a=(-c+1)/w[1,2], b=-w[1,1]/w[1,2], col="orange", lty=3)
  dev.off()
rm(w, c)
rm(hyper.cost,hyper.kernel,svm.Classification.Plot.Linear.C1)
#Linear cost 0.5
hyper.cost <- 0.5
hyper.kernel <- 'linear'
svm.Classification.Plot.Linear.C1 = svm(Classification ~ ., data=astroids.subset, 
                                        kernel=hyper.kernel, cost=hyper.cost, scale=FALSE)
img_name_plot <- paste("IMG_asteroids_model_SVM_","Classification_",hyper.kernel,"_C",as.character(hyper.cost),"_Amor_Apollo_support", ".png", sep = "")
png(img_name_plot)
  plot(x=astroids.subset.plt$Perihelion.Distance..AU.,y=astroids.subset.plt$Orbit.Axis..AU.,  col=astroids.subset.plt$Classification, pch=19)
  points(astroids.subset.plt[svm.Classification.Plot.Linear.C1$index,c(1,2)],col="blue",cex=2)
  w = t(svm.Classification.Plot.Linear.C1$coefs) %*% svm.Classification.Plot.Linear.C1$SV
  c = -svm.Classification.Plot.Linear.C1$rho
  abline (a=-c/w[1,2],b=-w[1,1]/w[1,2], col="red", lty=5)
  abline(a=(-c-1)/w[1,2], b=-w[1,1]/w[1,2], col="orange", lty=3)
  abline(a=(-c+1)/w[1,2], b=-w[1,1]/w[1,2], col="orange", lty=3)
  dev.off()
rm(w, c)
rm(hyper.cost,hyper.kernel,svm.Classification.Plot.Linear.C1)

#Poly  cost 1 
svm.Classification.Plot.Linear.C1 = svm(Classification ~ ., data=astroids.subset, 
                                        kernel=hyper.kernel, cost=hyper.cost, scale=TRUE)
img_name_plot <- paste("IMG_asteroids_model_SVM_","Classification_",hyper.kernel,"_C",as.character(hyper.cost),"_Amor_Apollo_support", ".png", sep = "")
png(img_name_plot)
plot(x=astroids.subset.plt$Perihelion.Distance..AU.,y=astroids.subset.plt$Orbit.Axis..AU.,  col=astroids.subset.plt$Classification, pch=19)
points(astroids.subset.plt[svm.Classification.Plot.Linear.C1$index,c(1,2)],col="blue",cex=2)
w = t(svm.Classification.Plot.Linear.C1$coefs) %*% svm.Classification.Plot.Linear.C1$SV
c = -svm.Classification.Plot.Linear.C1$rho
abline (a=-c/w[1,2],b=-w[1,1]/w[1,2], col="red", lty=5)
abline(a=(-c-1)/w[1,2], b=-w[1,1]/w[1,2], col="orange", lty=3)
abline(a=(-c+1)/w[1,2], b=-w[1,1]/w[1,2], col="orange", lty=3)
dev.off()
rm(w, c)
rm(hyper.cost,hyper.kernel,svm.Classification.Plot.Linear.C1)

#SMV CLASSIFICATION

hyper.cost <- 1
hyper.kernel <- 'linear'
svm.Classification.Linear.C1 = svm(Classification ~ Orbit.Axis..AU. + Orbit.Eccentricity + Orbit.Inclination..deg. + Perihelion.Argument..deg. + Node.Longitude..deg. + Mean.Anomoly..deg. + Perihelion.Distance..AU. + Aphelion.Distance..AU. + Orbital.Period..yr. + Minimum.Orbit.Intersection.Distance..AU. + Asteroid.Magnitude,
                                   data=asteroids_split$train, kernel='linear', cost=1)
print(svm.Classification.Linear.C1)
svm.Classification.Linear.C1.pred = predict(svm.Classification.Linear.C1, asteroids_split$test)
svm.table=table(svm.pred, testset$Species)