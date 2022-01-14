setwd("~/Github/AsteroidsClassification")

library(rpart)
library(rattle)
library(rpart.plot)
library(RColorBrewer)
library(plyr)
library(gridExtra)
library(neuralnet)
library(e1071)
library(caret)
library(tidyverse)
library(ROCR)

opt.cut = function(perf, pred){
  cut.ind = mapply(FUN=function(x, y, p){
    d = (x - 0)^2 + (y-1)^2
    ind = which(d == min(d))
    c(sensitivity = y[[ind]], specificity = 1-x[[ind]], 
      cutoff = p[[ind]])
  }, perf@x.values, perf@y.values, pred@cutoffs)
}


        
load("DATA_asteroids_dataset_split_0.7.RData")

asteroids_split$train$Hazardous.int = as.integer(asteroids_split$train$Hazardous)
asteroids_split$test$Hazardous.int = as.integer(asteroids_split$test$Hazardous)

asteroids_split$train = asteroids_split$train[1:1000,]
asteroids_split$test = asteroids_split$test[1:400,]
#DT
dt.Hazardous.model<- rpart(Hazardous.int ~ Orbit.Axis..AU. + Orbit.Eccentricity + Orbit.Inclination..deg. + Perihelion.Argument..deg. + Node.Longitude..deg. + Mean.Anomoly..deg. + Perihelion.Distance..AU. + Aphelion.Distance..AU. + Orbital.Period..yr. + Minimum.Orbit.Intersection.Distance..AU. + Asteroid.Magnitude,#Orbit.Axis..AU. + Orbit.Eccentricity + Orbit.Inclination..deg. + Perihelion.Argument..deg. + Node.Longitude..deg. + Mean.Anomoly..deg. + Perihelion.Distance..AU. + Aphelion.Distance..AU. + Orbital.Period..yr. + Minimum.Orbit.Intersection.Distance..AU. + Asteroid.Magnitude,
  data=asteroids_split$train, method="class", cp= 0.001,  prob=TRUE) #. all var
dt.Hazardous.pred <- predict(dt.Hazardous.model, asteroids_split$test, type = "class",  probability=TRUE)

# SVM
svm.Hazardous.model = svm(Hazardous ~ Orbit.Axis..AU. + Orbit.Eccentricity + Orbit.Inclination..deg. + Perihelion.Argument..deg. + Node.Longitude..deg. + Mean.Anomoly..deg. + Perihelion.Distance..AU. + Aphelion.Distance..AU. + Orbital.Period..yr. + Minimum.Orbit.Intersection.Distance..AU. + Asteroid.Magnitude,
  data=asteroids_split$train, kernel='linear', cost=1, type="C-classification", prob=TRUE)
model = svm.Hazardous.model

testset = asteroids_split$test
classLabel = "TRUE"
testLabels = testset$Hazardous
ROCFunction.BIN <- function(model,testset,testLabels,classLabel){
  ROCFun.pred = predict(model, testset,  probability=TRUE)
  ROCFun.pred.prob = attr(ROCFun.pred, "probabilities") 
  ROCFun.pred.class = colnames(ROCFun.pred.prob)
  ROCFun.pred.classindex = which(ROCFun.pred.class == classLabel)
  ROCFun.pred.to.roc = ROCFun.pred.prob[,ROCFun.pred.classindex]
  ROCFun.pred.rocr = predictions(ROCFun.pred.to.roc, testLabels) 
  ROCFun.perf.rocr = performance(ROCFun.pred.rocr, measure = "auc", x.measure = "cutoff")
  ROCFun.perf.tpr.rocr = performance(ROCFun.pred.rocr, "tpr","fpr")
}




plot(ROCFun.perf.tpr.rocr, colorize=T,main=paste("AUC:",(ROCFun.perf.rocr@y.values)))
abline(a=0, b=1)

print(opt.cut(svm.Hazardous.tpr.rocr, svm.Hazardous.pred.rocr))

#RNN
rnn.Hazardous.model = neuralnet(Hazardous ~ Orbit.Axis..AU. + Orbit.Eccentricity + Orbit.Inclination..deg. + Perihelion.Argument..deg. + Node.Longitude..deg. + Mean.Anomoly..deg. + Perihelion.Distance..AU. + Aphelion.Distance..AU. + Orbital.Period..yr. + Minimum.Orbit.Intersection.Distance..AU. + Asteroid.Magnitude,
  asteroids_split$train, hidden =c(7), act.fct = 'tanh', learningrate = 1e-2, stepmax = 1e7, linear.output = FALSE, prob=TRUE)
rnn.Hazardous.pred = predict(rnn.Hazardous.model, asteroids_split$test, probability=TRUE)[, 1] > 0.5



