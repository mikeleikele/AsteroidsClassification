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
library(gdata)
library(devtools)

ROCFunction.optcut = function(perf, pred){
  cut.ind = mapply(FUN=function(x, y, p){
    d = (x - 0)^2 + (y-1)^2
    ind = which(d == min(d))
    c(sensitivity = y[[ind]], 
      specificity = 1-x[[ind]], 
      cutoff = p[[ind]])
  }, perf@x.values, perf@y.values, pred@cutoffs)
}

Confusion_Sum <- function(cm_global, data, reference){
  cm_fold <- table(reference, data)
  if (is.null(cm_global)){
    cm_global <- cm_fold
  }else{
    cm_global <- cm_global + cm_fold
  }
  return(cm_global)
}

ROCFunction.BIN.RNN <- function(model,testset,testLabels,classLabel){
  ROCFun.pred.to.roc = unlist(ROCFun.pred.prob[,1], use.names=FALSE)
  ROCFun.pred.rocr = prediction(ROCFun.pred.to.roc, testLabels) 
  ROCFun.perf.rocr = performance(ROCFun.pred.rocr, measure = "auc", x.measure = "cutoff")
  ROCFun.perf.tpr.rocr = performance(ROCFun.pred.rocr, "tpr","fpr")
  ROCFun.perf.optcut = ROCFunction.optcut(ROCFun.perf.tpr.rocr, ROCFun.pred.rocr)
  ROCFun.perf.optcut = ROCFun.perf.optcut[[3]]
  return(c( 
    x.name = ROCFun.perf.tpr.rocr@x.name, x.value = ROCFun.perf.tpr.rocr@x.values,
    y.name = ROCFun.perf.tpr.rocr@y.name, y.value = ROCFun.perf.tpr.rocr@y.values,
    auc = ROCFun.perf.rocr@y.values, optcut = ROCFun.perf.optcut))
}

ROCFunction.BIN <- function(ROCFun.pred.prob, testLabels, classInLabel){
  #ROCFun.pred = predict(model, testset,  probability=TRUE)
  #ROCFun.pred.prob = attr(ROCFun.pred, "probabilities") 
  ROCFun.pred.class = colnames(ROCFun.pred.prob)
  ROCFun.pred.classindex = which(ROCFun.pred.class == classInLabel)
  ROCFun.pred.to.roc = unlist(ROCFun.pred.prob[,ROCFun.pred.classindex], use.names=FALSE)
  
  ROCFun.pred.rocr = prediction(ROCFun.pred.to.roc, testLabels) 
  ROCFun.perf.rocr = performance(ROCFun.pred.rocr, measure = "auc", x.measure = "cutoff")
  ROCFun.perf.tpr.rocr = performance(ROCFun.pred.rocr, "tpr","fpr")
  ROCFun.perf.optcut = ROCFunction.optcut(ROCFun.perf.tpr.rocr, ROCFun.pred.rocr)
  ROCFun.perf.optcut = ROCFun.perf.optcut[[3]]
  return(c( 
    x.name = ROCFun.perf.tpr.rocr@x.name, x.value = ROCFun.perf.tpr.rocr@x.values,
    y.name = ROCFun.perf.tpr.rocr@y.name, y.value = ROCFun.perf.tpr.rocr@y.values,
    auc = ROCFun.perf.rocr@y.values, optcut = ROCFun.perf.optcut))
}

ROCFunction.MULTI.RNN <- function(ROCFun.pred.prob, testLabels, classInLabel){
  #ROCFun.pred = predict(model, testset,  probability=TRUE)
  #ROCFun.pred.prob = attr(ROCFun.pred, "probabilities") 
  ROCFun.pred.class = colnames(ROCFun.pred.prob)
  ROCFun.pred.classindex = which(ROCFun.pred.class == classInLabel)
  ROCFun.pred.to.roc = as.vector(ROCFun.pred.prob[,ROCFun.pred.classindex])
  ROCFun.testClass = testLabels == classInLabel
  ROCFun.pred.rocr = prediction(ROCFun.pred.to.roc, ROCFun.testClass) 
  ROCFun.perf.rocr = performance(ROCFun.pred.rocr, measure = "auc", x.measure = "cutoff")
  ROCFun.perf.tpr.rocr = performance(ROCFun.pred.rocr, "tpr","fpr")
  ROCFun.perf.optcut = ROCFunction.optcut(ROCFun.perf.tpr.rocr, ROCFun.pred.rocr)
  ROCFun.perf.optcut = ROCFun.perf.optcut[[3]]
  return(c( 
    x.name = ROCFun.perf.tpr.rocr@x.name, x.value = ROCFun.perf.tpr.rocr@x.values,
    y.name = ROCFun.perf.tpr.rocr@y.name, y.value = ROCFun.perf.tpr.rocr@y.values,
    auc = ROCFun.perf.rocr@y.values, optcut = ROCFun.perf.optcut))
}

ROCFunction.MULTI <- function(ROCFun.pred.prob, testLabels, classInLabel){
  #ROCFun.pred = predict(model, testset,  probability=TRUE)
  #ROCFun.pred.prob = attr(ROCFun.pred, "probabilities") 
  ROCFun.pred.class = colnames(ROCFun.pred.prob)
  ROCFun.pred.classindex = which(ROCFun.pred.class == classInLabel)
  ROCFun.pred.to.roc = as.vector(ROCFun.pred.prob[,ROCFun.pred.classindex])
  ROCFun.testClass = testLabels == classInLabel
  ROCFun.pred.rocr = prediction(ROCFun.pred.to.roc, ROCFun.testClass) 
  ROCFun.perf.rocr = performance(ROCFun.pred.rocr, measure = "auc", x.measure = "cutoff")
  ROCFun.perf.tpr.rocr = performance(ROCFun.pred.rocr, "tpr","fpr")
  ROCFun.perf.optcut = ROCFunction.optcut(ROCFun.perf.tpr.rocr, ROCFun.pred.rocr)
  ROCFun.perf.optcut = ROCFun.perf.optcut[[3]]
  return(c( 
    x.name = ROCFun.perf.tpr.rocr@x.name, x.value = ROCFun.perf.tpr.rocr@x.values,
    y.name = ROCFun.perf.tpr.rocr@y.name, y.value = ROCFun.perf.tpr.rocr@y.values,
    auc = ROCFun.perf.rocr@y.values, optcut = ROCFun.perf.optcut))
}

load("DATA_asteroids_dataset_split_neg_0.7.RData")
asteroids_split$train$Hazardous.int = as.factor(asteroids_split$train$Hazardous)
asteroids_split$test$Hazardous.int = as.factor(asteroids_split$test$Hazardous)




#Classification

all.Classification <- list()
mx = matrix(NA, nrow = 3)

all.Classification$Accuracy = data.frame(mx)
all.Classification$MacroSensitivity <- data.frame(mx)
all.Classification$MacroSpecificity <- data.frame(mx)
all.Classification$MacroPrecision <- data.frame(mx)
all.Classification$MacroRecall <- data.frame(mx)
all.Classification$MacroF1 <- data.frame(mx)

#Amor
all.Classification$Amor.AUC <- data.frame(mx)
all.Classification$Amor.CutOffOpt <- data.frame(mx)
all.Classification_ROC.Amor.x <- matrix()
all.Classification_ROC.Amor.y <- matrix()
#Apohele
all.Classification$Apohele.AUC <- data.frame(mx)
all.Classification$Apohele.CutOffOpt <- data.frame(mx)
all.Classification_ROC.Apohele.x <- matrix()
all.Classification_ROC.Apohele.y <- matrix()
#Apollo
all.Classification$Apollo.AUC <- data.frame(mx)
all.Classification$Apollo.CutOffOpt <- data.frame(mx)
all.Classification_ROC.Apollo.x <- matrix()
all.Classification_ROC.Apollo.y <- matrix()
#Aten
all.Classification$Aten.AUC <- data.frame(mx)
all.Classification$Aten.CutOffOpt <- data.frame(mx)
all.Classification_ROC.Aten.x <- matrix()
all.Classification_ROC.Aten.y <- matrix()

all.Classification_ROC.name <- c(NA)

mx = matrix(NA, nrow = 6)
all.Classification.All <- data.frame(mx)
all.Classification.All["Performance"] = c("Accuracy","MacroSensitivity","MacroSpecificity","MacroPrecision","MacroRecall","MacroF1")
rm(mx)

mx = matrix(NA, nrow = 8)
all.Classification.ROC.All <- data.frame(mx)
all.Classification.ROC.All["Performance"] = c("Amor AUC","Amor CutOffOpt","Apohele AUC","Apohele CutOffOpt","Apollo AUC","Apollo CutOffOpt","Aten AUC","Aten CutOffOpt")
rm(mx)

#DT
dt.Classification.model <- rpart(Classification ~ Orbit.Axis..AU. + Orbit.Eccentricity + Orbit.Inclination..deg. + Perihelion.Argument..deg. + Node.Longitude..deg. + Mean.Anomoly..deg. + Perihelion.Distance..AU. + Aphelion.Distance..AU. + Orbital.Period..yr. + Minimum.Orbit.Intersection.Distance..AU. + Asteroid.Magnitude,#
                                 data=asteroids_split$train, method="class",  cp= 0.001)

dt.Classification.pred <- predict(dt.Classification.model, asteroids_split$test, type = "class")
dt.Classification.pred.prob <- predict(dt.Classification.model, asteroids_split$test, probability=TRUE)

dt.Classification.confusion_matrix_multiclass = confusionMatrix(
  data=dt.Classification.pred, reference=asteroids_split$test$Classification, mode = "prec_recall") 

folds_confusion <- NULL
folds_confusion <- Confusion_Sum(folds_confusion, reference=asteroids_split$test$Classification, data=dt.Classification.pred)
img_name_plot <- paste("IMG_asteroids_model_BEST_DT_Classification_confusion" ,".png", sep = "")
  png(img_name_plot)
  grid.table(folds_confusion)
  dev.off()

confusion.multi = dt.Classification.confusion_matrix_multiclass$byClass

sens_Amor = confusion.multi["Class: Amor Asteroid","Sensitivity"] 
spec_Amor = confusion.multi["Class: Amor Asteroid","Specificity"]
prec_Amor = confusion.multi["Class: Amor Asteroid","Precision"]
recal_Amor = confusion.multi["Class: Amor Asteroid","Recall"]
f1_Amor = confusion.multi["Class: Amor Asteroid","F1"]

sens_Apohele = confusion.multi["Class: Apohele Asteroid","Sensitivity"] 
spec_Apohele = confusion.multi["Class: Apohele Asteroid","Specificity"]
prec_Apohele = confusion.multi["Class: Apohele Asteroid","Precision"]
recal_Apohele = confusion.multi["Class: Apohele Asteroid","Recall"]
f1_Apohele = confusion.multi["Class: Apohele Asteroid","F1"]

sens_Apollo = confusion.multi["Class: Apollo Asteroid","Sensitivity"] 
spec_Apollo = confusion.multi["Class: Apollo Asteroid","Specificity"]
prec_Apollo = confusion.multi["Class: Apollo Asteroid","Precision"]
recal_Apollo = confusion.multi["Class: Apollo Asteroid","Recall"]
f1_Apollo = confusion.multi["Class: Apollo Asteroid","F1"]

sens_Aten = confusion.multi["Class: Aten Asteroid","Sensitivity"] 
spec_Aten = confusion.multi["Class: Aten Asteroid","Specificity"]
prec_Aten = confusion.multi["Class: Aten Asteroid","Precision"]
recal_Aten = confusion.multi["Class: Aten Asteroid","Recall"]
f1_Aten = confusion.multi["Class: Aten Asteroid","F1"]

Accuracy = dt.Classification.confusion_matrix_multiclass$overall["Accuracy"]
MacroSensitivity = (0.25 * sens_Amor) + (0.25 * sens_Apohele) + (0.25 * sens_Apollo) + (0.25 * sens_Aten)
MacroSpecificity = (0.25 * spec_Amor) + (0.25 * spec_Apohele) + (0.25 * spec_Apollo) + (0.25 * spec_Aten)
MacroPrecision = (0.25 * prec_Amor) + (0.25 * prec_Apohele) + (0.25 * prec_Apollo) + (0.25 * prec_Aten)
MacroRecall = (0.25 * recal_Amor) + (0.25 * recal_Apohele) + (0.25 * recal_Apollo) + (0.25 * recal_Aten)
MacroF1 = (0.25 * f1_Amor) + (0.25 * f1_Apohele) + (0.25 * f1_Apollo) + (0.25 * f1_Aten)
mod.name <- paste("Classification DT")
all.Classification$Accuracy[mod.name] <- Accuracy
all.Classification$MacroSensitivity[mod.name] <- MacroSensitivity
all.Classification$MacroSpecificity[mod.name] <- MacroSpecificity
all.Classification$MacroPrecision[mod.name] <- MacroPrecision
all.Classification$MacroRecall[mod.name] <- MacroRecall
all.Classification$MacroF1[mod.name] <- MacroF1

dt.Classification.roc.Amor = ROCFunction.MULTI(dt.Classification.pred.prob, as.factor(asteroids_split$test$Classification), "Amor Asteroid")
dt.Classification.roc.Apohele = ROCFunction.MULTI(dt.Classification.pred.prob, as.factor(asteroids_split$test$Classification), "Apohele Asteroid")
dt.Classification.roc.Apollo = ROCFunction.MULTI(dt.Classification.pred.prob, as.factor(asteroids_split$test$Classification), "Apollo Asteroid")
dt.Classification.roc.Aten = ROCFunction.MULTI(dt.Classification.pred.prob, as.factor(asteroids_split$test$Classification), "Aten Asteroid")

#roc
all.Classification$Amor.AUC[mod.name] <- dt.Classification.roc.Amor$auc
all.Classification$Amor.CutOffOpt[mod.name] <- dt.Classification.roc.Amor$optcut
all.Classification$Apohele.AUC[mod.name] <- dt.Classification.roc.Apohele$auc
all.Classification$Apohele.CutOffOpt[mod.name] <- dt.Classification.roc.Apohele$optcut
all.Classification$Apollo.AUC[mod.name] <- dt.Classification.roc.Apollo$auc
all.Classification$Apollo.CutOffOpt[mod.name] <- dt.Classification.roc.Apollo$optcut
all.Classification$Aten.AUC[mod.name] <- dt.Classification.roc.Aten$auc
all.Classification$Aten.CutOffOpt[mod.name] <- dt.Classification.roc.Aten$optcut

# ROC DATA FRAMES

all.Classification_ROC.name = c(all.Classification_ROC.name, mod.name)

all.Classification_ROC.Amor.x <- cbindX(all.Classification_ROC.Amor.x, data.frame(dt.Classification.roc.Amor$x.value))
colnames(all.Classification_ROC.Amor.x) <- all.Classification_ROC.name
all.Classification_ROC.Amor.y <- cbindX(all.Classification_ROC.Amor.y, data.frame(dt.Classification.roc.Amor$y.value))
colnames(all.Classification_ROC.Amor.y) <- all.Classification_ROC.name

all.Classification_ROC.Apohele.x <- cbindX(all.Classification_ROC.Apohele.x, data.frame(dt.Classification.roc.Apohele$x.value))
colnames(all.Classification_ROC.Apohele.x) <- all.Classification_ROC.name
all.Classification_ROC.Apohele.y <- cbindX(all.Classification_ROC.Apohele.y, data.frame(dt.Classification.roc.Apohele$y.value))
colnames(all.Classification_ROC.Apohele.y) <- all.Classification_ROC.name

all.Classification_ROC.Apollo.x <- cbindX(all.Classification_ROC.Apollo.x, data.frame(dt.Classification.roc.Apollo$x.value))
colnames(all.Classification_ROC.Apollo.x) <- all.Classification_ROC.name
all.Classification_ROC.Apollo.y <- cbindX(all.Classification_ROC.Apollo.y, data.frame(dt.Classification.roc.Apollo$y.value))
colnames(all.Classification_ROC.Apollo.y) <- all.Classification_ROC.name

all.Classification_ROC.Aten.x <- cbindX(all.Classification_ROC.Aten.x, data.frame(dt.Classification.roc.Aten$x.value))
colnames(all.Classification_ROC.Aten.x) <- all.Classification_ROC.name
all.Classification_ROC.Aten.y <- cbindX(all.Classification_ROC.Aten.y, data.frame(dt.Classification.roc.Aten$y.value))
colnames(all.Classification_ROC.Aten.y) <- all.Classification_ROC.name

all.Classification.All[mod.name] <- c(Accuracy, MacroSensitivity, MacroSpecificity, MacroPrecision, MacroRecall, MacroF1)

tdist <- list()
tdist$acc <- paste(as.character(round(Accuracy,4)))
tdist$sens <- paste(as.character(round(MacroSensitivity,4)))
tdist$spec <- paste(as.character(round(MacroSpecificity,4)))
tdist$prec <- paste(as.character(round(MacroPrecision,4)))
tdist$rec <- paste(as.character(round(MacroRecall,4)))
tdist$f1 <- paste(as.character(round(MacroF1,4)))
all.Classification.All[mod.name] <- c(tdist$acc,tdist$sens,tdist$spec,tdist$prec,tdist$rec,tdist$f1)

rdist <- list()
rdist_val = dt.Classification.roc.Amor$auc
rdist$Amor.auc <- paste(as.character(round(rdist_val,8)))
rdist_val = dt.Classification.roc.Amor$optcut
rdist$Amor.optcut <- paste(as.character(round(rdist_val,5)))

rdist_val = dt.Classification.roc.Apohele$auc
rdist$Apohele.auc <- paste(as.character(round(rdist_val,8)))
rdist_val = dt.Classification.roc.Apohele$optcut
rdist$Apohele.optcut <- paste(as.character(round(rdist_val,5)))

rdist_val = dt.Classification.roc.Apollo$auc
rdist$Apollo.auc <- paste(as.character(round(rdist_val,8)))
rdist_val = dt.Classification.roc.Apollo$optcut
rdist$Apollo.optcut <- paste(as.character(round(rdist_val,5)))

rdist_val = dt.Classification.roc.Aten$auc
rdist$Aten.auc <- paste(as.character(round(rdist_val,8)))
rdist_val = dt.Classification.roc.Aten$optcut
rdist$Aten.optcut <- paste(as.character(round(rdist_val,5)))
all.Classification.ROC.All[mod.name] <- c(rdist$Amor.auc,rdist$Amor.optcut,rdist$Apohele.auc,rdist$Apohele.optcut,rdist$Apollo.auc,rdist$Apollo.optcut,rdist$Aten.auc,rdist$Aten.optcut)
#SVM
dt.Classification.model <- svm(Classification ~ Orbit.Axis..AU. + Orbit.Eccentricity + Orbit.Inclination..deg. + Perihelion.Argument..deg. + Node.Longitude..deg. + Mean.Anomoly..deg. + Perihelion.Distance..AU. + Aphelion.Distance..AU. + Orbital.Period..yr. + Minimum.Orbit.Intersection.Distance..AU. + Asteroid.Magnitude,
  data=asteroids_split$train, kernel="linear", cost=10, type="C-classification",probability = TRUE)

dt.Classification.pred <- predict(dt.Classification.model, asteroids_split$test, type = "class")
dt.Classification.pred.prob <- predict(dt.Classification.model, asteroids_split$test, probability=TRUE)

folds_confusion <- NULL
folds_confusion <- Confusion_Sum(folds_confusion, reference=asteroids_split$test$Classification, data=dt.Classification.pred)
img_name_plot <- paste("IMG_asteroids_model_BEST_SVM_Classification_confusion" ,".png", sep = "")
  png(img_name_plot)
  grid.table(folds_confusion)
  dev.off()

dt.Classification.confusion_matrix_multiclass = confusionMatrix(
  data=dt.Classification.pred, reference=asteroids_split$test$Classification, mode = "prec_recall") 

confusion.multi = dt.Classification.confusion_matrix_multiclass$byClass

sens_Amor = confusion.multi["Class: Amor Asteroid","Sensitivity"] 
spec_Amor = confusion.multi["Class: Amor Asteroid","Specificity"]
prec_Amor = confusion.multi["Class: Amor Asteroid","Precision"]
recal_Amor = confusion.multi["Class: Amor Asteroid","Recall"]
f1_Amor = confusion.multi["Class: Amor Asteroid","F1"]

sens_Apohele = confusion.multi["Class: Apohele Asteroid","Sensitivity"] 
spec_Apohele = confusion.multi["Class: Apohele Asteroid","Specificity"]
prec_Apohele = confusion.multi["Class: Apohele Asteroid","Precision"]
recal_Apohele = confusion.multi["Class: Apohele Asteroid","Recall"]
f1_Apohele = confusion.multi["Class: Apohele Asteroid","F1"]

sens_Apollo = confusion.multi["Class: Apollo Asteroid","Sensitivity"] 
spec_Apollo = confusion.multi["Class: Apollo Asteroid","Specificity"]
prec_Apollo = confusion.multi["Class: Apollo Asteroid","Precision"]
recal_Apollo = confusion.multi["Class: Apollo Asteroid","Recall"]
f1_Apollo = confusion.multi["Class: Apollo Asteroid","F1"]

sens_Aten = confusion.multi["Class: Aten Asteroid","Sensitivity"] 
spec_Aten = confusion.multi["Class: Aten Asteroid","Specificity"]
prec_Aten = confusion.multi["Class: Aten Asteroid","Precision"]
recal_Aten = confusion.multi["Class: Aten Asteroid","Recall"]
f1_Aten = confusion.multi["Class: Aten Asteroid","F1"]

Accuracy = dt.Classification.confusion_matrix_multiclass$overall["Accuracy"]
MacroSensitivity = (0.25 * sens_Amor) + (0.25 * sens_Apohele) + (0.25 * sens_Apollo) + (0.25 * sens_Aten)
MacroSpecificity = (0.25 * spec_Amor) + (0.25 * spec_Apohele) + (0.25 * spec_Apollo) + (0.25 * spec_Aten)
MacroPrecision = (0.25 * prec_Amor) + (0.25 * prec_Apohele) + (0.25 * prec_Apollo) + (0.25 * prec_Aten)
MacroRecall = (0.25 * recal_Amor) + (0.25 * recal_Apohele) + (0.25 * recal_Apollo) + (0.25 * recal_Aten)
MacroF1 = (0.25 * f1_Amor) + (0.25 * f1_Apohele) + (0.25 * f1_Apollo) + (0.25 * f1_Aten)
mod.name <- paste("Classification SVM")

all.Classification$Accuracy[mod.name] <- Accuracy
all.Classification$MacroSensitivity[mod.name] <- MacroSensitivity
all.Classification$MacroSpecificity[mod.name] <- MacroSpecificity
all.Classification$MacroPrecision[mod.name] <- MacroPrecision
all.Classification$MacroRecall[mod.name] <- MacroRecall
all.Classification$MacroF1[mod.name] <- MacroF1

dt.Classification.roc.Amor = ROCFunction.MULTI(attr(dt.Classification.pred.prob, "probabilities"), as.factor(asteroids_split$test$Classification), "Amor Asteroid")
dt.Classification.roc.Apohele = ROCFunction.MULTI(attr(dt.Classification.pred.prob, "probabilities"), as.factor(asteroids_split$test$Classification), "Apohele Asteroid")
dt.Classification.roc.Apollo = ROCFunction.MULTI(attr(dt.Classification.pred.prob, "probabilities"), as.factor(asteroids_split$test$Classification), "Apollo Asteroid")
dt.Classification.roc.Aten = ROCFunction.MULTI(attr(dt.Classification.pred.prob, "probabilities"), as.factor(asteroids_split$test$Classification), "Aten Asteroid")

#roc
all.Classification$Amor.AUC[mod.name] <- dt.Classification.roc.Amor$auc
all.Classification$Amor.CutOffOpt[mod.name] <- dt.Classification.roc.Amor$optcut
all.Classification$Apohele.AUC[mod.name] <- dt.Classification.roc.Apohele$auc
all.Classification$Apohele.CutOffOpt[mod.name] <- dt.Classification.roc.Apohele$optcut
all.Classification$Apollo.AUC[mod.name] <- dt.Classification.roc.Apollo$auc
all.Classification$Apollo.CutOffOpt[mod.name] <- dt.Classification.roc.Apollo$optcut
all.Classification$Aten.AUC[mod.name] <- dt.Classification.roc.Aten$auc
all.Classification$Aten.CutOffOpt[mod.name] <- dt.Classification.roc.Aten$optcut

# ROC DATA FRAMES

all.Classification_ROC.name = c(all.Classification_ROC.name, mod.name)

all.Classification_ROC.Amor.x <- cbindX(all.Classification_ROC.Amor.x, data.frame(dt.Classification.roc.Amor$x.value))
colnames(all.Classification_ROC.Amor.x) <- all.Classification_ROC.name
all.Classification_ROC.Amor.y <- cbindX(all.Classification_ROC.Amor.y, data.frame(dt.Classification.roc.Amor$y.value))
colnames(all.Classification_ROC.Amor.y) <- all.Classification_ROC.name

all.Classification_ROC.Apohele.x <- cbindX(all.Classification_ROC.Apohele.x, data.frame(dt.Classification.roc.Apohele$x.value))
colnames(all.Classification_ROC.Apohele.x) <- all.Classification_ROC.name
all.Classification_ROC.Apohele.y <- cbindX(all.Classification_ROC.Apohele.y, data.frame(dt.Classification.roc.Apohele$y.value))
colnames(all.Classification_ROC.Apohele.y) <- all.Classification_ROC.name

all.Classification_ROC.Apollo.x <- cbindX(all.Classification_ROC.Apollo.x, data.frame(dt.Classification.roc.Apollo$x.value))
colnames(all.Classification_ROC.Apollo.x) <- all.Classification_ROC.name
all.Classification_ROC.Apollo.y <- cbindX(all.Classification_ROC.Apollo.y, data.frame(dt.Classification.roc.Apollo$y.value))
colnames(all.Classification_ROC.Apollo.y) <- all.Classification_ROC.name

all.Classification_ROC.Aten.x <- cbindX(all.Classification_ROC.Aten.x, data.frame(dt.Classification.roc.Aten$x.value))
colnames(all.Classification_ROC.Aten.x) <- all.Classification_ROC.name
all.Classification_ROC.Aten.y <- cbindX(all.Classification_ROC.Aten.y, data.frame(dt.Classification.roc.Aten$y.value))
colnames(all.Classification_ROC.Aten.y) <- all.Classification_ROC.name

all.Classification.All[mod.name] <- c(Accuracy, MacroSensitivity, MacroSpecificity, MacroPrecision, MacroRecall, MacroF1)

tdist <- list()
tdist$acc <- paste(as.character(round(Accuracy,4)))
tdist$sens <- paste(as.character(round(MacroSensitivity,4)))
tdist$spec <- paste(as.character(round(MacroSpecificity,4)))
tdist$prec <- paste(as.character(round(MacroPrecision,4)))
tdist$rec <- paste(as.character(round(MacroRecall,4)))
tdist$f1 <- paste(as.character(round(MacroF1,4)))
all.Classification.All[mod.name] <- c(tdist$acc,tdist$sens,tdist$spec,tdist$prec,tdist$rec,tdist$f1)

rdist <- list()
rdist_val = dt.Classification.roc.Amor$auc
rdist$Amor.auc <- paste(as.character(round(rdist_val,8)))
rdist_val = dt.Classification.roc.Amor$optcut
rdist$Amor.optcut <- paste(as.character(round(rdist_val,5)))

rdist_val = dt.Classification.roc.Apohele$auc
rdist$Apohele.auc <- paste(as.character(round(rdist_val,8)))
rdist_val = dt.Classification.roc.Apohele$optcut
rdist$Apohele.optcut <- paste(as.character(round(rdist_val,5)))

rdist_val = dt.Classification.roc.Apollo$auc
rdist$Apollo.auc <- paste(as.character(round(rdist_val,8)))
rdist_val = dt.Classification.roc.Apollo$optcut
rdist$Apollo.optcut <- paste(as.character(round(rdist_val,5)))

rdist_val = dt.Classification.roc.Aten$auc
rdist$Aten.auc <- paste(as.character(round(rdist_val,8)))
rdist_val = dt.Classification.roc.Aten$optcut
rdist$Aten.optcut <- paste(as.character(round(rdist_val,5)))
all.Classification.ROC.All[mod.name] <- c(rdist$Amor.auc,rdist$Amor.optcut,rdist$Apohele.auc,rdist$Apohele.optcut,rdist$Apollo.auc,rdist$Apollo.optcut,rdist$Aten.auc,rdist$Aten.optcut)
#RNN

asteroids_split$train$Amor = asteroids_split$train$Classification == "Amor Asteroid"
asteroids_split$train$Apohele = asteroids_split$train$Classification == "Apohele Asteroid"
asteroids_split$train$Apollo = asteroids_split$train$Classification == "Apollo Asteroid"
asteroids_split$train$Aten = asteroids_split$train$Classification == "Aten Asteroid"


dt.Classification.model = neuralnet(Amor+Apohele+Apollo+Aten ~ Orbit.Axis..AU._scaled_scaled + Orbit.Eccentricity_scaled + Orbit.Inclination..deg._scaled + Perihelion.Argument..deg._scaled + Node.Longitude..deg._scaled + Mean.Anomoly..deg._scaled + Perihelion.Distance..AU._scaled + Aphelion.Distance..AU._scaled + Orbital.Period..yr._scaled + Minimum.Orbit.Intersection.Distance..AU._scaled + Asteroid.Magnitude_scaled,
                               asteroids_split$train,
                               hidden = c(7),
                               act.fct = "tanh",
                               
                               learningrate.limit = NULL,
                               learningrate.factor =
                                 list(minus = 0.5, plus = 1.2),
                               algorithm = "rprop+",
                               
                               #err.fct = loss_f,
                               threshold = 0.5,
                               lifesign="full",
                               
                               stepmax = 1e6,
                               linear.output = TRUE)

dt.Classification.pred = predict(dt.Classification.model, asteroids_split$test)
dt.Classification.pred.max = as.factor(c("Amor Asteroid", "Apohele Asteroid", "Apollo Asteroid", "Aten Asteroid")[apply(dt.Classification.pred, 1, which.max)])

dt.Classification.confusion_matrix_multiclass = confusionMatrix(
  data=dt.Classification.pred.max, reference=asteroids_split$test$Classification, mode = "prec_recall") 

folds_confusion <- NULL
folds_confusion <- Confusion_Sum(folds_confusion, reference=asteroids_split$test$Classification, data=dt.Classification.pred.max)
img_name_plot <- paste("IMG_asteroids_model_BEST_RNN_Classification_confusion" ,".png", sep = "")
  png(img_name_plot)
  grid.table(folds_confusion)
  dev.off()

confusion.multi = dt.Classification.confusion_matrix_multiclass$byClass

sens_Amor = confusion.multi["Class: Amor Asteroid","Sensitivity"] 
spec_Amor = confusion.multi["Class: Amor Asteroid","Specificity"]
prec_Amor = confusion.multi["Class: Amor Asteroid","Precision"]
recal_Amor = confusion.multi["Class: Amor Asteroid","Recall"]
f1_Amor = confusion.multi["Class: Amor Asteroid","F1"]

sens_Apohele = confusion.multi["Class: Apohele Asteroid","Sensitivity"] 
spec_Apohele = confusion.multi["Class: Apohele Asteroid","Specificity"]
prec_Apohele = confusion.multi["Class: Apohele Asteroid","Precision"]
recal_Apohele = confusion.multi["Class: Apohele Asteroid","Recall"]
f1_Apohele = confusion.multi["Class: Apohele Asteroid","F1"]

sens_Apollo = confusion.multi["Class: Apollo Asteroid","Sensitivity"] 
spec_Apollo = confusion.multi["Class: Apollo Asteroid","Specificity"]
prec_Apollo = confusion.multi["Class: Apollo Asteroid","Precision"]
recal_Apollo = confusion.multi["Class: Apollo Asteroid","Recall"]
f1_Apollo = confusion.multi["Class: Apollo Asteroid","F1"]

sens_Aten = confusion.multi["Class: Aten Asteroid","Sensitivity"] 
spec_Aten = confusion.multi["Class: Aten Asteroid","Specificity"]
prec_Aten = confusion.multi["Class: Aten Asteroid","Precision"]
recal_Aten = confusion.multi["Class: Aten Asteroid","Recall"]
f1_Aten = confusion.multi["Class: Aten Asteroid","F1"]

Accuracy = dt.Classification.confusion_matrix_multiclass$overall["Accuracy"]
MacroSensitivity = (0.25 * sens_Amor) + (0.25 * sens_Apohele) + (0.25 * sens_Apollo) + (0.25 * sens_Aten)
MacroSpecificity = (0.25 * spec_Amor) + (0.25 * spec_Apohele) + (0.25 * spec_Apollo) + (0.25 * spec_Aten)
MacroPrecision = (0.25 * prec_Amor) + (0.25 * prec_Apohele) + (0.25 * prec_Apollo) + (0.25 * prec_Aten)
MacroRecall = (0.25 * recal_Amor) + (0.25 * recal_Apohele) + (0.25 * recal_Apollo) + (0.25 * recal_Aten)
MacroF1 = (0.25 * f1_Amor) + (0.25 * f1_Apohele) + (0.25 * f1_Apollo) + (0.25 * prec_Aten)
mod.name <- paste("Classification RNN")

all.Classification$Accuracy[mod.name] <- Accuracy
all.Classification$MacroSensitivity[mod.name] <- MacroSensitivity
all.Classification$MacroSpecificity[mod.name] <- MacroSpecificity
all.Classification$MacroPrecision[mod.name] <- MacroPrecision
all.Classification$MacroRecall[mod.name] <- MacroRecall
all.Classification$MacroF1[mod.name] <- MacroF1


colnames(dt.Classification.pred) <- c("Amor Asteroid","Apohele Asteroid","Apollo Asteroid","Aten Asteroid")
dt.Classification.roc.Amor = ROCFunction.MULTI.RNN(dt.Classification.pred,asteroids_split$test$Classification,"Amor Asteroid")
dt.Classification.roc.Apohele = ROCFunction.MULTI.RNN(dt.Classification.pred,asteroids_split$test$Classification,"Apohele Asteroid")
dt.Classification.roc.Apollo = ROCFunction.MULTI.RNN(dt.Classification.pred,asteroids_split$test$Classification,"Apollo Asteroid")
dt.Classification.roc.Aten = ROCFunction.MULTI.RNN(dt.Classification.pred,asteroids_split$test$Classification,"Aten Asteroid")

#roc
all.Classification$Amor.AUC[mod.name] <- dt.Classification.roc.Amor$auc
all.Classification$Amor.CutOffOpt[mod.name] <- dt.Classification.roc.Amor$optcut
all.Classification$Apohele.AUC[mod.name] <- dt.Classification.roc.Apohele$auc
all.Classification$Apohele.CutOffOpt[mod.name] <- dt.Classification.roc.Apohele$optcut
all.Classification$Apollo.AUC[mod.name] <- dt.Classification.roc.Apollo$auc
all.Classification$Apollo.CutOffOpt[mod.name] <- dt.Classification.roc.Apollo$optcut
all.Classification$Aten.AUC[mod.name] <- dt.Classification.roc.Aten$auc
all.Classification$Aten.CutOffOpt[mod.name] <- dt.Classification.roc.Aten$optcut

# ROC DATA FRAMES

all.Classification_ROC.name = c(all.Classification_ROC.name, mod.name)

all.Classification_ROC.Amor.x <- cbindX(all.Classification_ROC.Amor.x, data.frame(dt.Classification.roc.Amor$x.value))
colnames(all.Classification_ROC.Amor.x) <- all.Classification_ROC.name
all.Classification_ROC.Amor.y <- cbindX(all.Classification_ROC.Amor.y, data.frame(dt.Classification.roc.Amor$y.value))
colnames(all.Classification_ROC.Amor.y) <- all.Classification_ROC.name

all.Classification_ROC.Apohele.x <- cbindX(all.Classification_ROC.Apohele.x, data.frame(dt.Classification.roc.Apohele$x.value))
colnames(all.Classification_ROC.Apohele.x) <- all.Classification_ROC.name
all.Classification_ROC.Apohele.y <- cbindX(all.Classification_ROC.Apohele.y, data.frame(dt.Classification.roc.Apohele$y.value))
colnames(all.Classification_ROC.Apohele.y) <- all.Classification_ROC.name

all.Classification_ROC.Apollo.x <- cbindX(all.Classification_ROC.Apollo.x, data.frame(dt.Classification.roc.Apollo$x.value))
colnames(all.Classification_ROC.Apollo.x) <- all.Classification_ROC.name
all.Classification_ROC.Apollo.y <- cbindX(all.Classification_ROC.Apollo.y, data.frame(dt.Classification.roc.Apollo$y.value))
colnames(all.Classification_ROC.Apollo.y) <- all.Classification_ROC.name

all.Classification_ROC.Aten.x <- cbindX(all.Classification_ROC.Aten.x, data.frame(dt.Classification.roc.Aten$x.value))
colnames(all.Classification_ROC.Aten.x) <- all.Classification_ROC.name
all.Classification_ROC.Aten.y <- cbindX(all.Classification_ROC.Aten.y, data.frame(dt.Classification.roc.Aten$y.value))
colnames(all.Classification_ROC.Aten.y) <- all.Classification_ROC.name

all.Classification.All[mod.name] <- c(Accuracy, MacroSensitivity, MacroSpecificity, MacroPrecision, MacroRecall, MacroF1)

tdist <- list()
tdist$acc <- paste(as.character(round(Accuracy,4)))
tdist$sens <- paste(as.character(round(MacroSensitivity,4)))
tdist$spec <- paste(as.character(round(MacroSpecificity,4)))
tdist$prec <- paste(as.character(round(MacroPrecision,4)))
tdist$rec <- paste(as.character(round(MacroRecall,4)))
tdist$f1 <- paste(as.character(round(MacroF1,4)))
all.Classification.All[mod.name] <- c(tdist$acc,tdist$sens,tdist$spec,tdist$prec,tdist$rec,tdist$f1)

rdist <- list()
rdist_val = dt.Classification.roc.Amor$auc
rdist$Amor.auc <- paste(as.character(round(rdist_val,8)))
rdist_val = dt.Classification.roc.Amor$optcut
rdist$Amor.optcut <- paste(as.character(round(rdist_val,5)))

rdist_val = dt.Classification.roc.Apohele$auc
rdist$Apohele.auc <- paste(as.character(round(rdist_val,8)))
rdist_val = dt.Classification.roc.Apohele$optcut
rdist$Apohele.optcut <- paste(as.character(round(rdist_val,5)))

rdist_val = dt.Classification.roc.Apollo$auc
rdist$Apollo.auc <- paste(as.character(round(rdist_val,8)))
rdist_val = dt.Classification.roc.Apollo$optcut
rdist$Apollo.optcut <- paste(as.character(round(rdist_val,5)))

rdist_val = dt.Classification.roc.Aten$auc
rdist$Aten.auc <- paste(as.character(round(rdist_val,8)))
rdist_val = dt.Classification.roc.Aten$optcut
rdist$Aten.optcut <- paste(as.character(round(rdist_val,5)))
all.Classification.ROC.All[mod.name] <- c(rdist$Amor.auc,rdist$Amor.optcut,rdist$Apohele.auc,rdist$Apohele.optcut,rdist$Apollo.auc,rdist$Apollo.optcut,rdist$Aten.auc,rdist$Aten.optcut)


end_table <- length(all.Classification$Accuracy)
plot.models.color = rainbow(end_table-1)

img_name_plot <- paste("IMG_asteroids_model_BEST_", "Classification_KFOLD_ROC", ".png", sep = "")
png(img_name_plot,res = 800, height = 10, width = 15, unit='in')
par(mfrow=c(2,2))  
# Amor
plot.new()
ROCPlot.x.class = colnames(all.Classification_ROC.Amor.x)
ROCPlot.y.class = colnames(all.Classification_ROC.Amor.y)

title(main="ROC Class: Amor", xlab="Sensitivity - True Positive Rate", ylab="Specificity - False Positive Rate")
for (ROCPlot.x.classindex in 2:length(ROCPlot.x.class)){
  ROCPlot.name = ROCPlot.x.class[ROCPlot.x.classindex]
  ROCPlot.y.classindex = which(ROCPlot.y.class == ROCPlot.x.class[ROCPlot.x.classindex])
  
  ROCPlot.x = unlist(all.Classification_ROC.Amor.x[,ROCPlot.x.classindex], use.names=FALSE)
  ROCPlot.y = unlist(all.Classification_ROC.Amor.y[,ROCPlot.y.classindex], use.names=FALSE)
  lines(ROCPlot.x, ROCPlot.y, col=plot.models.color[ROCPlot.x.classindex-1],lwd=1)    
}
legend("right", title="models",legend=ROCPlot.y.class,lwd=5, col=plot.models.color,horiz=FALSE)
#Apohele
plot.new()
ROCPlot.x.class = colnames(all.Classification_ROC.Apohele.x)
ROCPlot.y.class = colnames(all.Classification_ROC.Apohele.y)

title(main="ROC Class: Apohele", xlab="Sensitivity - True Positive Rate", ylab="Specificity - False Positive Rate")
for (ROCPlot.x.classindex in 2:length(ROCPlot.x.class)){
  ROCPlot.name = ROCPlot.x.class[ROCPlot.x.classindex]
  ROCPlot.y.classindex = which(ROCPlot.y.class == ROCPlot.x.class[ROCPlot.x.classindex])
  
  ROCPlot.x = unlist(all.Classification_ROC.Apohele.x[,ROCPlot.x.classindex], use.names=FALSE)
  ROCPlot.y = unlist(all.Classification_ROC.Apohele.y[,ROCPlot.y.classindex], use.names=FALSE)
  lines(ROCPlot.x, ROCPlot.y, col=plot.models.color[ROCPlot.x.classindex-1],lwd=1)    
}
legend("right", title="models",legend=ROCPlot.y.class,lwd=5, col=plot.models.color,horiz=FALSE)
#Apollo
plot.new()
ROCPlot.x.class = colnames(all.Classification_ROC.Apollo.x)
ROCPlot.y.class = colnames(all.Classification_ROC.Apollo.y)

title(main="ROC Class: Apollo", xlab="Sensitivity - True Positive Rate", ylab="Specificity - False Positive Rate")
for (ROCPlot.x.classindex in 2:length(ROCPlot.x.class)){
  ROCPlot.name = ROCPlot.x.class[ROCPlot.x.classindex]
  ROCPlot.y.classindex = which(ROCPlot.y.class == ROCPlot.x.class[ROCPlot.x.classindex])
  
  ROCPlot.x = unlist(all.Classification_ROC.Apollo.x[,ROCPlot.x.classindex], use.names=FALSE)
  ROCPlot.y = unlist(all.Classification_ROC.Apollo.y[,ROCPlot.y.classindex], use.names=FALSE)
  lines(ROCPlot.x, ROCPlot.y, col=plot.models.color[ROCPlot.x.classindex-1],lwd=1)    
}
legend("right", title="models",legend=ROCPlot.y.class,lwd=5, col=plot.models.color,horiz=FALSE)
#Aten
plot.new()
ROCPlot.x.class = colnames(all.Classification_ROC.Aten.x)
ROCPlot.y.class = colnames(all.Classification_ROC.Aten.y)

title(main="ROC Class: Aten", xlab="Sensitivity - True Positive Rate", ylab="Specificity - False Positive Rate")
for (ROCPlot.x.classindex in 2:length(ROCPlot.x.class)){
  ROCPlot.name = ROCPlot.x.class[ROCPlot.x.classindex]
  ROCPlot.y.classindex = which(ROCPlot.y.class == ROCPlot.x.class[ROCPlot.x.classindex])
  
  ROCPlot.x = unlist(all.Classification_ROC.Aten.x[,ROCPlot.x.classindex], use.names=FALSE)
  ROCPlot.y = unlist(all.Classification_ROC.Aten.y[,ROCPlot.y.classindex], use.names=FALSE)
  lines(ROCPlot.x, ROCPlot.y, col=plot.models.color[ROCPlot.x.classindex-1],lwd=1)    
}
legend("right", title="models",legend=ROCPlot.y.class,lwd=5, col=plot.models.color,horiz=FALSE)
dev.off()



img_name_plot <- paste("IMG_asteroids_model_BEST_", "Classification_KFOLD_performance_plot_A", ".png", sep = "")
png(img_name_plot,res = 800, height = 10, width = 15, unit='in')
par(mfrow=c(2,2)) 
boxplot(all.Classification$Accuracy[2:end_table],vertical = TRUE, pch=19, las = 2, col = plot.models.color,main="Accuracy")  
boxplot(all.Classification$MacroSensitivity[2:end_table],vertical = TRUE, pch=19, las = 2, col = plot.models.color,main="MacroSensitivity")  
dev.off()
img_name_plot <- paste("IMG_asteroids_model_BEST_", "Classification_KFOLD_performance_plot_B", ".png", sep = "")
png(img_name_plot,res = 800, height = 10, width = 15, unit='in')
par(mfrow=c(2,2))  
boxplot(all.Classification$MacroSpecificity[2:end_table], vertical = TRUE, pch=19, las = 2, col = plot.models.color,main="MacroSpecificity")  
boxplot(all.Classification$MacroPrecision[2:end_table], vertical = TRUE, pch=19, las = 2, col = plot.models.color,main="MacroPrecision")  
dev.off()
img_name_plot <- paste("IMG_asteroids_model_BEST_", "Classification_KFOLD_performance_plot_C", ".png", sep = "")
png(img_name_plot,res = 800, height = 10, width = 15, unit='in')
par(mfrow=c(2,2))  
boxplot(all.Classification$MacroRecall[2:end_table],vertical = TRUE, pch=19, las = 2, col = plot.models.color,main="MacroRecall")  
boxplot(all.Classification$MacroF1[2:end_table],vertical = TRUE, pch=19, las = 2, col = plot.models.color,main="MacroF1")  
dev.off()

img_name_plot <- paste("PDF_asteroids_model_BEST_", "Classification_KFOLD_performance_tdist", ".pdf", sep = "")
pdf(img_name_plot, height = 20, width = 46)
grid.table(t(all.Classification.All[2:length(all.Classification.All)]))
dev.off()

img_name_plot <- paste("PDF_asteroids_model_BEST_", "Classification_KFOLD_performance_ROC", ".pdf", sep = "")
pdf(img_name_plot, height = 20, width = 46)
grid.table(t(all.Classification.ROC.All[2:length(all.Classification.ROC.All)]))
dev.off()
