setwd("~/Github/AsteroidsClassification")

library(plyr)
library(gridExtra)
library(e1071)
library(caret)
library(tidyverse)
library(ROCR)
library(gdata)

Confusion_Sum <- function(cm_global, data, reference){
  cm_fold <- table(reference, data)
  if (is.null(cm_global)){
    cm_global <- cm_fold
  }else{
    cm_global <- cm_global + cm_fold
  }
  return(cm_global)
}
#t-distribution
confidence_interval <- function(vector, interval) {
  # Standard deviation of sample
  vec_sd <- sd(vector, na.rm = TRUE)
  # Sample size
  n <- length(vector[!is.na(vector)])

  # Mean of sample
  vec_mean <- mean(vector, na.rm = TRUE)
  # Error according to t distribution
  error <- qt((interval + 1)/2, df = n - 1) * vec_sd / sqrt(n)
  # Confidence interval as a vector
  result <- c("err" = error, "mean" = vec_mean)
  return(result)
}


ROCFunction.optcut = function(perf, pred){
  cut.ind = mapply(FUN=function(x, y, p){
    d = (x - 0)^2 + (y-1)^2
    ind = which(d == min(d))
    c(sensitivity = y[[ind]], 
      specificity = 1-x[[ind]], 
      cutoff = p[[ind]])
  }, perf@x.values, perf@y.values, pred@cutoffs)
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

ROCFunction.MULTI <- function(ROCFun.pred.prob, testLabels, classInLabel){
  #ROCFun.pred = predict(model, testset,  probability=TRUE)
  #ROCFun.pred.prob = attr(ROCFun.pred, "probabilities") 
  ROCFun.pred.class = colnames(ROCFun.pred.prob)
  ROCFun.pred.classindex = which(ROCFun.pred.class == classInLabel)
  ROCFun.pred.to.roc = unlist(ROCFun.pred.prob[,ROCFun.pred.classindex], use.names=FALSE)
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

#load dataset RObject as asteroids_split
load("DATA_asteroids_dataset_split_0.7.RData")

asteroids_split$train$Hazardous.int = as.integer(asteroids_split$train$Hazardous)

folds.number = 5
kernel_list <- c('linear','polynomial','radial')
C_list <-c(1,10)

folds <- split(asteroids_split$train, cut(sample(1:nrow(asteroids_split$train)), folds.number))
rm(folds.number)
# Hazardous
svm.Hazardous <- list()

mx = matrix(NA, nrow = length(folds))

svm.Hazardous$Accuracy = data.frame(mx)
svm.Hazardous$MacroSensitivity <- data.frame(mx)
svm.Hazardous$MacroSpecificity <- data.frame(mx)
svm.Hazardous$MacroPrecision <- data.frame(mx)
svm.Hazardous$MacroRecall <- data.frame(mx)
svm.Hazardous$MacroF1 <- data.frame(mx)
svm.Hazardous$AUC <- data.frame(mx)
svm.Hazardous$CutOffOpt <- data.frame(mx)
svm.Hazardous_ROC.x <- matrix()
svm.Hazardous_ROC.y <- matrix()
svm.Hazardous_ROC.name <- c(NA)

mx = matrix(NA, nrow = 8)
svm.Hazardous.All <- data.frame(mx)
svm.Hazardous.All["Performance"] = c("Accuracy","MacroSensitivity","MacroSpecificity","MacroPrecision","MacroRecall","MacroF1","AUC","OptimalCutOff")
rm(mx)

for (j in 1:length(kernel_list)) {
  hyper.kernel <- kernel_list[j]
  
  for (l in 1:length(C_list)) {
    hyper.cost <- C_list[l]
    svm.Hazardous.stats <- list()
    
    svm.Hazardous.stats.roc.pred.prob <- list()
    svm.Hazardous.stats.roc.thruth <- list()
    folds_confusion <- NULL
    for (i in 1:length(folds)) {
      fold.valid <- ldply(folds[i], data.frame)
      fold.valid <- fold.valid[, !names(fold.valid) %in% c(".id")]
      fold.train <- ldply(folds[-i], data.frame)
      fold.train <- fold.train[, !names(fold.train) %in% c(".id")]
      
      
      print(paste(as.character(i), hyper.kernel , as.character(hyper.cost), sep = " "))
      svm.Hazardous.model = svm(Hazardous ~ Orbit.Axis..AU. + Orbit.Eccentricity + Orbit.Inclination..deg. + Perihelion.Argument..deg. + Node.Longitude..deg. + Mean.Anomoly..deg. + Perihelion.Distance..AU. + Aphelion.Distance..AU. + Orbital.Period..yr. + Minimum.Orbit.Intersection.Distance..AU. + Asteroid.Magnitude,
        data=fold.train, kernel=hyper.kernel, cost=hyper.cost, type="C-classification", probability = TRUE)
      svm.Hazardous.pred = predict(svm.Hazardous.model, fold.valid)
      
      
      svm.Hazardous.confusion_matrix_true = confusionMatrix(
        data=as.factor(svm.Hazardous.pred), reference=as.factor(fold.valid$Hazardous), positive="TRUE", mode = "prec_recall") 
      
      svm.Hazardous.confusion_matrix_false = confusionMatrix(
        data=as.factor(svm.Hazardous.pred), reference=as.factor(fold.valid$Hazardous), positive="FALSE", mode = "prec_recall") 
      
      folds_confusion <- Confusion_Sum(folds_confusion, reference=fold.valid$Hazardous, data=as.factor(svm.Hazardous.pred))
      
      sens_true = svm.Hazardous.confusion_matrix_true$byClass["Sensitivity"]
      spec_true = svm.Hazardous.confusion_matrix_true$byClass["Specificity"]
      prec_true = svm.Hazardous.confusion_matrix_true$byClass["Precision"]
      recal_true = svm.Hazardous.confusion_matrix_true$byClass["Recall"]
      f1_true = svm.Hazardous.confusion_matrix_true$byClass["F1"]
      
      sens_false = svm.Hazardous.confusion_matrix_false$byClass["Sensitivity"]
      spec_fasle = svm.Hazardous.confusion_matrix_false$byClass["Specificity"]
      prec_false = svm.Hazardous.confusion_matrix_false$byClass["Precision"]
      recal_false = svm.Hazardous.confusion_matrix_false$byClass["Recall"]
      f1_false = svm.Hazardous.confusion_matrix_false$byClass["F1"]
      
      MacroSensitivity = (0.5 * sens_true) + (0.5 * sens_false)
      MacroSpecificity = (0.5 * spec_true) + (0.5 * spec_fasle)
      MacroPrecision = (0.5 * prec_true) + (0.5 * prec_false)
      MacroRecall = (0.5 * recal_true) + (0.5 * recal_false)
      MacroF1 = (0.5 * f1_true) + (0.5 * f1_false)
    
      svm.Hazardous.stats$Accuracy    = append(svm.Hazardous.stats$Accuracy, svm.Hazardous.confusion_matrix_true$overall["Accuracy"])
      svm.Hazardous.stats$MacroSensitivity = append(svm.Hazardous.stats$MacroSensitivity, MacroSensitivity)
      svm.Hazardous.stats$MacroSpecificity = append(svm.Hazardous.stats$MacroSpecificity, MacroSpecificity)
      svm.Hazardous.stats$MacroPrecision = append(svm.Hazardous.stats$MacroPrecision, MacroPrecision)
      svm.Hazardous.stats$MacroRecall = append(svm.Hazardous.stats$MacroRecall, MacroRecall)
      svm.Hazardous.stats$MacroF1 = append(svm.Hazardous.stats$MacroF1, MacroF1)
      
      #ROC
      ROCFun.pred = predict(svm.Hazardous.model,fold.valid,  probability=TRUE)
      ROCFun.pred.prob = attr(ROCFun.pred, "probabilities")
      
      svm.Hazardous.stats.roc.pred.prob = rbind(svm.Hazardous.stats.roc.pred.prob,ROCFun.pred.prob )
      svm.Hazardous.stats.roc.thruth = append(svm.Hazardous.stats.roc.thruth, as.factor(fold.valid$Hazardous))
      
      
    }
    
    
    svm.Hazardous.roc = ROCFunction.BIN(svm.Hazardous.stats.roc.pred.prob,svm.Hazardous.stats.roc.thruth,"TRUE")
    
    #plot(svm.Hazardous.roc$x.value,svm.Hazardous.roc$y.value , main=paste("AUC:",(svm.Hazardous.roc$auc)))

    
    svm.name <- paste("Haz",hyper.kernel,as.character(hyper.cost),sep="_")
    
    img_name_plot <- paste("IMG_asteroids_model_SWM_", svm.name ,"_confusion" ,".png", sep = "")
    png(img_name_plot)
      grid.table(folds_confusion)
      dev.off()
    
    svm.Hazardous$Accuracy[svm.name] <- svm.Hazardous.stats$Accuracy
    svm.Hazardous$MacroSensitivity[svm.name] <- svm.Hazardous.stats$MacroSensitivity
    svm.Hazardous$MacroSpecificity[svm.name] <- svm.Hazardous.stats$MacroSpecificity
    svm.Hazardous$MacroPrecision[svm.name] <- svm.Hazardous.stats$MacroPrecision
    svm.Hazardous$MacroRecall[svm.name] <- svm.Hazardous.stats$MacroRecall
    svm.Hazardous$MacroF1[svm.name] <- svm.Hazardous.stats$MacroF1
    svm.Hazardous$AUC[svm.name] <- svm.Hazardous.roc$auc
    svm.Hazardous$CutOffOpt[svm.name] <- svm.Hazardous.roc$optcut
    svm.Hazardous_ROC.name = c(svm.Hazardous_ROC.name, svm.name)
    svm.Hazardous_ROC.x <- cbindX(svm.Hazardous_ROC.x, data.frame(svm.Hazardous.roc$x.value))
    colnames(svm.Hazardous_ROC.x) <- svm.Hazardous_ROC.name
    svm.Hazardous_ROC.y <- cbindX(svm.Hazardous_ROC.y, data.frame(svm.Hazardous.roc$y.value))
    colnames(svm.Hazardous_ROC.y) <- svm.Hazardous_ROC.name
    
    tdist <- list()
    tdist_name <- paste("Haz ",hyper.kernel,as.character(hyper.cost),sep=" ")
    tdist_val = confidence_interval(as.vector(svm.Hazardous.stats$Accuracy),0.95)
    tdist$acc <- paste(as.character(round(tdist_val[2],4))," ± ",as.character(round(tdist_val[1],4)))
    
    tdist_val = confidence_interval(as.vector(svm.Hazardous.stats$MacroSensitivity),0.95)
    tdist$sens <- paste(as.character(round(tdist_val[2],4))," ± ",as.character(round(tdist_val[1],4)))
    
    tdist_val = confidence_interval(as.vector(svm.Hazardous.stats$MacroSpecificity),0.95)
    tdist$spec <- paste(as.character(round(tdist_val[2],4))," ± ",as.character(round(tdist_val[1],4)))
    
    tdist_val = confidence_interval(as.vector(svm.Hazardous.stats$MacroPrecision),0.95)
    tdist$prec <- paste(as.character(round(tdist_val[2],4))," ± ",as.character(round(tdist_val[1],4)))
    
    tdist_val = confidence_interval(as.vector(svm.Hazardous.stats$MacroRecall),0.95)
    tdist$rec <- paste(as.character(round(tdist_val[2],4))," ± ",as.character(round(tdist_val[1],4)))
    
    tdist_val = confidence_interval(as.vector(svm.Hazardous.stats$MacroF1),0.95)
    tdist$f1 <- paste(as.character(round(tdist_val[2],4))," ± ",as.character(round(tdist_val[1],4)))
    
    tdist_val = svm.Hazardous.roc$auc
    tdist$auc <- paste(as.character(round(tdist_val,8)))
    tdist_val = svm.Hazardous.roc$optcut
    tdist$cutoff <- paste(as.character(round(tdist_val,5)))
    
    svm.Hazardous.All[tdist_name] <- c(tdist$acc,tdist$sens,tdist$spec,tdist$prec,tdist$rec,tdist$f1,tdist$auc,tdist$cutoff)
    rm(tdist_name,tdist,svm.name,fold.train, fold.valid)
    
  }
}

end_table <- length(svm.Hazardous$Accuracy)
plot.models.color = rainbow(end_table-1)

img_name_plot <- paste("IMG_asteroids_model_SVM_", "Hazardous_KFOLD_ROC", ".png", sep = "")
png(img_name_plot,res = 800, height = 10, width = 15, unit='in')
ROCPlot.x.class = colnames(svm.Hazardous_ROC.x)
ROCPlot.y.class = colnames(svm.Hazardous_ROC.y)
plot.new()
title(main="SVM Hazardous ROC", xlab="Sensitivity - True Positive Rate", ylab="Specificity - False Positive Rate")
for (ROCPlot.x.classindex in 2:length(ROCPlot.x.class)){
  ROCPlot.name = ROCPlot.x.class[ROCPlot.x.classindex]
  ROCPlot.y.classindex = which(ROCPlot.y.class == ROCPlot.x.class[ROCPlot.x.classindex])
  
  ROCPlot.x = unlist(svm.Hazardous_ROC.x[,ROCPlot.x.classindex], use.names=FALSE)
  ROCPlot.y = unlist(svm.Hazardous_ROC.y[,ROCPlot.y.classindex], use.names=FALSE)
  lines(ROCPlot.x, ROCPlot.y, col=plot.models.color[ROCPlot.x.classindex-1],lwd=1)

  
}
legend("right", title="models",legend=ROCPlot.y.class,lwd=5, col=plot.models.color,
        horiz=FALSE)
dev.off()
img_name_plot <- paste("IMG_asteroids_model_SVM_", "Hazardous_KFOLD_ROC_wcutoff", ".png", sep = "")
png(img_name_plot,res = 800, height = 10, width = 15, unit='in')
ROCPlot.x.class = colnames(svm.Hazardous_ROC.x)
ROCPlot.y.class = colnames(svm.Hazardous_ROC.y)
plot.new()
title(main="SVM Hazardous ROC", xlab="Sensitivity - True Positive Rate", ylab="Specificity - False Positive Rate")
for (ROCPlot.x.classindex in 2:length(ROCPlot.x.class)){
  ROCPlot.name = ROCPlot.x.class[ROCPlot.x.classindex]
  ROCPlot.y.classindex = which(ROCPlot.y.class == ROCPlot.x.class[ROCPlot.x.classindex])
  
  ROCPlot.x = unlist(svm.Hazardous_ROC.x[,ROCPlot.x.classindex], use.names=FALSE)
  ROCPlot.y = unlist(svm.Hazardous_ROC.y[,ROCPlot.y.classindex], use.names=FALSE)
  lines(ROCPlot.x, ROCPlot.y, col=plot.models.color[ROCPlot.x.classindex-1],lwd=1)
  points(svm.Hazardous$CutOffOpt[ROCPlot.name][[1]][1], 1, type="o", col=plot.models.color[ROCPlot.x.classindex-1], pch="o", lty=1, ylim=c(0,110) )
  
}
legend("right", title="models",legend=ROCPlot.y.class,lwd=5, col=plot.models.color,
       horiz=FALSE)
dev.off()



img_name_plot <- paste("IMG_asteroids_model_SVM_", "Hazardous_KFOLD_performance_plot_A", ".png", sep = "")
png(img_name_plot,res = 800, height = 10, width = 15, unit='in')
  par(mfrow=c(2,2)) 
  boxplot(svm.Hazardous$Accuracy[2:end_table],vertical = TRUE, pch=19, las = 2, col = plot.models.color,main="Accuracy")  
  boxplot(svm.Hazardous$MacroSensitivity[2:end_table],vertical = TRUE, pch=19, las = 2, col = plot.models.color,main="MacroSensitivity")  
  dev.off()
img_name_plot <- paste("IMG_asteroids_model_SVM_", "Hazardous_KFOLD_performance_plot_B", ".png", sep = "")
png(img_name_plot,res = 800, height = 10, width = 15, unit='in')
  par(mfrow=c(2,2))  
  boxplot(svm.Hazardous$MacroSpecificity[2:end_table], vertical = TRUE, pch=19, las = 2, col = plot.models.color,main="MacroSpecificity")  
  boxplot(svm.Hazardous$MacroPrecision[2:end_table], vertical = TRUE, pch=19, las = 2, col = plot.models.color,main="MacroPrecision")  
  dev.off()
img_name_plot <- paste("IMG_asteroids_model_SVM_", "Hazardous_KFOLD_performance_plot_C", ".png", sep = "")
png(img_name_plot,res = 800, height = 10, width = 15, unit='in')
  par(mfrow=c(2,2))  
  boxplot(svm.Hazardous$MacroRecall[2:end_table],vertical = TRUE, pch=19, las = 2, col = plot.models.color,main="MacroRecall")  
  boxplot(svm.Hazardous$MacroF1[2:end_table],vertical = TRUE, pch=19, las = 2, col = plot.models.color,main="MacroF1")  
  dev.off()

img_name_plot <- paste("PDF_asteroids_model_SVM_", "Hazardous_KFOLD_performance_tdist", ".pdf", sep = "")
pdf(img_name_plot, height = 20, width = 46)
  grid.table(t(svm.Hazardous.All[2:length(svm.Hazardous.All)]))
  dev.off()
    
#rm(svm.Hazardous, svm.Hazardous.All)

#Hazardous class
#a small cost creates a large margin (a soft margin) and allows more misclassifications;
#a large cost creates a narrow margin (a hard margin) and allows few misclassifications
#this is a soft margin svm


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
hyper.cost <- 1
hyper.kernel <- 'polynomial'
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

#Poly  cost 1 
hyper.cost <- 1
hyper.kernel <- 'radial'
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


folds.number = 5
kernel_list <- c('linear','polynomial','radial')
C_list <- c(1,10)

folds <- split(asteroids_split$train, cut(sample(1:nrow(asteroids_split$train)), folds.number))
rm(folds.number)
svm.Classification <- list()

mx = matrix(NA, nrow = length(folds))

svm.Classification$Accuracy = data.frame(mx)
svm.Classification$MacroSensitivity <- data.frame(mx)
svm.Classification$MacroSpecificity <- data.frame(mx)
svm.Classification$MacroPrecision <- data.frame(mx)
svm.Classification$MacroRecall <- data.frame(mx)
svm.Classification$MacroF1 <- data.frame(mx)

#Amor
svm.Classification$Amor.AUC <- data.frame(mx)
svm.Classification$Amor.CutOffOpt <- data.frame(mx)
svm.Classification_ROC.Amor.x <- matrix()
svm.Classification_ROC.Amor.y <- matrix()
#Apohele
svm.Classification$Apohele.AUC <- data.frame(mx)
svm.Classification$Apohele.CutOffOpt <- data.frame(mx)
svm.Classification_ROC.Apohele.x <- matrix()
svm.Classification_ROC.Apohele.y <- matrix()
#Apollo
svm.Classification$Apollo.AUC <- data.frame(mx)
svm.Classification$Apollo.CutOffOpt <- data.frame(mx)
svm.Classification_ROC.Apollo.x <- matrix()
svm.Classification_ROC.Apollo.y <- matrix()
#Aten
svm.Classification$Aten.AUC <- data.frame(mx)
svm.Classification$Aten.CutOffOpt <- data.frame(mx)
svm.Classification_ROC.Aten.x <- matrix()
svm.Classification_ROC.Aten.y <- matrix()

svm.Classification_ROC.name <- c(NA)

mx = matrix(NA, nrow = 6)
svm.Classification.All <- data.frame(mx)
svm.Classification.All["Performance"] = c("Accuracy","MacroSensitivity","MacroSpecificity","MacroPrecision","MacroRecall","MacroF1")
rm(mx)

mx = matrix(NA, nrow = 8)
svm.Classification.ROC.All <- data.frame(mx)
svm.Classification.ROC.All["Performance"] = c("Amor AUC","Amor CutOffOpt","Apohele AUC","Apohele CutOffOpt","Apollo AUC","Apollo CutOffOpt","Aten AUC","Aten CutOffOpt")
rm(mx)

for (j in 1:length(kernel_list)) {
  hyper.kernel <- kernel_list[j]
  
  for (l in 1:length(C_list)) {
    hyper.cost <- C_list[l]
    svm.Classification.stats <- list()
    
    svm.Classification.stats.roc.pred.prob <- list()
    svm.Classification.stats.roc.thruth <- list()
    
    folds_confusion <- NULL

    for (i in 1:length(folds)) {
      fold.valid <- ldply(folds[i], data.frame)
      fold.valid <- fold.valid[, !names(fold.valid) %in% c(".id")]
      fold.train <- ldply(folds[-i], data.frame)
      fold.train <- fold.train[, !names(fold.train) %in% c(".id")]
      
      
      print(paste(as.character(i), hyper.kernel , as.character(hyper.cost), sep = " "))
      svm.Classification.model = svm(Classification ~ Orbit.Axis..AU. + Orbit.Eccentricity + Orbit.Inclination..deg. + Perihelion.Argument..deg. + Node.Longitude..deg. + Mean.Anomoly..deg. + Perihelion.Distance..AU. + Aphelion.Distance..AU. + Orbital.Period..yr. + Minimum.Orbit.Intersection.Distance..AU. + Asteroid.Magnitude,
                                     data=fold.train, kernel=hyper.kernel, cost=hyper.cost,probability = TRUE, type="C-classification" )
      svm.Classification.pred = predict(svm.Classification.model, fold.valid)
      
      svm.Classification.confusion_matrix_multiclass = confusionMatrix(
        data=svm.Classification.pred, reference=fold.valid$Classification, mode = "prec_recall") 

      folds_confusion <- Confusion_Sum(folds_confusion, reference=fold.valid$Classification, data=svm.Classification.pred)
      
      confusion.multi = svm.Classification.confusion_matrix_multiclass$byClass
      
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
      
      MacroSensitivity = mean(c(sens_Amor, sens_Apohele, sens_Apollo, sens_Aten),  na.rm = TRUE)
      MacroSpecificity = mean(c(spec_Amor, spec_Apohele, spec_Apollo, spec_Aten),  na.rm = TRUE)
      MacroPrecision = mean(c(prec_Amor, prec_Apohele, prec_Apollo, prec_Aten),  na.rm = TRUE)
      MacroRecall = mean(c(recal_Amor, recal_Apohele, recal_Apollo, recal_Aten),  na.rm = TRUE)
      MacroF1 =  mean(c(f1_Amor,  f1_Apohele,  f1_Apollo,  f1_Aten),  na.rm = TRUE)
      
      svm.Classification.stats$Accuracy    = append(svm.Classification.stats$Accuracy, svm.Classification.confusion_matrix_multiclass$overall["Accuracy"])
      svm.Classification.stats$MacroSensitivity = append(svm.Classification.stats$MacroSensitivity, MacroSensitivity)
      svm.Classification.stats$MacroSpecificity = append(svm.Classification.stats$MacroSpecificity, MacroSpecificity)
      svm.Classification.stats$MacroPrecision = append(svm.Classification.stats$MacroPrecision, MacroPrecision)
      svm.Classification.stats$MacroRecall = append(svm.Classification.stats$MacroRecall, MacroRecall)
      svm.Classification.stats$MacroF1 = append(svm.Classification.stats$MacroF1, MacroF1)
      
      #ROC
      #ROC
      ROCFun.pred.fold = predict(svm.Classification.model,fold.valid,  probability=TRUE)
      ROCFun.pred.prob.fold = attr(ROCFun.pred.fold, "probabilities")
      
      svm.Classification.stats.roc.pred.prob = rbind(svm.Classification.stats.roc.pred.prob,ROCFun.pred.prob.fold)
      svm.Classification.stats.roc.thruth = append(svm.Classification.stats.roc.thruth, as.factor(fold.valid$Classification))
      
      rm(svm.Classification.model, svm.Classification.pred)
      rm(svm.Classification.confusion_matrix_true,svm.Classification.confusion_matrix_false,prec_true,recal_true,f1_true,prec_false,recal_false,f1_false,MacroPrecision,MacroRecall,MacroF1)
    }
    
    svm.Classification.roc.Amor = ROCFunction.MULTI(svm.Classification.stats.roc.pred.prob,svm.Classification.stats.roc.thruth,"Amor Asteroid")
    svm.Classification.roc.Apohele = ROCFunction.MULTI(svm.Classification.stats.roc.pred.prob,svm.Classification.stats.roc.thruth,"Apohele Asteroid")
    svm.Classification.roc.Apollo = ROCFunction.MULTI(svm.Classification.stats.roc.pred.prob,svm.Classification.stats.roc.thruth,"Apollo Asteroid")
    svm.Classification.roc.Aten = ROCFunction.MULTI(svm.Classification.stats.roc.pred.prob,svm.Classification.stats.roc.thruth,"Aten Asteroid")
    
    svm.name <- paste("Clas",hyper.kernel,as.character(hyper.cost),sep="_")
    
    img_name_plot <- paste("IMG_asteroids_model_SWM_", svm.name ,"_confusion" ,".png", sep = "")
      png(img_name_plot)
      grid.table(folds_confusion)
      dev.off()
    
    svm.Classification$Accuracy[svm.name] <- svm.Classification.stats$Accuracy
    svm.Classification$MacroSensitivity[svm.name] <- svm.Classification.stats$MacroSensitivity
    svm.Classification$MacroSpecificity[svm.name] <- svm.Classification.stats$MacroSpecificity
    svm.Classification$MacroPrecision[svm.name] <- svm.Classification.stats$MacroPrecision
    svm.Classification$MacroRecall[svm.name] <- svm.Classification.stats$MacroRecall
    svm.Classification$MacroF1[svm.name] <- svm.Classification.stats$MacroF1
    #roc
    svm.Classification$Amor.AUC[svm.name] <- svm.Classification.roc.Amor$auc
    svm.Classification$Amor.CutOffOpt[svm.name] <- svm.Classification.roc.Amor$optcut
    svm.Classification$Apohele.AUC[svm.name] <- svm.Classification.roc.Apohele$auc
    svm.Classification$Apohele.CutOffOpt[svm.name] <- svm.Classification.roc.Apohele$optcut
    svm.Classification$Apollo.AUC[svm.name] <- svm.Classification.roc.Apollo$auc
    svm.Classification$Apollo.CutOffOpt[svm.name] <- svm.Classification.roc.Apollo$optcut
    svm.Classification$Aten.AUC[svm.name] <- svm.Classification.roc.Aten$auc
    svm.Classification$Aten.CutOffOpt[svm.name] <- svm.Classification.roc.Aten$optcut
    
    # ROC DATA FRAMES
    svm.Classification_ROC.name = c(svm.Classification_ROC.name, svm.name)
    svm.Classification_ROC.Amor.x <- cbindX(svm.Classification_ROC.Amor.x, data.frame(svm.Classification.roc.Amor$x.value))
    colnames(svm.Classification_ROC.Amor.x) <- svm.Classification_ROC.name
    svm.Classification_ROC.Amor.y <- cbindX(svm.Classification_ROC.Amor.y, data.frame(svm.Classification.roc.Amor$y.value))
    colnames(svm.Classification_ROC.Amor.y) <- svm.Classification_ROC.name
    
    svm.Classification_ROC.Apohele.x <- cbindX(svm.Classification_ROC.Apohele.x, data.frame(svm.Classification.roc.Apohele$x.value))
    colnames(svm.Classification_ROC.Apohele.x) <- svm.Classification_ROC.name
    svm.Classification_ROC.Apohele.y <- cbindX(svm.Classification_ROC.Apohele.y, data.frame(svm.Classification.roc.Apohele$y.value))
    colnames(svm.Classification_ROC.Apohele.y) <- svm.Classification_ROC.name
    
    svm.Classification_ROC.Apollo.x <- cbindX(svm.Classification_ROC.Apollo.x, data.frame(svm.Classification.roc.Apollo$x.value))
    colnames(svm.Classification_ROC.Apollo.x) <- svm.Classification_ROC.name
    svm.Classification_ROC.Apollo.y <- cbindX(svm.Classification_ROC.Apollo.y, data.frame(svm.Classification.roc.Apollo$y.value))
    colnames(svm.Classification_ROC.Apollo.y) <- svm.Classification_ROC.name
    
    svm.Classification_ROC.Aten.x <- cbindX(svm.Classification_ROC.Aten.x, data.frame(svm.Classification.roc.Aten$x.value))
    colnames(svm.Classification_ROC.Aten.x) <- svm.Classification_ROC.name
    svm.Classification_ROC.Aten.y <- cbindX(svm.Classification_ROC.Aten.y, data.frame(svm.Classification.roc.Aten$y.value))
    colnames(svm.Classification_ROC.Aten.y) <- svm.Classification_ROC.name
    
    
    tdist <- list()
    tdist_name <- paste("Class ",hyper.kernel,as.character(hyper.cost),sep=" ")
    tdist_val = confidence_interval(as.vector(svm.Classification.stats$Accuracy),0.95)
    tdist$acc <- paste(as.character(round(tdist_val[2],4))," ± ",as.character(round(tdist_val[1],4)))
    
    tdist_val = confidence_interval(as.vector(svm.Classification.stats$MacroSensitivity),0.95)
    tdist$sens <- paste(as.character(round(tdist_val[2],4))," ± ",as.character(round(tdist_val[1],4)))
    
    tdist_val = confidence_interval(as.vector(svm.Classification.stats$MacroSpecificity),0.95)
    tdist$spec <- paste(as.character(round(tdist_val[2],4))," ± ",as.character(round(tdist_val[1],4)))
    
    tdist_val = confidence_interval(as.vector(svm.Classification.stats$MacroPrecision),0.95)
    tdist$prec <- paste(as.character(round(tdist_val[2],4))," ± ",as.character(round(tdist_val[1],4)))
    
    tdist_val = confidence_interval(as.vector(svm.Classification.stats$MacroRecall),0.95)
    tdist$rec <- paste(as.character(round(tdist_val[2],4))," ± ",as.character(round(tdist_val[1],4)))
    
    tdist_val = confidence_interval(as.vector(svm.Classification.stats$MacroF1),0.95)
    tdist$f1 <- paste(as.character(round(tdist_val[2],4))," ± ",as.character(round(tdist_val[1],4)))
    
    
    svm.Classification.All[tdist_name] <- c(tdist$acc,tdist$sens,tdist$spec,tdist$prec,tdist$rec,tdist$f1)
    
    
    
    rdist_name <- paste("Class ",hyper.kernel,as.character(hyper.cost),sep=" ")  
    rdist <- list()
    rdist_val = svm.Classification.roc.Amor$auc
    rdist$Amor.auc <- paste(as.character(round(rdist_val,8)))
    rdist_val = svm.Classification.roc.Amor$optcut
    rdist$Amor.optcut <- paste(as.character(round(rdist_val,5)))
    
    rdist_val = svm.Classification.roc.Apohele$auc
    rdist$Apohele.auc <- paste(as.character(round(rdist_val,8)))
    rdist_val = svm.Classification.roc.Apohele$optcut
    rdist$Apohele.optcut <- paste(as.character(round(rdist_val,5)))
    
    rdist_val = svm.Classification.roc.Apollo$auc
    rdist$Apollo.auc <- paste(as.character(round(rdist_val,8)))
    rdist_val = svm.Classification.roc.Apollo$optcut
    rdist$Apollo.optcut <- paste(as.character(round(rdist_val,5)))
    
    rdist_val = svm.Classification.roc.Aten$auc
    rdist$Aten.auc <- paste(as.character(round(rdist_val,8)))
    rdist_val = svm.Classification.roc.Aten$optcut
    rdist$Aten.optcut <- paste(as.character(round(rdist_val,5)))
    svm.Classification.ROC.All[rdist_name] <- c(rdist$Amor.auc,rdist$Amor.optcut,rdist$Apohele.auc,rdist$Apohele.optcut,rdist$Apollo.auc,rdist$Apollo.optcut,rdist$Aten.auc,rdist$Aten.optcut)
    
  }
}
end_table <- length(svm.Classification$Accuracy)
plot.models.color = rainbow(end_table-1)

img_name_plot <- paste("IMG_asteroids_model_SVM_", "Classification_KFOLD_ROC", ".png", sep = "")
png(img_name_plot,res = 800, height = 10, width = 15, unit='in')
par(mfrow=c(2,2))  
# Amor
plot.new()
ROCPlot.x.class = colnames(svm.Classification_ROC.Amor.x)
ROCPlot.y.class = colnames(svm.Classification_ROC.Amor.y)

title(main="ROC Class: Amor", xlab="Sensitivity - True Positive Rate", ylab="Specificity - False Positive Rate")
for (ROCPlot.x.classindex in 2:length(ROCPlot.x.class)){
  ROCPlot.name = ROCPlot.x.class[ROCPlot.x.classindex]
  ROCPlot.y.classindex = which(ROCPlot.y.class == ROCPlot.x.class[ROCPlot.x.classindex])
  
  ROCPlot.x = unlist(svm.Classification_ROC.Amor.x[,ROCPlot.x.classindex], use.names=FALSE)
  ROCPlot.y = unlist(svm.Classification_ROC.Amor.x[,ROCPlot.y.classindex], use.names=FALSE)
  lines(ROCPlot.x, ROCPlot.y, col=plot.models.color[ROCPlot.x.classindex-1],lwd=1)    
}
legend("right", title="models",legend=ROCPlot.y.class,lwd=5, col=plot.models.color,horiz=FALSE)
#Apohele
plot.new()
ROCPlot.x.class = colnames(svm.Classification_ROC.Apohele.x)
ROCPlot.y.class = colnames(svm.Classification_ROC.Apohele.y)

title(main="ROC Class: Apohele", xlab="Sensitivity - True Positive Rate", ylab="Specificity - False Positive Rate")
for (ROCPlot.x.classindex in 2:length(ROCPlot.x.class)){
  ROCPlot.name = ROCPlot.x.class[ROCPlot.x.classindex]
  ROCPlot.y.classindex = which(ROCPlot.y.class == ROCPlot.x.class[ROCPlot.x.classindex])
  
  ROCPlot.x = unlist(svm.Classification_ROC.Apohele.x[,ROCPlot.x.classindex], use.names=FALSE)
  ROCPlot.y = unlist(svm.Classification_ROC.Apohele.y[,ROCPlot.y.classindex], use.names=FALSE)
  lines(ROCPlot.x, ROCPlot.y, col=plot.models.color[ROCPlot.x.classindex-1],lwd=1)    
}
legend("right", title="models",legend=ROCPlot.y.class,lwd=5, col=plot.models.color,horiz=FALSE)
#Apollo
plot.new()
ROCPlot.x.class = colnames(svm.Classification_ROC.Apollo.x)
ROCPlot.y.class = colnames(svm.Classification_ROC.Apollo.y)

title(main="ROC Class: Apollo", xlab="Sensitivity - True Positive Rate", ylab="Specificity - False Positive Rate")
for (ROCPlot.x.classindex in 2:length(ROCPlot.x.class)){
  ROCPlot.name = ROCPlot.x.class[ROCPlot.x.classindex]
  ROCPlot.y.classindex = which(ROCPlot.y.class == ROCPlot.x.class[ROCPlot.x.classindex])
  
  ROCPlot.x = unlist(svm.Classification_ROC.Apollo.x[,ROCPlot.x.classindex], use.names=FALSE)
  ROCPlot.y = unlist(svm.Classification_ROC.Apollo.y[,ROCPlot.y.classindex], use.names=FALSE)
  lines(ROCPlot.x, ROCPlot.y, col=plot.models.color[ROCPlot.x.classindex-1],lwd=1)    
}
legend("right", title="models",legend=ROCPlot.y.class,lwd=5, col=plot.models.color,horiz=FALSE)
#Aten
plot.new()
ROCPlot.x.class = colnames(svm.Classification_ROC.Aten.x)
ROCPlot.y.class = colnames(svm.Classification_ROC.Aten.y)

title(main="ROC Class: Aten", xlab="Sensitivity - True Positive Rate", ylab="Specificity - False Positive Rate")
for (ROCPlot.x.classindex in 2:length(ROCPlot.x.class)){
  ROCPlot.name = ROCPlot.x.class[ROCPlot.x.classindex]
  ROCPlot.y.classindex = which(ROCPlot.y.class == ROCPlot.x.class[ROCPlot.x.classindex])
  
  ROCPlot.x = unlist(svm.Classification_ROC.Aten.x[,ROCPlot.x.classindex], use.names=FALSE)
  ROCPlot.y = unlist(svm.Classification_ROC.Aten.y[,ROCPlot.y.classindex], use.names=FALSE)
  lines(ROCPlot.x, ROCPlot.y, col=plot.models.color[ROCPlot.x.classindex-1],lwd=1)    
}
legend("right", title="models",legend=ROCPlot.y.class,lwd=5, col=plot.models.color,horiz=FALSE)
dev.off()



img_name_plot <- paste("IMG_asteroids_model_SVM_", "Classification_KFOLD_performance_plot_A", ".png", sep = "")
png(img_name_plot,res = 800, height = 10, width = 15, unit='in')
par(mfrow=c(2,2)) 
boxplot(svm.Classification$Accuracy[2:end_table],vertical = TRUE, pch=19, las = 2, col = plot.models.color,main="Accuracy")  
boxplot(svm.Classification$MacroSensitivity[2:end_table],vertical = TRUE, pch=19, las = 2, col = plot.models.color,main="MacroSensitivity")  
dev.off()
img_name_plot <- paste("IMG_asteroids_model_SVM_", "Classification_KFOLD_performance_plot_B", ".png", sep = "")
png(img_name_plot,res = 800, height = 10, width = 15, unit='in')
par(mfrow=c(2,2))  
boxplot(svm.Classification$MacroSpecificity[2:end_table], vertical = TRUE, pch=19, las = 2, col = plot.models.color,main="MacroSpecificity")  
boxplot(svm.Classification$MacroPrecision[2:end_table], vertical = TRUE, pch=19, las = 2, col = plot.models.color,main="MacroPrecision")  
dev.off()
img_name_plot <- paste("IMG_asteroids_model_SVM_", "Classification_KFOLD_performance_plot_C", ".png", sep = "")
png(img_name_plot,res = 800, height = 10, width = 15, unit='in')
par(mfrow=c(2,2))  
boxplot(svm.Classification$MacroRecall[2:end_table],vertical = TRUE, pch=19, las = 2, col = plot.models.color,main="MacroRecall")  
boxplot(svm.Classification$MacroF1[2:end_table],vertical = TRUE, pch=19, las = 2, col = plot.models.color,main="MacroF1")  
dev.off()

img_name_plot <- paste("PDF_asteroids_model_SVM_", "Classification_KFOLD_performance_tdist", ".pdf", sep = "")
pdf(img_name_plot, height = 20, width = 46)
grid.table(t(svm.Classification.All[2:length(svm.Classification.All)]))
dev.off()

img_name_plot <- paste("PDF_asteroids_model_SVM_", "Classification_KFOLD_performance_ROC", ".pdf", sep = "")
pdf(img_name_plot, height = 20, width = 46)
grid.table(t(svm.Classification.ROC.All[2:length(svm.Classification.ROC.All)]))
dev.off()