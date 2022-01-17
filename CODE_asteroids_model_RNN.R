setwd("~/Github/AsteroidsClassification")

library(plyr)
library(gridExtra)
library(neuralnet)
library(caret)
library(tidyverse)
library(ROCR)
library(gdata)
library(reshape)
library(devtools)
source_url('https://gist.githubusercontent.com/fawda123/7471137/raw/466c1474d0a505ff044412703516c34f1a4684a5/nnet_plot_update.r')

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

Confusion_Sum <- function(cm_global, data, reference){
  cm_fold <- data.frame(Predicted=data,Reality=reference)
  cm_global <- rbind(cm_global, cm_fold)
  return(cm_global)
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


ROCFunction.BIN <- function(ROCFun.pred.prob, testLabels){
  #ROCFun.pred = predict(model, testset,  probability=TRUE)
  #ROCFun.pred.prob = attr(ROCFun.pred, "probabilities")
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


actFun <- list()
actFun$sigmoid <- function(x) {
  1 / (1 + exp(-x))
}
actFun$relu <- function(x) {
  max(0,x)
}
actFun$softplus <- function(x){ 
  log(1+exp(x))
}
#load dataset RObject as asteroids_split
load("DATA_asteroids_dataset_split_0.7.RData")

folds.number = 5

hiddenLayers_list = list(c(7))#,c(14,7))#,c(5),c(10,5))
actFun_list = list("tanh","logistic")#,actFun$softplus,actFun$relu,actFun$sigmoid)
actFun_listname = list("tanh","logistic")#,"softplus","relu","sigmoid")
learningRate_list = list(1e-4)
lossFun_list = list(NULL)
lossFun_listname = list("Loss A")
stepmax_val = 1e5

folds <- split(asteroids_split$train, cut(sample(1:nrow(asteroids_split$train)), folds.number))
rm(folds.number)

#Hazardous
rnn.Hazardous <- list()

mx = matrix(NA, nrow = length(folds))

rnn.Hazardous$Accuracy = data.frame(mx)
rnn.Hazardous$MacroSensitivity <- data.frame(mx)
rnn.Hazardous$MacroSpecificity <- data.frame(mx)
rnn.Hazardous$MacroPrecision <- data.frame(mx)
rnn.Hazardous$MacroRecall <- data.frame(mx)
rnn.Hazardous$MacroF1 <- data.frame(mx)
rnn.Hazardous$AUC <- data.frame(mx)
rnn.Hazardous$CutOffOpt <- data.frame(mx)
rnn.Hazardous_ROC.x <- matrix()
rnn.Hazardous_ROC.y <- matrix()
rnn.Hazardous_ROC.name <- c(NA)

mx = matrix(NA, nrow = 8)
rnn.Hazardous.All <- data.frame(mx)
rnn.Hazardous.All["Performance"] = c("Accuracy","MacroSensitivity","MacroSpecificity","MacroPrecision","MacroRecall","MacroF1","AUC","OptimalCutOff")
rm(mx)

rnn.Hazardous.model <- NULL
for (hl in 1:length(hiddenLayers_list)) {
  for (ac in 1:length(actFun_list)) {
    for (lrr in 1:length(learningRate_list)) {
      for (lf in 1:length(lossFun_list)) {
        hiddenL_val = hiddenLayers_list[[hl]]
        actFun_val = actFun_list[[ac]]
        actFun_name = actFun_listname[[ac]]
        lr_val = learningRate_list[[lrr]]
        loss_f = lossFun_list[[lf]]
        loss_name = lossFun_listname[[lf]]
        rnn.Hazardous.stats <- list()
        
        rnn.model_printname = paste("Hazardous"," ACT:",actFun_name," LOSS:",loss_name," HIDDENL: (", paste(hiddenL_val, collapse = "+"),") LR:",as.character(lr_val),sep=" ")
        rnn.name = paste("Hazardous",actFun_name,loss_name,paste(hiddenL_val, collapse = "+"),as.character(lr_val),sep="_")
        
        rnn.Hazardous.stats.roc.pred.prob <- list()
        rnn.Hazardous.stats.roc.thruth <- list()
        
        folds_confusion <- NULL
        
        print(rnn.model_printname)        
        for (i in 1:length(folds)) {
          fold.valid <- ldply(folds[i], data.frame)
          fold.valid <- fold.valid[, !names(fold.valid) %in% c(".id")]
          fold.train <- ldply(folds[-i], data.frame)
          fold.train <- fold.train[, !names(fold.train) %in% c(".id")]
          
          #manca LOSS loss_f
          rnn.Hazardous.model <- neuralnet(Hazardous ~ Orbit.Axis..AU._scaled_scaled + Orbit.Eccentricity_scaled + Orbit.Inclination..deg._scaled + Perihelion.Argument..deg._scaled + Node.Longitude..deg._scaled + Mean.Anomoly..deg._scaled + Perihelion.Distance..AU._scaled + Aphelion.Distance..AU._scaled + Orbital.Period..yr._scaled + Minimum.Orbit.Intersection.Distance..AU._scaled + Asteroid.Magnitude_scaled,
                              fold.train[1:200,],
                              hidden = hiddenL_val,
                              act.fct = actFun_val,
                              learningrate = lr_val,
                              #lossFunction = loss_f,
                              stepmax = 1e7,
                              linear.output = FALSE)
          
          rnn.Hazardous.pred = predict(rnn.Hazardous.model, fold.valid)[, 1] > 0.5
          
          folds_confusion <- Confusion_Sum(folds_confusion, reference=as.factor(fold.valid$Hazardous), data=as.factor(rnn.Hazardous.pred))
          
          rnn.Hazardous.confusion_matrix_true = confusionMatrix(
            data=as.factor(rnn.Hazardous.pred), reference=as.factor(fold.valid$Hazardous), positive="TRUE", mode = "prec_recall") 
          rnn.Hazardous.confusion_matrix_false = confusionMatrix(
            data=as.factor(rnn.Hazardous.pred), reference=as.factor(fold.valid$Hazardous), positive="FALSE", mode = "prec_recall") 
          
          sens_true = rnn.Hazardous.confusion_matrix_true$byClass["Sensitivity"]
          spec_true = rnn.Hazardous.confusion_matrix_true$byClass["Specificity"]
          prec_true = rnn.Hazardous.confusion_matrix_true$byClass["Precision"]
          recal_true = rnn.Hazardous.confusion_matrix_true$byClass["Recall"]
          f1_true   = rnn.Hazardous.confusion_matrix_true$byClass["F1"]
          
          sens_false = rnn.Hazardous.confusion_matrix_false$byClass["Sensitivity"]
          spec_false = rnn.Hazardous.confusion_matrix_false$byClass["Specificity"]
          prec_false = rnn.Hazardous.confusion_matrix_false$byClass["Precision"]
          recal_false = rnn.Hazardous.confusion_matrix_false$byClass["Recall"]
          f1_false = rnn.Hazardous.confusion_matrix_false$byClass["F1"]
          
          MacroSensitivity = mean(c(sens_true, sens_false),  na.rm = TRUE)
          MacroSpecificity = mean(c(spec_true, spec_false),  na.rm = TRUE)
          MacroPrecision = mean(c(prec_true, prec_false),  na.rm = TRUE)
          MacroRecall = mean(c(recal_true, recal_false),  na.rm = TRUE)
          MacroF1 = mean(c(f1_true, f1_false),  na.rm = TRUE)
          
          rnn.Hazardous.stats$Accuracy    = append(rnn.Hazardous.stats$Accuracy, rnn.Hazardous.confusion_matrix_true$overall["Accuracy"])
          rnn.Hazardous.stats$MacroSensitivity = append(rnn.Hazardous.stats$MacroSensitivity, MacroSensitivity)
          rnn.Hazardous.stats$MacroSpecificity = append(rnn.Hazardous.stats$MacroSpecificity, MacroSpecificity)
          rnn.Hazardous.stats$MacroPrecision = append(rnn.Hazardous.stats$MacroPrecision, MacroPrecision)
          rnn.Hazardous.stats$MacroRecall = append(rnn.Hazardous.stats$MacroRecall, MacroRecall)
          rnn.Hazardous.stats$MacroF1 = append(rnn.Hazardous.stats$MacroF1, MacroF1)
          
          # ROC
          ROCFun.pred.prob.fold = predict(rnn.Hazardous.model,fold.valid,  probability=TRUE)
          rnn.Hazardous.stats.roc.pred.prob = rbind(rnn.Hazardous.stats.roc.pred.prob,ROCFun.pred.prob.fold )
          rnn.Hazardous.stats.roc.thruth = append(rnn.Hazardous.stats.roc.thruth, as.factor(fold.valid$Hazardous))
        }
        
        #end kfold
        
        img_name_plot <- paste("IMG_asteroids_model_RNN_", rnn.name ,"_confusion" ,".png", sep = "")
          png(img_name_plot,res = 800, height = 10, width = 15, unit='in')
          grid.table(table(folds_confusion))
          dev.off()
        
        rnn.Hazardous.roc = ROCFunction.BIN(rnn.Hazardous.stats.roc.pred.prob,rnn.Hazardous.stats.roc.thruth)
        
        rnn.Hazardous$Accuracy[rnn.name] <- rnn.Hazardous.stats$Accuracy
        rnn.Hazardous$MacroSensitivity[rnn.name] <- rnn.Hazardous.stats$MacroSensitivity
        rnn.Hazardous$MacroSpecificity[rnn.name] <- rnn.Hazardous.stats$MacroSpecificity
        rnn.Hazardous$MacroPrecision[rnn.name] <- rnn.Hazardous.stats$MacroPrecision
        rnn.Hazardous$MacroRecall[rnn.name] <- rnn.Hazardous.stats$MacroRecall
        rnn.Hazardous$MacroF1[rnn.name] <- rnn.Hazardous.stats$MacroF1
        rnn.Hazardous$AUC[rnn.name] <- rnn.Hazardous.roc$auc
        rnn.Hazardous$CutOffOpt[rnn.name] <- rnn.Hazardous.roc$optcut
        rnn.Hazardous_ROC.name = c(rnn.Hazardous_ROC.name, rnn.name)
        rnn.Hazardous_ROC.x <- cbindX(rnn.Hazardous_ROC.x, data.frame(rnn.Hazardous.roc$x.value))
        colnames(rnn.Hazardous_ROC.x) <- rnn.Hazardous_ROC.name
        rnn.Hazardous_ROC.y <- cbindX(rnn.Hazardous_ROC.y, data.frame(rnn.Hazardous.roc$y.value))
        colnames(rnn.Hazardous_ROC.y) <- rnn.Hazardous_ROC.name
        
        tdist <- list()
        tdist_name <-  paste("Hazardous",actFun_name,loss_name,"(", paste(hiddenL_val, collapse = "+"),")",as.character(lr_val),sep=" ")
        
        tdist_val = confidence_interval(as.vector(rnn.Hazardous.stats$Accuracy),0.95)
        tdist$acc <- paste(as.character(round(tdist_val[2],4))," � ",as.character(round(tdist_val[1],4)))
        
        tdist_val = confidence_interval(as.vector(rnn.Hazardous.stats$MacroSensitivity),0.95)
        tdist$sens <- paste(as.character(round(tdist_val[2],4))," � ",as.character(round(tdist_val[1],4)))
        
        tdist_val = confidence_interval(as.vector(rnn.Hazardous.stats$MacroSpecificity),0.95)
        tdist$spec <- paste(as.character(round(tdist_val[2],4))," � ",as.character(round(tdist_val[1],4)))
        
        tdist_val = confidence_interval(as.vector(rnn.Hazardous.stats$MacroPrecision),0.95)
        tdist$prec <- paste(as.character(round(tdist_val[2],4))," � ",as.character(round(tdist_val[1],4)))
        
        tdist_val = confidence_interval(as.vector(rnn.Hazardous.stats$MacroRecall),0.95)
        tdist$rec <- paste(as.character(round(tdist_val[2],4))," � ",as.character(round(tdist_val[1],4)))
        
        tdist_val = confidence_interval(as.vector(rnn.Hazardous.stats$MacroF1),0.95)
        tdist$f1 <- paste(as.character(round(tdist_val[2],4))," � ",as.character(round(tdist_val[1],4)))
        
        
        tdist_val = rnn.Hazardous.roc$auc
        tdist$auc <- paste(as.character(round(tdist_val,8)))
        tdist_val = rnn.Hazardous.roc$optcut
        tdist$cutoff <- paste(as.character(round(tdist_val,5)))
        
        rnn.Hazardous.All[tdist_name] <- c(tdist$acc,tdist$sens,tdist$spec,tdist$prec,tdist$rec,tdist$f1,tdist$auc,tdist$cutoff)
        
      }
    }
  }
  img_name_plot <- paste("IMG_asteroids_model_RNN_", "Hazardous_HIDDEN_LAYER_",paste(hiddenL_val, collapse = "+") ,".png", sep = "")
  png(img_name_plot, res = 800, height = 10, width = 15, unit='in')
  plot.nnet(rnn.Hazardous.model)
    dev.off()
  
  img_name_plot <- paste("IMG_asteroids_model_RNN_", "Hazardous_generalized_weights", ".png", sep = "")
  png(img_name_plot, res = 800, height = 10, width = 15, unit='in')
    par(mfrow=c(2,2))
    gwplot(rnn.Hazardous.model,selected.covariate="Minimum.Orbit.Intersection.Distance..AU._scaled")
    gwplot(rnn.Hazardous.model,selected.covariate="Asteroid.Magnitude_scaled")
    gwplot(rnn.Hazardous.model,selected.covariate="Perihelion.Distance..AU._scaled")
    gwplot(rnn.Hazardous.model,selected.covariate="Orbit.Axis..AU._scaled_scaled")
    
    dev.off()
}

end_table <- length(rnn.Hazardous$Accuracy)
plot.models.color = rainbow(end_table-1)

img_name_plot <- paste("IMG_asteroids_model_RNN_", "Hazardous_KFOLD_ROC", ".png", sep = "")
png(img_name_plot,res = 800, height = 10, width = 15, unit='in')
ROCPlot.x.class = colnames(rnn.Hazardous_ROC.x)
ROCPlot.y.class = colnames(rnn.Hazardous_ROC.y)
plot.new()
title(main="RNN Hazardous ROC", xlab="Sensitivity - True Positive Rate", ylab="Specificity - False Positive Rate")
for (ROCPlot.x.classindex in 2:length(ROCPlot.x.class)){
  ROCPlot.name = ROCPlot.x.class[ROCPlot.x.classindex]
  ROCPlot.y.classindex = which(ROCPlot.y.class == ROCPlot.x.class[ROCPlot.x.classindex])
  
  ROCPlot.x = unlist(rnn.Hazardous_ROC.x[,ROCPlot.x.classindex], use.names=FALSE)
  ROCPlot.y = unlist(rnn.Hazardous_ROC.y[,ROCPlot.y.classindex], use.names=FALSE)
  lines(ROCPlot.x, ROCPlot.y, col=plot.models.color[ROCPlot.x.classindex-1],lwd=1)
  
  
}
legend("right", title="models",legend=ROCPlot.y.class,lwd=5, col=plot.models.color,
       horiz=FALSE)
dev.off()

img_name_plot <- paste("IMG_asteroids_model_RNN_", "Hazardous_KFOLD_ROC_wcutoff", ".png", sep = "")
png(img_name_plot,res = 800, height = 10, width = 15, unit='in')
ROCPlot.x.class = colnames(rnn.Hazardous_ROC.x)
ROCPlot.y.class = colnames(rnn.Hazardous_ROC.y)
plot.new()
title(main="RNN Hazardous ROC", xlab="Sensitivity - True Positive Rate", ylab="Specificity - False Positive Rate")
for (ROCPlot.x.classindex in 2:length(ROCPlot.x.class)){
  ROCPlot.name = ROCPlot.x.class[ROCPlot.x.classindex]
  ROCPlot.y.classindex = which(ROCPlot.y.class == ROCPlot.x.class[ROCPlot.x.classindex])
  
  ROCPlot.x = unlist(rnn.Hazardous_ROC.x[,ROCPlot.x.classindex], use.names=FALSE)
  ROCPlot.y = unlist(rnn.Hazardous_ROC.y[,ROCPlot.y.classindex], use.names=FALSE)
  lines(ROCPlot.x, ROCPlot.y, col=plot.models.color[ROCPlot.x.classindex-1],lwd=1)
  points(rnn.Hazardous$CutOffOpt[ROCPlot.name][[1]][1], 1, type="o", col=plot.models.color[ROCPlot.x.classindex-1], pch="o", lty=1, ylim=c(0,110) )
  
}
legend("right", title="models",legend=ROCPlot.y.class,lwd=5, col=plot.models.color,
       horiz=FALSE)
dev.off()

img_name_plot <- paste("IMG_asteroids_model_RNN_", "Hazardous_KFOLD_performance_plot_A", ".png", sep = "")
png(img_name_plot,res = 800, height = 10, width = 15, unit='in')
par(mfrow=c(2,2)) 
boxplot(rnn.Hazardous$Accuracy[2:end_table],vertical = TRUE, pch=19, las = 2, col = plot.models.color,main="Accuracy")  
boxplot(rnn.Hazardous$MacroSensitivity[2:end_table],vertical = TRUE, pch=19, las = 2, col = plot.models.color,main="MacroSensitivity")  
dev.off()
img_name_plot <- paste("IMG_asteroids_model_RNN_", "Hazardous_KFOLD_performance_plot_B", ".png", sep = "")
png(img_name_plot,res = 800, height = 10, width = 15, unit='in')
par(mfrow=c(2,2))  
boxplot(rnn.Hazardous$MacroSpecificity[2:end_table], vertical = TRUE, pch=19, las = 2, col = plot.models.color,main="MacroSpecificity")  
boxplot(rnn.Hazardous$MacroPrecision[2:end_table], vertical = TRUE, pch=19, las = 2, col = plot.models.color,main="MacroPrecision")  
dev.off()
img_name_plot <- paste("IMG_asteroids_model_RNN_", "Hazardous_KFOLD_performance_plot_C", ".png", sep = "")
png(img_name_plot,res = 800, height = 10, width = 15, unit='in')
par(mfrow=c(2,2))  
boxplot(rnn.Hazardous$MacroRecall[2:end_table],vertical = TRUE, pch=19, las = 2, col = plot.models.color,main="MacroRecall")  
boxplot(rnn.Hazardous$MacroF1[2:end_table],vertical = TRUE, pch=19, las = 2, col = plot.models.color,main="MacroF1")  
dev.off()

img_name_plot <- paste("PDF_asteroids_model_RNN_", "Hazardous_KFOLD_performance_tdist", ".pdf", sep = "")
pdf(img_name_plot, height = 20, width = 46)
grid.table(t(rnn.Hazardous.All[2:length(rnn.Hazardous.All)]))
dev.off()


#rm(rnn.Hazardous, rnn.Hazardous.All)

#Hazardous class
#a small cost creates a large margin (a soft margin) and allows more misclassifications;
#a large cost creates a narrow margin (a hard margin) and allows few misclassifications
#this is a soft margin rnn


#Classification type asteroids


#RNN CLASSIFICATION

folds.number = 5

hiddenLayers_list = list(c(7),c(14,7))#c(5),c(10,5))
actFun_list = list("tanh")#,"logistic")#,actFun$softplus,actFun$relu,actFun$sigmoid)
actFun_listname = list("tanh")#,"logistic")#,"softplus","relu","sigmoid")
learningRate_list = list(1e-4)
lossFun_list = list(NULL)
lossFun_listname = list("Loss A")
stepmax_val = 1e5

asteroids_split$train$Amor = asteroids_split$train$Classification == "Amor Asteroid"
asteroids_split$train$Apohele = asteroids_split$train$Classification == "Apohele Asteroid"
asteroids_split$train$Apollo = asteroids_split$train$Classification == "Apollo Asteroid"
asteroids_split$train$Aten = asteroids_split$train$Classification == "Aten Asteroid"

folds <- split(asteroids_split$train, cut(sample(1:nrow(asteroids_split$train)), folds.number))
rm(folds.number)
rnn.Classification <- list()
  
mx = matrix(NA, nrow = length(folds))

rnn.Classification$Accuracy = data.frame(mx)
rnn.Classification$MacroSensitivity <- data.frame(mx)
rnn.Classification$MacroSpecificity <- data.frame(mx)
rnn.Classification$MacroPrecision <- data.frame(mx)
rnn.Classification$MacroRecall <- data.frame(mx)
rnn.Classification$MacroF1 <- data.frame(mx)

#Amor
rnn.Classification$Amor.AUC <- data.frame(mx)
rnn.Classification$Amor.CutOffOpt <- data.frame(mx)
rnn.Classification_ROC.Amor.x <- matrix()
rnn.Classification_ROC.Amor.y <- matrix()
#Apohele
rnn.Classification$Apohele.AUC <- data.frame(mx)
rnn.Classification$Apohele.CutOffOpt <- data.frame(mx)
rnn.Classification_ROC.Apohele.x <- matrix()
rnn.Classification_ROC.Apohele.y <- matrix()
#Apollo
rnn.Classification$Apollo.AUC <- data.frame(mx)
rnn.Classification$Apollo.CutOffOpt <- data.frame(mx)
rnn.Classification_ROC.Apollo.x <- matrix()
rnn.Classification_ROC.Apollo.y <- matrix()
#Aten
rnn.Classification$Aten.AUC <- data.frame(mx)
rnn.Classification$Aten.CutOffOpt <- data.frame(mx)
rnn.Classification_ROC.Aten.x <- matrix()
rnn.Classification_ROC.Aten.y <- matrix()

rnn.Classification_ROC.name <- c(NA)

mx = matrix(NA, nrow = 6)
rnn.Classification.All <- data.frame(mx)
rnn.Classification.All["Performance"] = c("Accuracy","MacroSensitivity","MacroSpecificity","MacroPrecision","MacroRecall","MacroF1")
rm(mx)

mx = matrix(NA, nrow = 8)
rnn.Classification.ROC.All <- data.frame(mx)
rnn.Classification.ROC.All["Performance"] = c("Amor AUC","Amor CutOffOpt","Apohele AUC","Apohele CutOffOpt","Apollo AUC","Apollo CutOffOpt","Aten AUC","Aten CutOffOpt")
rm(mx)

for (hl in 1:length(hiddenLayers_list)) {
  for (ac in 1:length(actFun_list)) {
    for (lrr in 1:length(learningRate_list)) {
      for (lf in 1:length(lossFun_list)) {
        hiddenL_val = hiddenLayers_list[[hl]]
        actFun_val = actFun_list[[ac]]
        actFun_name = actFun_listname[[ac]]
        lr_val = learningRate_list[[lrr]]
        loss_f = lossFun_list[[lf]]
        loss_name = lossFun_listname[[lf]]
        rnn.Classification.stats <- list()
        
        rnn.model_printname = paste("Classification"," ACT:",actFun_name," LOSS:",loss_name," HIDDENL: (", paste(hiddenL_val, collapse = "+"),") LR:",as.character(lr_val),sep=" ")
        rnn.name = paste("Class",actFun_name,loss_name,paste(hiddenL_val, collapse = "+"),as.character(lr_val),sep="_")
        
        rnn.Classification.stats.roc.pred.prob <- list()
        rnn.Classification.stats.roc.thruth <- list()
        
        folds_confusion <- NULL
        
        for (i in 1:length(folds)) {
          fold.valid <- ldply(folds[i], data.frame)
          fold.valid <- fold.valid[, !names(fold.valid) %in% c(".id")]
          fold.train <- ldply(folds[-i], data.frame)
          fold.train <- fold.train[, !names(fold.train) %in% c(".id")]
          
          print(paste(as.character(i), rnn.model_printname, sep = " "))
          #manca LOSS loss_f
          rnn.Classification.model = neuralnet(Amor+Apohele+Apollo+Aten ~ Orbit.Axis..AU._scaled_scaled + Orbit.Eccentricity_scaled + Orbit.Inclination..deg._scaled + Perihelion.Argument..deg._scaled + Node.Longitude..deg._scaled + Mean.Anomoly..deg._scaled + Perihelion.Distance..AU._scaled + Aphelion.Distance..AU._scaled + Orbital.Period..yr._scaled + Minimum.Orbit.Intersection.Distance..AU._scaled + Asteroid.Magnitude_scaled,
                                               fold.train[1:20,],
                                               hidden = hiddenL_val,
                                               act.fct = actFun_val,
                                               learningrate = lr_val,
                                               stepmax=stepmax_val,
                                               linear.output = TRUE)
            
          rnn.Classification.pred = predict(rnn.Classification.model, fold.valid)
          rnn.Classification.pred.max = as.factor(c("Amor Asteroid", "Apohele Asteroid", "Apollo Asteroid", "Aten Asteroid")[apply(rnn.Classification.pred, 1, which.max)])
          
          folds_confusion <- Confusion_Sum(folds_confusion, reference=fold.valid$Classification, data=rnn.Classification.pred.max)
          
          rnn.Classification.confusion_matrix_multiclass = confusionMatrix(
            data=rnn.Classification.pred.max, reference=fold.valid$Classification, mode = "prec_recall") 
          
          confusion.multi = rnn.Classification.confusion_matrix_multiclass$byClass
          
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
          
          MacroSensitivity = mean(c(sens_Amor,sens_Apohele,sens_Apollo,sens_Aten), na.rm = TRUE)
          MacroSpecificity = mean(c(spec_Amor,spec_Apohele,spec_Apollo,spec_Aten), na.rm = TRUE)
          MacroPrecision = mean(c(prec_Amor,prec_Apohele,prec_Apollo,prec_Aten), na.rm = TRUE)
          MacroRecall = mean(c(recal_Amor,recal_Apohele,recal_Apollo,recal_Aten), na.rm = TRUE)
          MacroF1 =  mean(c(f1_Amor,f1_Apohele,f1_Apollo,f1_Aten), na.rm = TRUE)
          
          rnn.Classification.stats$Accuracy    = append(rnn.Classification.stats$Accuracy, rnn.Classification.confusion_matrix_multiclass$overall["Accuracy"])
          rnn.Classification.stats$MacroSensitivity = append(rnn.Classification.stats$MacroSensitivity, MacroSensitivity)
          rnn.Classification.stats$MacroSpecificity = append(rnn.Classification.stats$MacroSpecificity, MacroSpecificity)
          rnn.Classification.stats$MacroPrecision = append(rnn.Classification.stats$MacroPrecision, MacroPrecision)
          rnn.Classification.stats$MacroRecall = append(rnn.Classification.stats$MacroRecall, MacroRecall)
          rnn.Classification.stats$MacroF1 = append(rnn.Classification.stats$MacroF1, MacroF1)
          #ROC
          ROCFun.pred.prob.fold = predict(rnn.Classification.model,fold.valid,  probability=TRUE)
          rnn.Classification.stats.roc.pred.prob = rbind(rnn.Classification.stats.roc.pred.prob,ROCFun.pred.prob.fold )
          rnn.Classification.stats.roc.thruth = append(rnn.Classification.stats.roc.thruth, as.factor(fold.valid$Classification))
        }
        
        #end kfold
        img_name_plot <- paste("IMG_asteroids_model_SWM_", rnn.name ,"_confusion" ,".png", sep = "")
        png(img_name_plot,res = 800, height = 10, width = 15, unit='in')
          grid.table(table(folds_confusion))
          dev.off()
        
        
        colnames(rnn.Classification.stats.roc.pred.prob) <- c("Amor Asteroid","Apohele Asteroid","Apollo Asteroid","Aten Asteroid")
        rnn.Classification.roc.Amor = ROCFunction.MULTI(rnn.Classification.stats.roc.pred.prob,rnn.Classification.stats.roc.thruth,"Amor Asteroid")
        rnn.Classification.roc.Apohele = ROCFunction.MULTI(rnn.Classification.stats.roc.pred.prob,rnn.Classification.stats.roc.thruth,"Apohele Asteroid")
        rnn.Classification.roc.Apollo = ROCFunction.MULTI(rnn.Classification.stats.roc.pred.prob,rnn.Classification.stats.roc.thruth,"Apollo Asteroid")
        rnn.Classification.roc.Aten = ROCFunction.MULTI(rnn.Classification.stats.roc.pred.prob,rnn.Classification.stats.roc.thruth,"Aten Asteroid")
        
        rnn.Classification$Accuracy[rnn.name] <- rnn.Classification.stats$Accuracy
        rnn.Classification$MacroSensitivity[rnn.name] <- rnn.Classification.stats$MacroSensitivity
        rnn.Classification$MacroSpecificity[rnn.name] <- rnn.Classification.stats$MacroSpecificity
        rnn.Classification$MacroPrecision[rnn.name] <- rnn.Classification.stats$MacroPrecision
        rnn.Classification$MacroRecall[rnn.name] <- rnn.Classification.stats$MacroRecall
        rnn.Classification$MacroF1[rnn.name] <- rnn.Classification.stats$MacroF1
        
        #roc
        rnn.Classification$Amor.AUC[rnn.name] <- rnn.Classification.roc.Amor$auc
        rnn.Classification$Amor.CutOffOpt[rnn.name] <- rnn.Classification.roc.Amor$optcut
        rnn.Classification$Apohele.AUC[rnn.name] <- rnn.Classification.roc.Apohele$auc
        rnn.Classification$Apohele.CutOffOpt[rnn.name] <- rnn.Classification.roc.Apohele$optcut
        rnn.Classification$Apollo.AUC[rnn.name] <- rnn.Classification.roc.Apollo$auc
        rnn.Classification$Apollo.CutOffOpt[rnn.name] <- rnn.Classification.roc.Apollo$optcut
        rnn.Classification$Aten.AUC[rnn.name] <- rnn.Classification.roc.Aten$auc
        rnn.Classification$Aten.CutOffOpt[rnn.name] <- rnn.Classification.roc.Aten$optcut
        # ROC DATA FRAMES
        rnn.Classification_ROC.name = c(rnn.Classification_ROC.name, rnn.name)
        rnn.Classification_ROC.Amor.x <- cbindX(rnn.Classification_ROC.Amor.x, data.frame(rnn.Classification.roc.Amor$x.value))
        colnames(rnn.Classification_ROC.Amor.x) <- rnn.Classification_ROC.name
        rnn.Classification_ROC.Amor.y <- cbindX(rnn.Classification_ROC.Amor.y, data.frame(rnn.Classification.roc.Amor$y.value))
        colnames(rnn.Classification_ROC.Amor.y) <- rnn.Classification_ROC.name
        
        rnn.Classification_ROC.Apohele.x <- cbindX(rnn.Classification_ROC.Apohele.x, data.frame(rnn.Classification.roc.Apohele$x.value))
        colnames(rnn.Classification_ROC.Apohele.x) <- rnn.Classification_ROC.name
        rnn.Classification_ROC.Apohele.y <- cbindX(rnn.Classification_ROC.Apohele.y, data.frame(rnn.Classification.roc.Apohele$y.value))
        colnames(rnn.Classification_ROC.Apohele.y) <- rnn.Classification_ROC.name
        
        rnn.Classification_ROC.Apollo.x <- cbindX(rnn.Classification_ROC.Apollo.x, data.frame(rnn.Classification.roc.Apollo$x.value))
        colnames(rnn.Classification_ROC.Apollo.x) <- rnn.Classification_ROC.name
        rnn.Classification_ROC.Apollo.y <- cbindX(rnn.Classification_ROC.Apollo.y, data.frame(rnn.Classification.roc.Apollo$y.value))
        colnames(rnn.Classification_ROC.Apollo.y) <- rnn.Classification_ROC.name
        
        rnn.Classification_ROC.Aten.x <- cbindX(rnn.Classification_ROC.Aten.x, data.frame(rnn.Classification.roc.Aten$x.value))
        colnames(rnn.Classification_ROC.Aten.x) <- rnn.Classification_ROC.name
        rnn.Classification_ROC.Aten.y <- cbindX(rnn.Classification_ROC.Aten.y, data.frame(rnn.Classification.roc.Aten$y.value))
        colnames(rnn.Classification_ROC.Aten.y) <- rnn.Classification_ROC.name
        
        tdist <- list()
        tdist_name <-  paste("Class ",actFun_name,loss_name,"(", paste(hiddenL_val, collapse = "+"),")",as.character(lr_val),sep=" ")
        
        tdist_val = confidence_interval(as.vector(rnn.Classification.stats$Accuracy),0.95)
        tdist$acc <- paste(as.character(round(tdist_val[2],4))," � ",as.character(round(tdist_val[1],4)))
        
        tdist_val = confidence_interval(as.vector(rnn.Classification.stats$MacroSensitivity),0.95)
        tdist$sens <- paste(as.character(round(tdist_val[2],4))," � ",as.character(round(tdist_val[1],4)))
        
        tdist_val = confidence_interval(as.vector(rnn.Classification.stats$MacroSpecificity),0.95)
        tdist$spec <- paste(as.character(round(tdist_val[2],4))," � ",as.character(round(tdist_val[1],4)))
        
        tdist_val = confidence_interval(as.vector(rnn.Classification.stats$MacroPrecision),0.95)
        tdist$prec <- paste(as.character(round(tdist_val[2],4))," � ",as.character(round(tdist_val[1],4)))
        
        tdist_val = confidence_interval(as.vector(rnn.Classification.stats$MacroRecall),0.95)
        tdist$rec <- paste(as.character(round(tdist_val[2],4))," � ",as.character(round(tdist_val[1],4)))
        
        tdist_val = confidence_interval(as.vector(rnn.Classification.stats$MacroF1),0.95)
        tdist$f1 <- paste(as.character(round(tdist_val[2],4))," � ",as.character(round(tdist_val[1],4)))
        
        rnn.Classification.All[tdist_name] <- c(tdist$acc,tdist$sens,tdist$spec,tdist$prec,tdist$rec,tdist$f1)
        
        
        rdist_name <-  paste("Class ",actFun_name,loss_name,"(", paste(hiddenL_val, collapse = "+"),")",as.character(lr_val),sep=" ")
        
        rdist <- list()
        rdist_val = rnn.Classification.roc.Amor$auc
        rdist$Amor.auc <- paste(as.character(round(rdist_val,8)))
        rdist_val = rnn.Classification.roc.Amor$optcut
        rdist$Amor.optcut <- paste(as.character(round(rdist_val,5)))
        
        rdist_val = rnn.Classification.roc.Apohele$auc
        rdist$Apohele.auc <- paste(as.character(round(rdist_val,8)))
        rdist_val = rnn.Classification.roc.Apohele$optcut
        rdist$Apohele.optcut <- paste(as.character(round(rdist_val,5)))
        
        rdist_val = rnn.Classification.roc.Apollo$auc
        rdist$Apollo.auc <- paste(as.character(round(rdist_val,8)))
        rdist_val = rnn.Classification.roc.Apollo$optcut
        rdist$Apollo.optcut <- paste(as.character(round(rdist_val,5)))
        
        rdist_val = rnn.Classification.roc.Aten$auc
        rdist$Aten.auc <- paste(as.character(round(rdist_val,8)))
        rdist_val = rnn.Classification.roc.Aten$optcut
        rdist$Aten.optcut <- paste(as.character(round(rdist_val,5)))
        rnn.Classification.ROC.All[rdist_name] <- c(rdist$Amor.auc,rdist$Amor.optcut,rdist$Apohele.auc,rdist$Apohele.optcut,rdist$Apollo.auc,rdist$Apollo.optcut,rdist$Aten.auc,rdist$Aten.optcut)
        
        
      }
    }
  }
  img_name_plot <- paste("IMG_asteroids_model_RNN_", "Classification_HIDDEN_LAYER_",paste(hiddenL_val, collapse = "+") ,".png", sep = "")
  png(img_name_plot, res = 800, height = 10, width = 15, unit='in')
    plot.nnet(rnn.Classification.model)
    dev.off()
    
  img_name_plot <- paste("IMG_asteroids_model_RNN_", "Classification_generalized_weights", ".png", sep = "")
  png(img_name_plot, res = 800, height = 10, width = 15, unit='in')
  par(mfrow=c(2,2))
    gwplot(rnn.Classification.model,selected.covariate="Perihelion.Distance..AU._scaled")
    gwplot(rnn.Classification.model,selected.covariate="Orbit.Axis..AU._scaled_scaled")
    gwplot(rnn.Classification.model,selected.covariate="Aphelion.Distance..AU._scaled")
    gwplot(rnn.Classification.model,selected.covariate="Perihelion.Argument..deg._scaled")
    dev.off()
     
}
end_table <- length(rnn.Classification$Accuracy)
plot.models.color = rainbow(end_table-1)

img_name_plot <- paste("IMG_asteroids_model_RNN_", "Classification_KFOLD_ROC", ".png", sep = "")
png(img_name_plot,res = 800, height = 10, width = 15, unit='in')
par(mfrow=c(2,2))  
# Amor
plot.new()
ROCPlot.x.class = colnames(rnn.Classification_ROC.Amor.x)
ROCPlot.y.class = colnames(rnn.Classification_ROC.Amor.y)

title(main="ROC Class: Amor", xlab="Sensitivity - True Positive Rate", ylab="Specificity - False Positive Rate")
for (ROCPlot.x.classindex in 2:length(ROCPlot.x.class)){
  ROCPlot.name = ROCPlot.x.class[ROCPlot.x.classindex]
  ROCPlot.y.classindex = which(ROCPlot.y.class == ROCPlot.x.class[ROCPlot.x.classindex])
  
  ROCPlot.x = unlist(rnn.Classification_ROC.Amor.x[,ROCPlot.x.classindex], use.names=FALSE)
  ROCPlot.y = unlist(rnn.Classification_ROC.Amor.y[,ROCPlot.y.classindex], use.names=FALSE)
  lines(ROCPlot.x, ROCPlot.y, col=plot.models.color[ROCPlot.x.classindex-1],lwd=1)    
}
legend("right", title="models",legend=ROCPlot.y.class,lwd=5, col=plot.models.color,horiz=FALSE)
#Apohele
plot.new()
ROCPlot.x.class = colnames(rnn.Classification_ROC.Apohele.x)
ROCPlot.y.class = colnames(rnn.Classification_ROC.Apohele.y)

title(main="ROC Class: Apohele", xlab="Sensitivity - True Positive Rate", ylab="Specificity - False Positive Rate")
for (ROCPlot.x.classindex in 2:length(ROCPlot.x.class)){
  ROCPlot.name = ROCPlot.x.class[ROCPlot.x.classindex]
  ROCPlot.y.classindex = which(ROCPlot.y.class == ROCPlot.x.class[ROCPlot.x.classindex])
  
  ROCPlot.x = unlist(rnn.Classification_ROC.Apohele.x[,ROCPlot.x.classindex], use.names=FALSE)
  ROCPlot.y = unlist(rnn.Classification_ROC.Apohele.y[,ROCPlot.y.classindex], use.names=FALSE)
  lines(ROCPlot.x, ROCPlot.y, col=plot.models.color[ROCPlot.x.classindex-1],lwd=1)    
}
legend("right", title="models",legend=ROCPlot.y.class,lwd=5, col=plot.models.color,horiz=FALSE)
#Apollo
plot.new()
ROCPlot.x.class = colnames(rnn.Classification_ROC.Apollo.x)
ROCPlot.y.class = colnames(rnn.Classification_ROC.Apollo.y)

title(main="ROC Class: Apollo", xlab="Sensitivity - True Positive Rate", ylab="Specificity - False Positive Rate")
for (ROCPlot.x.classindex in 2:length(ROCPlot.x.class)){
  ROCPlot.name = ROCPlot.x.class[ROCPlot.x.classindex]
  ROCPlot.y.classindex = which(ROCPlot.y.class == ROCPlot.x.class[ROCPlot.x.classindex])
  
  ROCPlot.x = unlist(rnn.Classification_ROC.Apollo.x[,ROCPlot.x.classindex], use.names=FALSE)
  ROCPlot.y = unlist(rnn.Classification_ROC.Apollo.y[,ROCPlot.y.classindex], use.names=FALSE)
  lines(ROCPlot.x, ROCPlot.y, col=plot.models.color[ROCPlot.x.classindex-1],lwd=1)    
}
legend("right", title="models",legend=ROCPlot.y.class,lwd=5, col=plot.models.color,horiz=FALSE)
#Aten
plot.new()
ROCPlot.x.class = colnames(rnn.Classification_ROC.Aten.x)
ROCPlot.y.class = colnames(rnn.Classification_ROC.Aten.y)

title(main="ROC Class: Aten", xlab="Sensitivity - True Positive Rate", ylab="Specificity - False Positive Rate")
for (ROCPlot.x.classindex in 2:length(ROCPlot.x.class)){
  ROCPlot.name = ROCPlot.x.class[ROCPlot.x.classindex]
  ROCPlot.y.classindex = which(ROCPlot.y.class == ROCPlot.x.class[ROCPlot.x.classindex])
  
  ROCPlot.x = unlist(rnn.Classification_ROC.Aten.x[,ROCPlot.x.classindex], use.names=FALSE)
  ROCPlot.y = unlist(rnn.Classification_ROC.Aten.y[,ROCPlot.y.classindex], use.names=FALSE)
  lines(ROCPlot.x, ROCPlot.y, col=plot.models.color[ROCPlot.x.classindex-1],lwd=1)    
}
legend("right", title="models",legend=ROCPlot.y.class,lwd=5, col=plot.models.color,horiz=FALSE)
dev.off()



img_name_plot <- paste("IMG_asteroids_model_RNN_", "Classification_KFOLD_performance_plot_A", ".png", sep = "")
png(img_name_plot,res = 800, height = 10, width = 15, unit='in')
par(mfrow=c(2,2)) 
boxplot(rnn.Classification$Accuracy[2:end_table],vertical = TRUE, pch=19, las = 2, col = plot.models.color,main="Accuracy")  
boxplot(rnn.Classification$MacroSensitivity[2:end_table],vertical = TRUE, pch=19, las = 2, col = plot.models.color,main="MacroSensitivity")  
dev.off()
img_name_plot <- paste("IMG_asteroids_model_RNN_", "Classification_KFOLD_performance_plot_B", ".png", sep = "")
png(img_name_plot,res = 800, height = 10, width = 15, unit='in')
par(mfrow=c(2,2))  
boxplot(rnn.Classification$MacroSpecificity[2:end_table], vertical = TRUE, pch=19, las = 2, col = plot.models.color,main="MacroSpecificity")  
boxplot(rnn.Classification$MacroPrecision[2:end_table], vertical = TRUE, pch=19, las = 2, col = plot.models.color,main="MacroPrecision")  
dev.off()
img_name_plot <- paste("IMG_asteroids_model_RNN_", "Classification_KFOLD_performance_plot_C", ".png", sep = "")
png(img_name_plot,res = 800, height = 10, width = 15, unit='in')
par(mfrow=c(2,2))  
boxplot(rnn.Classification$MacroRecall[2:end_table],vertical = TRUE, pch=19, las = 2, col = plot.models.color,main="MacroRecall")  
boxplot(rnn.Classification$MacroF1[2:end_table],vertical = TRUE, pch=19, las = 2, col = plot.models.color,main="MacroF1")  
dev.off()

img_name_plot <- paste("PDF_asteroids_model_RNN_", "Classification_KFOLD_performance_tdist", ".pdf", sep = "")
pdf(img_name_plot, height = 20, width = 46)
grid.table(t(rnn.Classification.All[2:length(rnn.Classification.All)]))
dev.off()

img_name_plot <- paste("PDF_asteroids_model_RNN_", "Classification_KFOLD_performance_ROC", ".pdf", sep = "")
pdf(img_name_plot, height = 20, width = 46)
grid.table(t(rnn.Classification.ROC.All[2:length(rnn.Classification.ROC.All)]))
dev.off()

