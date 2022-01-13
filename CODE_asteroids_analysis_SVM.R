setwd("~/Github/AsteroidsClassification")

library(plyr)
library(gridExtra)
library(e1071)
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

folds.number = 5
kernel_list <- c('linear','polynomial','radial')
C_list <- c(0.5,1,10,100)

folds <- split(asteroids_split$train, cut(sample(1:nrow(asteroids_split$train)), folds.number))
rm(folds.number)
svm.Hazardous <- list()

mx = matrix(NA, nrow = length(folds))

svm.Hazardous$Accuracy = data.frame(mx)
svm.Hazardous$MacroSensitivity <- data.frame(mx)
svm.Hazardous$MacroSpecificity <- data.frame(mx)
svm.Hazardous$MacroPrecision <- data.frame(mx)
svm.Hazardous$MacroRecall <- data.frame(mx)
svm.Hazardous$MacroF1 <- data.frame(mx)

mx = matrix(NA, nrow = 6)
svm.Hazardous.All <- data.frame(mx)
svm.Hazardous.All["Performance"] = c("Accuracy","MacroSensitivity","Specificity","MacroPrecision","MacroRecall","MacroF1")
rm(mx)


for (j in 1:length(kernel_list)) {
  hyper.kernel <- kernel_list[j]
  
  for (l in 1:length(C_list)) {
    hyper.cost <- C_list[l]
    svm.Hazardous.stats <- list()
    
    for (i in 1:length(folds)) {
      fold.valid <- ldply(folds[i], data.frame)
      fold.valid <- fold.valid[, !names(fold.valid) %in% c(".id")]
      fold.train <- ldply(folds[-i], data.frame)
      fold.train <- fold.train[, !names(fold.train) %in% c(".id")]
      
      svm.Hazardous$performance[hyper.kernel][hyper.cost] <- list()
      print(paste(as.character(i), hyper.kernel , as.character(hyper.cost), sep = " "))
      svm.Hazardous.model = svm(Hazardous.int ~ Orbit.Axis..AU. + Orbit.Eccentricity + Orbit.Inclination..deg. + Perihelion.Argument..deg. + Node.Longitude..deg. + Mean.Anomoly..deg. + Perihelion.Distance..AU. + Aphelion.Distance..AU. + Orbital.Period..yr. + Minimum.Orbit.Intersection.Distance..AU. + Asteroid.Magnitude,
        data=fold.train, kernel=hyper.kernel, cost=hyper.cost)
      svm.Hazardous.pred = predict(svm.Hazardous.model, fold.valid)
      
      svm.Hazardous.confusion_matrix_true = confusionMatrix(
        svm.Hazardous.pred, fold.valid$Hazardous.int, positive="TRUE", mode = "prec_recall") 
      
      svm.Hazardous.confusion_matrix_false = confusionMatrix(
        svm.Hazardous.pred, fold.valid$Hazardous.int, positive="FALSE", mode = "prec_recall") 
      
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
      rm(svm.Hazardous.model, svm.Hazardous.pred)
      rm(svm.Hazardous.confusion_matrix_true,svm.Hazardous.confusion_matrix_false,prec_true,recal_true,f1_true,prec_false,recal_false,f1_false,MacroPrecision,MacroRecall,MacroF1)
    }
    
    svm.name <- paste("Haz",hyper.kernel,as.character(hyper.cost),sep="_")
    svm.Hazardous$Accuracy[svm.name] <- svm.Hazardous.stats$Accuracy
    svm.Hazardous$MacroSensitivity[svm.name] <- svm.Hazardous.stats$MacroSensitivity
    svm.Hazardous$MacroSpecificity[svm.name] <- svm.Hazardous.stats$MacroSpecificity
    svm.Hazardous$MacroPrecision[svm.name] <- svm.Hazardous.stats$MacroPrecision
    svm.Hazardous$MacroRecall[svm.name] <- svm.Hazardous.stats$MacroRecall
    svm.Hazardous$MacroF1[svm.name] <- svm.Hazardous.stats$MacroF1

    
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
    
    
    svm.Hazardous.All[tdist_name] <- c(tdist$acc,tdist$sens,tdist$spec,tdist$prec,tdist$rec,tdist$f1)
    rm(tdist_name,tdist,svm.name,fold.train, fold.valid)
    
  }
}
end_table <- length(svm.Hazardous$Accuracy)
img_name_plot <- paste("IMG_asteroids_model_SVM_", "Hazardous_KFOLD_performance_plot_A", ".png", sep = "")
png(img_name_plot,res = 800, height = 10, width = 15, unit='in')
  par(mfrow=c(2,2)) 
  boxplot(svm.Hazardous$Accuracy[2:end_table],vertical = TRUE, pch=19, las = 2, col = 1:length(svm.Hazardous$Accuracy),main="Accuracy")  
  boxplot(svm.Hazardous$MacroSensitivity[2:end_table],vertical = TRUE, pch=19, las = 2, col = 1:length(svm.Hazardous$MacroSensitivity),main="MacroSensitivity")  
  dev.off()
img_name_plot <- paste("IMG_asteroids_model_SVM_", "Hazardous_KFOLD_performance_plot_B", ".png", sep = "")
png(img_name_plot,res = 800, height = 10, width = 15, unit='in')
  par(mfrow=c(2,2))  
  boxplot(svm.Hazardous$MacroSpecificity[2:end_table], vertical = TRUE, pch=19, las = 2, col = 1:length(svm.Hazardous$Specificity),main="MacroSpecificity")  
  boxplot(svm.Hazardous$MacroPrecision[2:end_table], vertical = TRUE, pch=19, las = 2, col = 1:length(svm.Hazardous$MacroPrecision),main="MacroPrecision")  
  dev.off()
img_name_plot <- paste("IMG_asteroids_model_SVM_", "Hazardous_KFOLD_performance_plot_C", ".png", sep = "")
png(img_name_plot,res = 800, height = 10, width = 15, unit='in')
  par(mfrow=c(2,2))  
  boxplot(svm.Hazardous$MacroRecall[2:end_table],vertical = TRUE, pch=19, las = 2, col = 1:length(svm.Hazardous$MacroRecall),main="MacroRecall")  
  boxplot(svm.Hazardous$MacroF1[2:end_table],vertical = TRUE, pch=19, las = 2, col = 1:length(svm.Hazardous$MacroF1),main="MacroF1")  
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
C_list <- c(0.5,1,10)

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

mx = matrix(NA, nrow = 6)
svm.Classification.All <- data.frame(mx)
svm.Classification.All["Performance"] = c("Accuracy","MacroSensitivity","MacroSpecificity","MacroPrecision","MacroRecall","MacroF1")
rm(mx)


for (j in 1:length(kernel_list)) {
  hyper.kernel <- kernel_list[j]
  
  for (l in 1:length(C_list)) {
    hyper.cost <- C_list[l]
    svm.Classification.stats <- list()
    
    for (i in 1:length(folds)) {
      fold.valid <- ldply(folds[i], data.frame)
      fold.valid <- fold.valid[, !names(fold.valid) %in% c(".id")]
      fold.train <- ldply(folds[-i], data.frame)
      fold.train <- fold.train[, !names(fold.train) %in% c(".id")]
      
      svm.Classification$performance[hyper.kernel][hyper.cost] <- list()
      print(paste(as.character(i), hyper.kernel , as.character(hyper.cost), sep = " "))
      svm.Classification.model = svm(Classification ~ Orbit.Axis..AU. + Orbit.Eccentricity + Orbit.Inclination..deg. + Perihelion.Argument..deg. + Node.Longitude..deg. + Mean.Anomoly..deg. + Perihelion.Distance..AU. + Aphelion.Distance..AU. + Orbital.Period..yr. + Minimum.Orbit.Intersection.Distance..AU. + Asteroid.Magnitude,
                                     data=fold.train, kernel=hyper.kernel, cost=hyper.cost)
      svm.Classification.pred = predict(svm.Classification.model, fold.valid)
      
      svm.Classification.confusion_matrix_multiclass = confusionMatrix(
        svm.Classification.pred, fold.valid$Classification, mode = "prec_recall") 

      
      
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
      
      MacroSensitivity = (0.25 * sens_Amor) + (0.25 * sens_Apohele) + (0.25 * sens_Apollo) + (0.25 * sens_Aten)
      MacroSpecificity = (0.25 * spec_Amor) + (0.25 * spec_Apohele) + (0.25 * spec_Apollo) + (0.25 * spec_Aten)
      MacroPrecision = (0.25 * prec_Amor) + (0.25 * prec_Apohele) + (0.25 * prec_Apollo) + (0.25 * prec_Aten)
      MacroRecall = (0.25 * recal_Amor) + (0.25 * recal_Apohele) + (0.25 * recal_Apollo) + (0.25 * recal_Aten)
      MacroF1 = (0.25 * f1_Amor) + (0.25 * f1_Apohele) + (0.25 * f1_Apollo) + (0.25 * prec_Aten)
      
      svm.Classification.stats$Accuracy    = append(svm.Classification.stats$Accuracy, svm.Classification.confusion_matrix_multiclass$overall["Accuracy"])
      svm.Classification.stats$MacroSensitivity = append(svm.Classification.stats$MacroSensitivity, MacroSensitivity)
      svm.Classification.stats$MacroSpecificity = append(svm.Classification.stats$MacroSpecificity, MacroSpecificity)
      svm.Classification.stats$MacroPrecision = append(svm.Classification.stats$MacroPrecision, MacroPrecision)
      svm.Classification.stats$MacroRecall = append(svm.Classification.stats$MacroRecall, MacroRecall)
      svm.Classification.stats$MacroF1 = append(svm.Classification.stats$MacroF1, MacroF1)
      rm(svm.Classification.model, svm.Classification.pred)
      rm(svm.Classification.confusion_matrix_true,svm.Classification.confusion_matrix_false,prec_true,recal_true,f1_true,prec_false,recal_false,f1_false,MacroPrecision,MacroRecall,MacroF1)
    }
    
    svm.name <- paste("Haz",hyper.kernel,as.character(hyper.cost),sep="_")
    svm.Classification$Accuracy[svm.name] <- svm.Classification.stats$Accuracy
    svm.Classification$MacroSensitivity[svm.name] <- svm.Classification.stats$MacroSensitivity
    svm.Classification$MacroSpecificity[svm.name] <- svm.Classification.stats$MacroSpecificity
    svm.Classification$MacroPrecision[svm.name] <- svm.Classification.stats$MacroPrecision
    svm.Classification$MacroRecall[svm.name] <- svm.Classification.stats$MacroRecall
    svm.Classification$MacroF1[svm.name] <- svm.Classification.stats$MacroF1
    
    
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
    rm(tdist_name,tdist,svm.name,fold.train, fold.valid)
    
  }
}
end_table <- length(svm.Classification$Accuracy)
img_name_plot <- paste("IMG_asteroids_model_SVM_", "Classification_KFOLD_performance_plot_A", ".png", sep = "")
png(img_name_plot,res = 800, height = 10, width = 15, unit='in')
par(mfrow=c(2,2)) 
  boxplot(svm.Classification$Accuracy[2:end_table],vertical = TRUE, pch=19, las = 2, col = 1:length(svm.Classification$Accuracy),main="Accuracy")  
  boxplot(svm.Classification$MacroSensitivity[2:end_table],vertical = TRUE, pch=19, las = 2, col = 1:length(svm.Classification$MacroSensitivity),main="MacroSensitivity")  
  dev.off()
img_name_plot <- paste("IMG_asteroids_model_SVM_", "Classification_KFOLD_performance_plot_B", ".png", sep = "")
png(img_name_plot,res = 800, height = 10, width = 15, unit='in')
  par(mfrow=c(2,2))  
  boxplot(svm.Classification$MacroSpecificity[2:end_table], vertical = TRUE, pch=19, las = 2, col = 1:length(svm.Classification$MacroSpecificity),main="MacroSpecificity")  
  boxplot(svm.Classification$MacroPrecision[2:end_table], vertical = TRUE, pch=19, las = 2, col = 1:length(svm.Classification$MacroPrecision),main="MacroPrecision")  
  dev.off()
img_name_plot <- paste("IMG_asteroids_model_SVM_", "Classification_KFOLD_performance_plot_C", ".png", sep = "")
png(img_name_plot,res = 800, height = 10, width = 15, unit='in')
  par(mfrow=c(2,2))  
  boxplot(svm.Classification$MacroRecall[2:end_table],vertical = TRUE, pch=19, las = 2, col = 1:length(svm.Classification$MacroRecall),main="MacroRecall")  
  boxplot(svm.Classification$MacroF1[2:end_table],vertical = TRUE, pch=19, las = 2, col = 1:length(svm.Classification$MacroF1),main="MacroF1")  
  dev.off()

img_name_plot <- paste("PDF_asteroids_model_SVM_", "Classification_KFOLD_performance_tdist", ".pdf", sep = "")
pdf(img_name_plot, height = 20, width = 46)
  grid.table(t(svm.Classification.All[2:length(svm.Classification.All)]))
  dev.off()

