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
rnn.Hazardous <- list()

mx = matrix(NA, nrow = length(folds))

rnn.Hazardous$Accuracy = data.frame(mx)
rnn.Hazardous$MacroSensitivity <- data.frame(mx)
rnn.Hazardous$MacroSpecificity <- data.frame(mx)
rnn.Hazardous$MacroPrecision <- data.frame(mx)
rnn.Hazardous$MacroRecall <- data.frame(mx)
rnn.Hazardous$MacroF1 <- data.frame(mx)

mx = matrix(NA, nrow = 6)
rnn.Hazardous.All <- data.frame(mx)
rnn.Hazardous.All["Performance"] = c("Accuracy","MacroSensitivity","Specificity","MacroPrecision","MacroRecall","MacroF1")
rm(mx)


for (j in 1:length(kernel_list)) {
  hyper.kernel <- kernel_list[j]
  
  for (l in 1:length(C_list)) {
    hyper.cost <- C_list[l]
    rnn.Hazardous.stats <- list()
    
    for (i in 1:length(folds)) {
      fold.valid <- ldply(folds[i], data.frame)
      fold.valid <- fold.valid[, !names(fold.valid) %in% c(".id")]
      fold.train <- ldply(folds[-i], data.frame)
      fold.train <- fold.train[, !names(fold.train) %in% c(".id")]
      
      rnn.Hazardous$performance[hyper.kernel][hyper.cost] <- list()
      print(paste(as.character(i), hyper.kernel , as.character(hyper.cost), sep = " "))
      rnn.Hazardous.model = rnn(Hazardous.int ~ Orbit.Axis..AU. + Orbit.Eccentricity + Orbit.Inclination..deg. + Perihelion.Argument..deg. + Node.Longitude..deg. + Mean.Anomoly..deg. + Perihelion.Distance..AU. + Aphelion.Distance..AU. + Orbital.Period..yr. + Minimum.Orbit.Intersection.Distance..AU. + Asteroid.Magnitude,
                                data=fold.train, kernel=hyper.kernel, cost=hyper.cost)
      rnn.Hazardous.pred = predict(rnn.Hazardous.model, fold.valid)
      
      rnn.Hazardous.confusion_matrix_true = confusionMatrix(
        rnn.Hazardous.pred, fold.valid$Hazardous.int, positive="TRUE", mode = "prec_recall") 
      
      rnn.Hazardous.confusion_matrix_false = confusionMatrix(
        rnn.Hazardous.pred, fold.valid$Hazardous.int, positive="FALSE", mode = "prec_recall") 
      
      sens_true = rnn.Hazardous.confusion_matrix_true$byClass["Sensitivity"]
      spec_true = rnn.Hazardous.confusion_matrix_true$byClass["Specificity"]
      prec_true = rnn.Hazardous.confusion_matrix_true$byClass["Precision"]
      recal_true = rnn.Hazardous.confusion_matrix_true$byClass["Recall"]
      f1_true = rnn.Hazardous.confusion_matrix_true$byClass["F1"]
      
      sens_false = rnn.Hazardous.confusion_matrix_false$byClass["Sensitivity"]
      spec_fasle = rnn.Hazardous.confusion_matrix_false$byClass["Specificity"]
      prec_false = rnn.Hazardous.confusion_matrix_false$byClass["Precision"]
      recal_false = rnn.Hazardous.confusion_matrix_false$byClass["Recall"]
      f1_false = rnn.Hazardous.confusion_matrix_false$byClass["F1"]
      
      MacroSensitivity = (0.5 * sens_true) + (0.5 * sens_false)
      MacroSpecificity = (0.5 * spec_true) + (0.5 * spec_fasle)
      MacroPrecision = (0.5 * prec_true) + (0.5 * prec_false)
      MacroRecall = (0.5 * recal_true) + (0.5 * recal_false)
      MacroF1 = (0.5 * f1_true) + (0.5 * f1_false)
      
      rnn.Hazardous.stats$Accuracy    = append(rnn.Hazardous.stats$Accuracy, rnn.Hazardous.confusion_matrix_true$overall["Accuracy"])
      rnn.Hazardous.stats$MacroSensitivity = append(rnn.Hazardous.stats$MacroSensitivity, MacroSensitivity)
      rnn.Hazardous.stats$MacroSpecificity = append(rnn.Hazardous.stats$MacroSpecificity, MacroSpecificity)
      rnn.Hazardous.stats$MacroPrecision = append(rnn.Hazardous.stats$MacroPrecision, MacroPrecision)
      rnn.Hazardous.stats$MacroRecall = append(rnn.Hazardous.stats$MacroRecall, MacroRecall)
      rnn.Hazardous.stats$MacroF1 = append(rnn.Hazardous.stats$MacroF1, MacroF1)
      rm(rnn.Hazardous.model, rnn.Hazardous.pred)
      rm(rnn.Hazardous.confusion_matrix_true,rnn.Hazardous.confusion_matrix_false,prec_true,recal_true,f1_true,prec_false,recal_false,f1_false,MacroPrecision,MacroRecall,MacroF1)
    }
    
    rnn.name <- paste("Haz",hyper.kernel,as.character(hyper.cost),sep="_")
    rnn.Hazardous$Accuracy[rnn.name] <- rnn.Hazardous.stats$Accuracy
    rnn.Hazardous$MacroSensitivity[rnn.name] <- rnn.Hazardous.stats$MacroSensitivity
    rnn.Hazardous$MacroSpecificity[rnn.name] <- rnn.Hazardous.stats$MacroSpecificity
    rnn.Hazardous$MacroPrecision[rnn.name] <- rnn.Hazardous.stats$MacroPrecision
    rnn.Hazardous$MacroRecall[rnn.name] <- rnn.Hazardous.stats$MacroRecall
    rnn.Hazardous$MacroF1[rnn.name] <- rnn.Hazardous.stats$MacroF1
    
    
    tdist <- list()
    tdist_name <- paste("Haz ",hyper.kernel,as.character(hyper.cost),sep=" ")
    tdist_val = confidence_interval(as.vector(rnn.Hazardous.stats$Accuracy),0.95)
    tdist$acc <- paste(as.character(round(tdist_val[2],4))," ± ",as.character(round(tdist_val[1],4)))
    
    tdist_val = confidence_interval(as.vector(rnn.Hazardous.stats$MacroSensitivity),0.95)
    tdist$sens <- paste(as.character(round(tdist_val[2],4))," ± ",as.character(round(tdist_val[1],4)))
    
    tdist_val = confidence_interval(as.vector(rnn.Hazardous.stats$MacroSpecificity),0.95)
    tdist$spec <- paste(as.character(round(tdist_val[2],4))," ± ",as.character(round(tdist_val[1],4)))
    
    tdist_val = confidence_interval(as.vector(rnn.Hazardous.stats$MacroPrecision),0.95)
    tdist$prec <- paste(as.character(round(tdist_val[2],4))," ± ",as.character(round(tdist_val[1],4)))
    
    tdist_val = confidence_interval(as.vector(rnn.Hazardous.stats$MacroRecall),0.95)
    tdist$rec <- paste(as.character(round(tdist_val[2],4))," ± ",as.character(round(tdist_val[1],4)))
    
    tdist_val = confidence_interval(as.vector(rnn.Hazardous.stats$MacroF1),0.95)
    tdist$f1 <- paste(as.character(round(tdist_val[2],4))," ± ",as.character(round(tdist_val[1],4)))
    
    
    rnn.Hazardous.All[tdist_name] <- c(tdist$acc,tdist$sens,tdist$spec,tdist$prec,tdist$rec,tdist$f1)
    rm(tdist_name,tdist,rnn.name,fold.train, fold.valid)
    
  }
}
end_table <- length(rnn.Hazardous$Accuracy)
img_name_plot <- paste("IMG_asteroids_model_SVM_", "Hazardous_KFOLD_performance_plot_A", ".png", sep = "")
png(img_name_plot,res = 800, height = 10, width = 15, unit='in')
par(mfrow=c(2,2)) 
boxplot(rnn.Hazardous$Accuracy[2:end_table],vertical = TRUE, pch=19, las = 2, col = 1:length(rnn.Hazardous$Accuracy),main="Accuracy")  
boxplot(rnn.Hazardous$MacroSensitivity[2:end_table],vertical = TRUE, pch=19, las = 2, col = 1:length(rnn.Hazardous$MacroSensitivity),main="MacroSensitivity")  
dev.off()
img_name_plot <- paste("IMG_asteroids_model_SVM_", "Hazardous_KFOLD_performance_plot_B", ".png", sep = "")
png(img_name_plot,res = 800, height = 10, width = 15, unit='in')
par(mfrow=c(2,2))  
boxplot(rnn.Hazardous$MacroSpecificity[2:end_table], vertical = TRUE, pch=19, las = 2, col = 1:length(rnn.Hazardous$Specificity),main="MacroSpecificity")  
boxplot(rnn.Hazardous$MacroPrecision[2:end_table], vertical = TRUE, pch=19, las = 2, col = 1:length(rnn.Hazardous$MacroPrecision),main="MacroPrecision")  
dev.off()
img_name_plot <- paste("IMG_asteroids_model_SVM_", "Hazardous_KFOLD_performance_plot_C", ".png", sep = "")
png(img_name_plot,res = 800, height = 10, width = 15, unit='in')
par(mfrow=c(2,2))  
boxplot(rnn.Hazardous$MacroRecall[2:end_table],vertical = TRUE, pch=19, las = 2, col = 1:length(rnn.Hazardous$MacroRecall),main="MacroRecall")  
boxplot(rnn.Hazardous$MacroF1[2:end_table],vertical = TRUE, pch=19, las = 2, col = 1:length(rnn.Hazardous$MacroF1),main="MacroF1")  
dev.off()

img_name_plot <- paste("PDF_asteroids_model_SVM_", "Hazardous_KFOLD_performance_tdist", ".pdf", sep = "")
pdf(img_name_plot, height = 20, width = 46)
grid.table(t(rnn.Hazardous.All[2:length(rnn.Hazardous.All)]))
dev.off()

#rm(rnn.Hazardous, rnn.Hazardous.All)

#Hazardous class
#a small cost creates a large margin (a soft margin) and allows more misclassifications;
#a large cost creates a narrow margin (a hard margin) and allows few misclassifications
#this is a soft margin rnn


#Classification type asteroids

#Amor Asteroid - Apollo Asteroid hyperpiano

astroids.subset = subset(asteroids_split$train, select=c("Perihelion.Distance..AU.", "Orbit.Axis..AU.", "Classification"),
                         Classification %in% c("Amor Asteroid","Apollo Asteroid"))
astroids.subset.plt = astroids.subset[1:1000,]

#Linear cost 1
rnn.Classification.Plot.Linear.C1 = rnn(Classification ~ ., data=astroids.subset, 
                                        kernel=hyper.kernel, cost=hyper.cost, scale=TRUE)
img_name_plot <- paste("IMG_asteroids_model_SVM_","Classification_",hyper.kernel,"_C",as.character(hyper.cost),"_Amor_Apollo_support", ".png", sep = "")
png(img_name_plot)
plot(x=astroids.subset.plt$Perihelion.Distance..AU.,y=astroids.subset.plt$Orbit.Axis..AU.,  col=astroids.subset.plt$Classification, pch=19)
points(astroids.subset.plt[rnn.Classification.Plot.Linear.C1$index,c(1,2)],col="blue",cex=2)
w = t(rnn.Classification.Plot.Linear.C1$coefs) %*% rnn.Classification.Plot.Linear.C1$SV
c = -rnn.Classification.Plot.Linear.C1$rho
abline (a=-c/w[1,2],b=-w[1,1]/w[1,2], col="red", lty=5)
abline(a=(-c-1)/w[1,2], b=-w[1,1]/w[1,2], col="orange", lty=3)
abline(a=(-c+1)/w[1,2], b=-w[1,1]/w[1,2], col="orange", lty=3)
dev.off()
rm(w, c)
rm(hyper.cost,hyper.kernel,rnn.Classification.Plot.Linear.C1)
#Linear cost 1.000
hyper.cost <- 1000
hyper.kernel <- 'linear'
rnn.Classification.Plot.Linear.C1 = rnn(Classification ~ ., data=astroids.subset, 
                                        kernel=hyper.kernel, cost=hyper.cost, scale=FALSE)
img_name_plot <- paste("IMG_asteroids_model_SVM_","Classification_",hyper.kernel,"_C",as.character(hyper.cost),"_Amor_Apollo_support", ".png", sep = "")
png(img_name_plot)
plot(x=astroids.subset.plt$Perihelion.Distance..AU.,y=astroids.subset.plt$Orbit.Axis..AU.,  col=astroids.subset.plt$Classification, pch=19)
points(astroids.subset.plt[rnn.Classification.Plot.Linear.C1$index,c(1,2)],col="blue",cex=2)
w = t(rnn.Classification.Plot.Linear.C1$coefs) %*% rnn.Classification.Plot.Linear.C1$SV
c = -rnn.Classification.Plot.Linear.C1$rho
abline (a=-c/w[1,2],b=-w[1,1]/w[1,2], col="red", lty=5)
abline(a=(-c-1)/w[1,2], b=-w[1,1]/w[1,2], col="orange", lty=3)
abline(a=(-c+1)/w[1,2], b=-w[1,1]/w[1,2], col="orange", lty=3)
dev.off()
rm(w, c)
rm(hyper.cost,hyper.kernel,rnn.Classification.Plot.Linear.C1)
#Linear cost 1.000
hyper.cost <- 10000
hyper.kernel <- 'linear'
rnn.Classification.Plot.Linear.C1 = rnn(Classification ~ ., data=astroids.subset, 
                                        kernel=hyper.kernel, cost=hyper.cost, scale=FALSE)
img_name_plot <- paste("IMG_asteroids_model_SVM_","Classification_",hyper.kernel,"_C",as.character(hyper.cost),"_Amor_Apollo_support", ".png", sep = "")
png(img_name_plot)
plot(x=astroids.subset.plt$Perihelion.Distance..AU.,y=astroids.subset.plt$Orbit.Axis..AU.,  col=astroids.subset.plt$Classification, pch=19)
points(astroids.subset.plt[rnn.Classification.Plot.Linear.C1$index,c(1,2)],col="blue",cex=2)
w = t(rnn.Classification.Plot.Linear.C1$coefs) %*% rnn.Classification.Plot.Linear.C1$SV
c = -rnn.Classification.Plot.Linear.C1$rho
abline (a=-c/w[1,2],b=-w[1,1]/w[1,2], col="red", lty=5)
abline(a=(-c-1)/w[1,2], b=-w[1,1]/w[1,2], col="orange", lty=3)
abline(a=(-c+1)/w[1,2], b=-w[1,1]/w[1,2], col="orange", lty=3)
dev.off()
rm(w, c)
rm(hyper.cost,hyper.kernel,rnn.Classification.Plot.Linear.C1)
#Linear cost 0.5
hyper.cost <- 0.5
hyper.kernel <- 'linear'
rnn.Classification.Plot.Linear.C1 = rnn(Classification ~ ., data=astroids.subset, 
                                        kernel=hyper.kernel, cost=hyper.cost, scale=FALSE)
img_name_plot <- paste("IMG_asteroids_model_SVM_","Classification_",hyper.kernel,"_C",as.character(hyper.cost),"_Amor_Apollo_support", ".png", sep = "")
png(img_name_plot)
plot(x=astroids.subset.plt$Perihelion.Distance..AU.,y=astroids.subset.plt$Orbit.Axis..AU.,  col=astroids.subset.plt$Classification, pch=19)
points(astroids.subset.plt[rnn.Classification.Plot.Linear.C1$index,c(1,2)],col="blue",cex=2)
w = t(rnn.Classification.Plot.Linear.C1$coefs) %*% rnn.Classification.Plot.Linear.C1$SV
c = -rnn.Classification.Plot.Linear.C1$rho
abline (a=-c/w[1,2],b=-w[1,1]/w[1,2], col="red", lty=5)
abline(a=(-c-1)/w[1,2], b=-w[1,1]/w[1,2], col="orange", lty=3)
abline(a=(-c+1)/w[1,2], b=-w[1,1]/w[1,2], col="orange", lty=3)
dev.off()
rm(w, c)
rm(hyper.cost,hyper.kernel,rnn.Classification.Plot.Linear.C1)

#Poly  cost 1 
hyper.cost <- 1
hyper.kernel <- 'polynomial'
rnn.Classification.Plot.Linear.C1 = rnn(Classification ~ ., data=astroids.subset, 
                                        kernel=hyper.kernel, cost=hyper.cost, scale=TRUE)
img_name_plot <- paste("IMG_asteroids_model_SVM_","Classification_",hyper.kernel,"_C",as.character(hyper.cost),"_Amor_Apollo_support", ".png", sep = "")
png(img_name_plot)
plot(x=astroids.subset.plt$Perihelion.Distance..AU.,y=astroids.subset.plt$Orbit.Axis..AU.,  col=astroids.subset.plt$Classification, pch=19)
points(astroids.subset.plt[rnn.Classification.Plot.Linear.C1$index,c(1,2)],col="blue",cex=2)
w = t(rnn.Classification.Plot.Linear.C1$coefs) %*% rnn.Classification.Plot.Linear.C1$SV
c = -rnn.Classification.Plot.Linear.C1$rho
abline (a=-c/w[1,2],b=-w[1,1]/w[1,2], col="red", lty=5)
abline(a=(-c-1)/w[1,2], b=-w[1,1]/w[1,2], col="orange", lty=3)
abline(a=(-c+1)/w[1,2], b=-w[1,1]/w[1,2], col="orange", lty=3)
dev.off()
rm(w, c)
rm(hyper.cost,hyper.kernel,rnn.Classification.Plot.Linear.C1)

#Poly  cost 1 
hyper.cost <- 1
hyper.kernel <- 'radial'
rnn.Classification.Plot.Linear.C1 = rnn(Classification ~ ., data=astroids.subset, 
                                        kernel=hyper.kernel, cost=hyper.cost, scale=TRUE)
img_name_plot <- paste("IMG_asteroids_model_SVM_","Classification_",hyper.kernel,"_C",as.character(hyper.cost),"_Amor_Apollo_support", ".png", sep = "")
png(img_name_plot)
plot(x=astroids.subset.plt$Perihelion.Distance..AU.,y=astroids.subset.plt$Orbit.Axis..AU.,  col=astroids.subset.plt$Classification, pch=19)
points(astroids.subset.plt[rnn.Classification.Plot.Linear.C1$index,c(1,2)],col="blue",cex=2)
w = t(rnn.Classification.Plot.Linear.C1$coefs) %*% rnn.Classification.Plot.Linear.C1$SV
c = -rnn.Classification.Plot.Linear.C1$rho
abline (a=-c/w[1,2],b=-w[1,1]/w[1,2], col="red", lty=5)
abline(a=(-c-1)/w[1,2], b=-w[1,1]/w[1,2], col="orange", lty=3)
abline(a=(-c+1)/w[1,2], b=-w[1,1]/w[1,2], col="orange", lty=3)
dev.off()
rm(w, c)
rm(hyper.cost,hyper.kernel,rnn.Classification.Plot.Linear.C1)

#SMV CLASSIFICATION

folds.number = 5
kernel_list <- c('linear','polynomial','radial')
C_list <- c(0.5,1,10)

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

mx = matrix(NA, nrow = 6)
rnn.Classification.All <- data.frame(mx)
rnn.Classification.All["Performance"] = c("Accuracy","MacroSensitivity","MacroSpecificity","MacroPrecision","MacroRecall","MacroF1")
rm(mx)


for (j in 1:length(kernel_list)) {
  hyper.kernel <- kernel_list[j]
  
  for (l in 1:length(C_list)) {
    hyper.cost <- C_list[l]
    rnn.Classification.stats <- list()
    
    for (i in 1:length(folds)) {
      fold.valid <- ldply(folds[i], data.frame)
      fold.valid <- fold.valid[, !names(fold.valid) %in% c(".id")]
      fold.train <- ldply(folds[-i], data.frame)
      fold.train <- fold.train[, !names(fold.train) %in% c(".id")]
      
      rnn.Classification$performance[hyper.kernel][hyper.cost] <- list()
      print(paste(as.character(i), hyper.kernel , as.character(hyper.cost), sep = " "))
      rnn.Classification.model = rnn(Classification ~ Orbit.Axis..AU. + Orbit.Eccentricity + Orbit.Inclination..deg. + Perihelion.Argument..deg. + Node.Longitude..deg. + Mean.Anomoly..deg. + Perihelion.Distance..AU. + Aphelion.Distance..AU. + Orbital.Period..yr. + Minimum.Orbit.Intersection.Distance..AU. + Asteroid.Magnitude,
                                     data=fold.train, kernel=hyper.kernel, cost=hyper.cost)
      rnn.Classification.pred = predict(rnn.Classification.model, fold.valid)
      
      rnn.Classification.confusion_matrix_multiclass = confusionMatrix(
        rnn.Classification.pred, fold.valid$Classification, mode = "prec_recall") 
      
      
      
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
      
      MacroSensitivity = (0.25 * sens_Amor) + (0.25 * sens_Apohele) + (0.25 * sens_Apollo) + (0.25 * sens_Aten)
      MacroSpecificity = (0.25 * spec_Amor) + (0.25 * spec_Apohele) + (0.25 * spec_Apollo) + (0.25 * spec_Aten)
      MacroPrecision = (0.25 * prec_Amor) + (0.25 * prec_Apohele) + (0.25 * prec_Apollo) + (0.25 * prec_Aten)
      MacroRecall = (0.25 * recal_Amor) + (0.25 * recal_Apohele) + (0.25 * recal_Apollo) + (0.25 * recal_Aten)
      MacroF1 = (0.25 * f1_Amor) + (0.25 * f1_Apohele) + (0.25 * f1_Apollo) + (0.25 * prec_Aten)
      
      rnn.Classification.stats$Accuracy    = append(rnn.Classification.stats$Accuracy, rnn.Classification.confusion_matrix_multiclass$overall["Accuracy"])
      rnn.Classification.stats$MacroSensitivity = append(rnn.Classification.stats$MacroSensitivity, MacroSensitivity)
      rnn.Classification.stats$MacroSpecificity = append(rnn.Classification.stats$MacroSpecificity, MacroSpecificity)
      rnn.Classification.stats$MacroPrecision = append(rnn.Classification.stats$MacroPrecision, MacroPrecision)
      rnn.Classification.stats$MacroRecall = append(rnn.Classification.stats$MacroRecall, MacroRecall)
      rnn.Classification.stats$MacroF1 = append(rnn.Classification.stats$MacroF1, MacroF1)
      rm(rnn.Classification.model, rnn.Classification.pred)
      rm(rnn.Classification.confusion_matrix_true,rnn.Classification.confusion_matrix_false,prec_true,recal_true,f1_true,prec_false,recal_false,f1_false,MacroPrecision,MacroRecall,MacroF1)
    }
    
    rnn.name <- paste("Haz",hyper.kernel,as.character(hyper.cost),sep="_")
    rnn.Classification$Accuracy[rnn.name] <- rnn.Classification.stats$Accuracy
    rnn.Classification$MacroSensitivity[rnn.name] <- rnn.Classification.stats$MacroSensitivity
    rnn.Classification$MacroSpecificity[rnn.name] <- rnn.Classification.stats$MacroSpecificity
    rnn.Classification$MacroPrecision[rnn.name] <- rnn.Classification.stats$MacroPrecision
    rnn.Classification$MacroRecall[rnn.name] <- rnn.Classification.stats$MacroRecall
    rnn.Classification$MacroF1[rnn.name] <- rnn.Classification.stats$MacroF1
    
    
    tdist <- list()
    tdist_name <- paste("Class ",hyper.kernel,as.character(hyper.cost),sep=" ")
    tdist_val = confidence_interval(as.vector(rnn.Classification.stats$Accuracy),0.95)
    tdist$acc <- paste(as.character(round(tdist_val[2],4))," ± ",as.character(round(tdist_val[1],4)))
    
    tdist_val = confidence_interval(as.vector(rnn.Classification.stats$MacroSensitivity),0.95)
    tdist$sens <- paste(as.character(round(tdist_val[2],4))," ± ",as.character(round(tdist_val[1],4)))
    
    tdist_val = confidence_interval(as.vector(rnn.Classification.stats$MacroSpecificity),0.95)
    tdist$spec <- paste(as.character(round(tdist_val[2],4))," ± ",as.character(round(tdist_val[1],4)))
    
    tdist_val = confidence_interval(as.vector(rnn.Classification.stats$MacroPrecision),0.95)
    tdist$prec <- paste(as.character(round(tdist_val[2],4))," ± ",as.character(round(tdist_val[1],4)))
    
    tdist_val = confidence_interval(as.vector(rnn.Classification.stats$MacroRecall),0.95)
    tdist$rec <- paste(as.character(round(tdist_val[2],4))," ± ",as.character(round(tdist_val[1],4)))
    
    tdist_val = confidence_interval(as.vector(rnn.Classification.stats$MacroF1),0.95)
    tdist$f1 <- paste(as.character(round(tdist_val[2],4))," ± ",as.character(round(tdist_val[1],4)))
    
    
    rnn.Classification.All[tdist_name] <- c(tdist$acc,tdist$sens,tdist$spec,tdist$prec,tdist$rec,tdist$f1)
    rm(tdist_name,tdist,rnn.name,fold.train, fold.valid)
    
  }
}
end_table <- length(rnn.Classification$Accuracy)
img_name_plot <- paste("IMG_asteroids_model_SVM_", "Classification_KFOLD_performance_plot_A", ".png", sep = "")
png(img_name_plot,res = 800, height = 10, width = 15, unit='in')
par(mfrow=c(2,2)) 
boxplot(rnn.Classification$Accuracy[2:end_table],vertical = TRUE, pch=19, las = 2, col = 1:length(rnn.Classification$Accuracy),main="Accuracy")  
boxplot(rnn.Classification$MacroSensitivity[2:end_table],vertical = TRUE, pch=19, las = 2, col = 1:length(rnn.Classification$MacroSensitivity),main="MacroSensitivity")  
dev.off()
img_name_plot <- paste("IMG_asteroids_model_SVM_", "Classification_KFOLD_performance_plot_B", ".png", sep = "")
png(img_name_plot,res = 800, height = 10, width = 15, unit='in')
par(mfrow=c(2,2))  
boxplot(rnn.Classification$MacroSpecificity[2:end_table], vertical = TRUE, pch=19, las = 2, col = 1:length(rnn.Classification$MacroSpecificity),main="MacroSpecificity")  
boxplot(rnn.Classification$MacroPrecision[2:end_table], vertical = TRUE, pch=19, las = 2, col = 1:length(rnn.Classification$MacroPrecision),main="MacroPrecision")  
dev.off()
img_name_plot <- paste("IMG_asteroids_model_SVM_", "Classification_KFOLD_performance_plot_C", ".png", sep = "")
png(img_name_plot,res = 800, height = 10, width = 15, unit='in')
par(mfrow=c(2,2))  
boxplot(rnn.Classification$MacroRecall[2:end_table],vertical = TRUE, pch=19, las = 2, col = 1:length(rnn.Classification$MacroRecall),main="MacroRecall")  
boxplot(rnn.Classification$MacroF1[2:end_table],vertical = TRUE, pch=19, las = 2, col = 1:length(rnn.Classification$MacroF1),main="MacroF1")  
dev.off()

img_name_plot <- paste("PDF_asteroids_model_SVM_", "Classification_KFOLD_performance_tdist", ".pdf", sep = "")
pdf(img_name_plot, height = 20, width = 46)
grid.table(t(rnn.Classification.All[2:length(rnn.Classification.All)]))
dev.off()

