setwd("~/Github/AsteroidsClassification")

library(rpart)
library(rattle)
library(rpart.plot)
library(RColorBrewer)
library(plyr)
library(gridExtra)
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
folds.number = 10

folds <- split(asteroids_split$train, cut(sample(1:nrow(asteroids_split$train)), folds.number))
rm(folds.number)

#Hazardous
folds.stats.namesplit <- list()
folds.stats.Hazardous.trainsplit <- list()
folds.stats.Hazardous.validsplit <- list()

decisionTree.Hazardous.GINI.stats <- NULL
decisionTree.Hazardous.IGHE.stats <- NULL

mx = matrix(NA, nrow = length(folds))
dt.Hazardous <- list()
dt.Hazardous$Accuracy = data.frame(mx)
dt.Hazardous$MacroSensitivity <- data.frame(mx)
dt.Hazardous$MacroSpecificity <- data.frame(mx)
dt.Hazardous$MacroPrecision <- data.frame(mx)
dt.Hazardous$MacroRecall <- data.frame(mx)
dt.Hazardous$MacroF1 <- data.frame(mx)
mx = matrix(NA, nrow = 6)
dt.Hazardous.All <- data.frame(mx)
dt.Hazardous.All["Performance"] = c("Accuracy","MacroSensitivity","Specificity","MacroPrecision","MacroRecall","MacroF1")

dt.Hazardous.GINI.foldstats <- list()
dt.Hazardous.IGHE.foldstats <- list()

for (i in 1:length(folds)) {
  fold.valid <- ldply(folds[i], data.frame)
  fold.valid <- fold.valid[ , !names(fold.valid) %in% c(".id")]
  fold.train <- ldply(folds[-i], data.frame)
  fold.train <- fold.train[ , !names(fold.train) %in% c(".id")]
  
  name_fold <- paste("fold", as.character(i), sep = "")
  
  folds.stats.namesplit <- append(folds.stats.namesplit, name_fold)
  
  folds.stats.Hazardous.trainsplit <- append(folds.stats.Hazardous.trainsplit, round(sum(fold.train$Hazardous, na.rm = TRUE)/length(fold.train$Hazardous)*100, 4))
  folds.stats.Hazardous.validsplit <- append(folds.stats.Hazardous.validsplit, round(sum(fold.valid$Hazardous, na.rm = TRUE)/length(fold.valid$Hazardous)*100, 4))
  
  decisionTree.Hazardous.GINI <- rpart(Hazardous.int ~ Orbit.Axis..AU. + Orbit.Eccentricity + Orbit.Inclination..deg. + Perihelion.Argument..deg. + Node.Longitude..deg. + Mean.Anomoly..deg. + Perihelion.Distance..AU. + Aphelion.Distance..AU. + Orbital.Period..yr. + Minimum.Orbit.Intersection.Distance..AU. + Asteroid.Magnitude,#Orbit.Axis..AU. + Orbit.Eccentricity + Orbit.Inclination..deg. + Perihelion.Argument..deg. + Node.Longitude..deg. + Mean.Anomoly..deg. + Perihelion.Distance..AU. + Aphelion.Distance..AU. + Orbital.Period..yr. + Minimum.Orbit.Intersection.Distance..AU. + Asteroid.Magnitude,
    data=fold.train, method="class", cp= 0.001) #. all var

  decisionTree.Hazardous.IGHE <- rpart(Hazardous.int ~ Orbit.Axis..AU. + Orbit.Eccentricity + Orbit.Inclination..deg. + Perihelion.Argument..deg. + Node.Longitude..deg. + Mean.Anomoly..deg. + Perihelion.Distance..AU. + Aphelion.Distance..AU. + Orbital.Period..yr. + Minimum.Orbit.Intersection.Distance..AU. + Asteroid.Magnitude,#Orbit.Axis..AU. + Orbit.Eccentricity + Orbit.Inclination..deg. + Perihelion.Argument..deg. + Node.Longitude..deg. + Mean.Anomoly..deg. + Perihelion.Distance..AU. + Aphelion.Distance..AU. + Orbital.Period..yr. + Minimum.Orbit.Intersection.Distance..AU. + Asteroid.Magnitude,
    data=fold.train, method="class", parms = list(split="information"), cp= 0.001) #. all var
  
  
  #salvo anche la comple 
  #altezza
  
  decisionTree.Hazardous.GINI.prediction <- predict(decisionTree.Hazardous.GINI, fold.valid, type = "class")

  decisionTree.Hazardous.GINI.confusion_matrix_true = confusionMatrix(
    decisionTree.Hazardous.GINI.prediction, fold.valid$Hazardous.int, positive="TRUE", mode = "prec_recall") 
  decisionTree.Hazardous.GINI.confusion_matrix_false = confusionMatrix(
    decisionTree.Hazardous.GINI.prediction, fold.valid$Hazardous.int, positive="FALSE", mode = "prec_recall") 
  
  sens_GINI_true = decisionTree.Hazardous.GINI.confusion_matrix_true$byClass["Sensitivity"]
  spec_GINI_true= decisionTree.Hazardous.GINI.confusion_matrix_true$byClass["Specificity"]
  prec_GINI_true= decisionTree.Hazardous.GINI.confusion_matrix_true$byClass["Precision"]
  recal_GINI_true= decisionTree.Hazardous.GINI.confusion_matrix_true$byClass["Recall"]
  f1_GINI_true= decisionTree.Hazardous.GINI.confusion_matrix_true$byClass["F1"]

  sens_GINI_false = decisionTree.Hazardous.GINI.confusion_matrix_false$byClass["Sensitivity"]
  spec_GINI_false = decisionTree.Hazardous.GINI.confusion_matrix_false$byClass["Specificity"]
  prec_GINI_false = decisionTree.Hazardous.GINI.confusion_matrix_false$byClass["Precision"]
  recal_GINI_false = decisionTree.Hazardous.GINI.confusion_matrix_false$byClass["Recall"]
  f1_GINI_false = decisionTree.Hazardous.GINI.confusion_matrix_false$byClass["F1"]
  
  MacroSensitivity = (0.5 * sens_GINI_true) + (0.5 * sens_GINI_false)
  MacroSpecificity = (0.5 * spec_GINI_true) + (0.5 * spec_GINI_false)
  MacroPrecision = (0.5 * prec_GINI_true) + (0.5 * prec_GINI_false)
  MacroRecall = (0.5 * recal_GINI_true) + (0.5 * recal_GINI_false)
  MacroF1 = (0.5 * f1_GINI_true) + (0.5 * f1_GINI_false)
             
    dt.Hazardous.GINI.foldstats$Accuracy    = append(dt.Hazardous.GINI.foldstats$Accuracy, decisionTree.Hazardous.GINI.confusion_matrix_true$overall["Accuracy"])
    dt.Hazardous.GINI.foldstats$MacroSensitivity = append(dt.Hazardous.GINI.foldstats$MacroSensitivity, MacroSensitivity)
   dt.Hazardous.GINI.foldstats$MacroSpecificity = append(dt.Hazardous.GINI.foldstats$MacroSpecificity, MacroSpecificity)
   dt.Hazardous.GINI.foldstats$MacroPrecision = append(dt.Hazardous.GINI.foldstats$MacroPrecision, MacroPrecision)
   dt.Hazardous.GINI.foldstats$MacroRecall = append(dt.Hazardous.GINI.foldstats$MacroRecall, MacroRecall)
   dt.Hazardous.GINI.foldstats$MacroF1 = append(dt.Hazardous.GINI.foldstats$MacroF1, MacroF1)
  
   
   
   #IGHE
   decisionTree.Hazardous.IGHE.prediction <- predict(decisionTree.Hazardous.IGHE, fold.valid, type = "class")
   
   decisionTree.Hazardous.IGHE.confusion_matrix_true = confusionMatrix(
     decisionTree.Hazardous.IGHE.prediction, fold.valid$Hazardous.int, positive="TRUE", mode = "prec_recall") 
   decisionTree.Hazardous.IGHE.confusion_matrix_false = confusionMatrix(
     decisionTree.Hazardous.IGHE.prediction, fold.valid$Hazardous.int, positive="FALSE", mode = "prec_recall") 
   
   sens_IGHE_true = decisionTree.Hazardous.IGHE.confusion_matrix_true$byClass["Sensitivity"]
   spec_IGHE_true= decisionTree.Hazardous.IGHE.confusion_matrix_true$byClass["Specificity"]
   prec_IGHE_true= decisionTree.Hazardous.IGHE.confusion_matrix_true$byClass["Precision"]
   recal_IGHE_true= decisionTree.Hazardous.IGHE.confusion_matrix_true$byClass["Recall"]
   f1_IGHE_true= decisionTree.Hazardous.IGHE.confusion_matrix_true$byClass["F1"]
   
   sens_IGHE_false = decisionTree.Hazardous.IGHE.confusion_matrix_false$byClass["Sensitivity"]
   spec_IGHE_false = decisionTree.Hazardous.IGHE.confusion_matrix_false$byClass["Specificity"]
   prec_IGHE_false = decisionTree.Hazardous.IGHE.confusion_matrix_false$byClass["Precision"]
   recal_IGHE_false = decisionTree.Hazardous.IGHE.confusion_matrix_false$byClass["Recall"]
   f1_IGHE_false = decisionTree.Hazardous.IGHE.confusion_matrix_false$byClass["F1"]
   
   sens_IGHE_true = decisionTree.Hazardous.IGHE.confusion_matrix_true$byClass["Sensitivity"]
   spec_IGHE_true= decisionTree.Hazardous.IGHE.confusion_matrix_true$byClass["Specificity"]
   prec_IGHE_true= decisionTree.Hazardous.IGHE.confusion_matrix_true$byClass["Precision"]
   recal_IGHE_true= decisionTree.Hazardous.IGHE.confusion_matrix_true$byClass["Recall"]
   f1_IGHE_true= decisionTree.Hazardous.IGHE.confusion_matrix_true$byClass["F1"]
   
   sens_IGHE_false = decisionTree.Hazardous.IGHE.confusion_matrix_false$byClass["Sensitivity"]
   spec_IGHE_false = decisionTree.Hazardous.IGHE.confusion_matrix_false$byClass["Specificity"]
   prec_IGHE_false = decisionTree.Hazardous.IGHE.confusion_matrix_false$byClass["Precision"]
   recal_IGHE_false = decisionTree.Hazardous.IGHE.confusion_matrix_false$byClass["Recall"]
   f1_IGHE_false = decisionTree.Hazardous.IGHE.confusion_matrix_false$byClass["F1"]
   
   MacroSensitivity = (0.5 * sens_IGHE_true) + (0.5 * sens_IGHE_false)
   MacroSpecificity = (0.5 * spec_IGHE_true) + (0.5 * spec_IGHE_false)
   MacroPrecision = (0.5 * prec_IGHE_true) + (0.5 * prec_IGHE_false)
   MacroRecall = (0.5 * recal_IGHE_true) + (0.5 * recal_IGHE_false)
   MacroF1 = (0.5 * f1_IGHE_true) + (0.5 * f1_IGHE_false)
              
   dt.Hazardous.IGHE.foldstats$Accuracy    = append(dt.Hazardous.IGHE.foldstats$Accuracy, decisionTree.Hazardous.IGHE.confusion_matrix_true$overall["Accuracy"])
   dt.Hazardous.IGHE.foldstats$MacroSensitivity = append(dt.Hazardous.IGHE.foldstats$MacroSensitivity, MacroSensitivity)
   dt.Hazardous.IGHE.foldstats$MacroSpecificity = append(dt.Hazardous.IGHE.foldstats$MacroSpecificity, MacroSpecificity)
   dt.Hazardous.IGHE.foldstats$MacroPrecision = append(dt.Hazardous.IGHE.foldstats$MacroPrecision, MacroPrecision)
   dt.Hazardous.IGHE.foldstats$MacroRecall = append(dt.Hazardous.IGHE.foldstats$MacroRecall, MacroRecall)
   dt.Hazardous.IGHE.foldstats$MacroF1 = append(dt.Hazardous.IGHE.foldstats$MacroF1, MacroF1)
  
  decisionTree.Hazardous.GINI.confusion_matrix = table(fold.valid$Hazardous, decisionTree.Hazardous.GINI.prediction)
  decisionTree.Hazardous.IGHE.confusion_matrix = table(fold.valid$Hazardous, decisionTree.Hazardous.IGHE.prediction)
  
  if (is.null(decisionTree.Hazardous.GINI.stats)){
    decisionTree.Hazardous.GINI.stats = decisionTree.Hazardous.GINI.confusion_matrix
  }else{
    decisionTree.Hazardous.GINI.stats <- decisionTree.Hazardous.GINI.stats + decisionTree.Hazardous.GINI.confusion_matrix
  }
  
  if (is.null(decisionTree.Hazardous.IGHE.stats)){
    decisionTree.Hazardous.IGHE.stats = decisionTree.Hazardous.IGHE.confusion_matrix
  }else{
    decisionTree.Hazardous.IGHE.stats <- decisionTree.Hazardous.IGHE.stats + decisionTree.Hazardous.IGHE.confusion_matrix
  }
  rm(name_fold, fold.train, fold.valid)
}

dt.GINI.name <- paste("Hazardous GINI")
dt.Hazardous$Accuracy[dt.GINI.name] <- dt.Hazardous.GINI.foldstats$Accuracy
dt.Hazardous$MacroSensitivity[dt.GINI.name] <- dt.Hazardous.GINI.foldstats$MacroSensitivity
dt.Hazardous$MacroSpecificity[dt.GINI.name] <- dt.Hazardous.GINI.foldstats$MacroSpecificity
dt.Hazardous$MacroPrecision[dt.GINI.name] <- dt.Hazardous.GINI.foldstats$MacroPrecision
dt.Hazardous$MacroRecall[dt.GINI.name] <- dt.Hazardous.GINI.foldstats$MacroRecall
dt.Hazardous$MacroF1[dt.GINI.name] <- dt.Hazardous.GINI.foldstats$MacroF1
tdist <- list()

tdist_val = confidence_interval(as.vector(dt.Hazardous.GINI.foldstats$Accuracy),0.95)
tdist$acc <- paste(as.character(round(tdist_val[2],4))," ± ",as.character(round(tdist_val[1],4)))

tdist_val = confidence_interval(as.vector(dt.Hazardous.GINI.foldstats$MacroSensitivity),0.95)
tdist$sens <- paste(as.character(round(tdist_val[2],4))," ± ",as.character(round(tdist_val[1],4)))

tdist_val = confidence_interval(as.vector(dt.Hazardous.GINI.foldstats$MacroSpecificity),0.95)
tdist$spec <- paste(as.character(round(tdist_val[2],4))," ± ",as.character(round(tdist_val[1],4)))

tdist_val = confidence_interval(as.vector(dt.Hazardous.GINI.foldstats$MacroPrecision),0.95)
tdist$prec <- paste(as.character(round(tdist_val[2],4))," ± ",as.character(round(tdist_val[1],4)))

tdist_val = confidence_interval(as.vector(dt.Hazardous.GINI.foldstats$MacroRecall),0.95)
tdist$rec <- paste(as.character(round(tdist_val[2],4))," ± ",as.character(round(tdist_val[1],4)))

tdist_val = confidence_interval(as.vector(dt.Hazardous.GINI.foldstats$MacroF1),0.95)
tdist$f1 <- paste(as.character(round(tdist_val[2],4))," ± ",as.character(round(tdist_val[1],4)))

dt.Hazardous.All[dt.GINI.name] <- c(tdist$acc,tdist$sens,tdist$spec,tdist$prec,tdist$rec,tdist$f1)


dt.IGHE.name <- paste("Hazardous IGHE")
dt.Hazardous$Accuracy[dt.IGHE.name] <- dt.Hazardous.IGHE.foldstats$Accuracy
dt.Hazardous$MacroSensitivity[dt.IGHE.name] <- dt.Hazardous.IGHE.foldstats$MacroSensitivity
dt.Hazardous$MacroSpecificity[dt.IGHE.name] <- dt.Hazardous.IGHE.foldstats$MacroSpecificity
dt.Hazardous$MacroPrecision[dt.IGHE.name] <- dt.Hazardous.IGHE.foldstats$MacroPrecision
dt.Hazardous$MacroRecall[dt.IGHE.name] <- dt.Hazardous.IGHE.foldstats$MacroRecall
dt.Hazardous$MacroF1[dt.IGHE.name] <- dt.Hazardous.IGHE.foldstats$MacroF1

tdist <- list()

tdist_val = confidence_interval(as.vector(dt.Hazardous.IGHE.foldstats$Accuracy),0.95)
tdist$acc <- paste(as.character(round(tdist_val[2],4))," ± ",as.character(round(tdist_val[1],4)))

tdist_val = confidence_interval(as.vector(dt.Hazardous.IGHE.foldstats$MacroSensitivity),0.95)
tdist$sens <- paste(as.character(round(tdist_val[2],4))," ± ",as.character(round(tdist_val[1],4)))

tdist_val = confidence_interval(as.vector(dt.Hazardous.IGHE.foldstats$MacroSpecificity),0.95)
tdist$spec <- paste(as.character(round(tdist_val[2],4))," ± ",as.character(round(tdist_val[1],4)))

tdist_val = confidence_interval(as.vector(dt.Hazardous.IGHE.foldstats$MacroPrecision),0.95)
tdist$prec <- paste(as.character(round(tdist_val[2],4))," ± ",as.character(round(tdist_val[1],4)))

tdist_val = confidence_interval(as.vector(dt.Hazardous.IGHE.foldstats$MacroRecall),0.95)
tdist$rec <- paste(as.character(round(tdist_val[2],4))," ± ",as.character(round(tdist_val[1],4)))

tdist_val = confidence_interval(as.vector(dt.Hazardous.IGHE.foldstats$MacroF1),0.95)
tdist$f1 <- paste(as.character(round(tdist_val[2],4))," ± ",as.character(round(tdist_val[1],4)))

dt.Hazardous.All[dt.IGHE.name] <- c(tdist$acc,tdist$sens,tdist$spec,tdist$prec,tdist$rec,tdist$f1)

end_table <- length(dt.Hazardous$Accuracy)
img_name_plot <- paste("IMG_asteroids_model_DecisionTree_", "Hazardous_KFOLD_performance_plot_A", ".png", sep = "")
png(img_name_plot,res = 800, height = 10, width = 15, unit='in')
par(mfrow=c(2,2)) 
boxplot(dt.Hazardous$Accuracy[2:end_table],vertical = TRUE, pch=19, las = 2, col = 1:length(dt.Hazardous$Accuracy),main="Accuracy")  
boxplot(dt.Hazardous$MacroSensitivity[2:end_table],vertical = TRUE, pch=19, las = 2, col = 1:length(dt.Hazardous$MacroSensitivity),main="MacroSensitivity")  
dev.off()
img_name_plot <- paste("IMG_asteroids_model_DecisionTree_", "Hazardous_KFOLD_performance_plot_B", ".png", sep = "")
png(img_name_plot,res = 800, height = 10, width = 15, unit='in')
par(mfrow=c(2,2))  
boxplot(dt.Hazardous$MacroSpecificity[2:end_table], vertical = TRUE, pch=19, las = 2, col = 1:length(dt.Hazardous$Specificity),main="MacroSpecificity")  
boxplot(dt.Hazardous$MacroPrecision[2:end_table], vertical = TRUE, pch=19, las = 2, col = 1:length(dt.Hazardous$MacroPrecision),main="MacroPrecision")  
dev.off()
img_name_plot <- paste("IMG_asteroids_model_DecisionTree_", "Hazardous_KFOLD_performance_plot_C", ".png", sep = "")
png(img_name_plot,res = 800, height = 10, width = 15, unit='in')
par(mfrow=c(2,2))  
boxplot(dt.Hazardous$MacroRecall[2:end_table],vertical = TRUE, pch=19, las = 2, col = 1:length(dt.Hazardous$MacroRecall),main="MacroRecall")  
boxplot(dt.Hazardous$MacroF1[2:end_table],vertical = TRUE, pch=19, las = 2, col = 1:length(dt.Hazardous$MacroF1),main="MacroF1")  
dev.off()

img_name_plot <- paste("PDF_asteroids_model_DecisionTree_", "Hazardous_KFOLD_performance_tdist", ".pdf", sep = "")
pdf(img_name_plot, height = 20, width = 46)
grid.table(t(dt.Hazardous.All[2:length(dt.Hazardous.All)]))
dev.off()

rm(i)
decisionTree.Hazardous.GINI.stats <- decisionTree.Hazardous.GINI.stats / length(folds)
decisionTree.Hazardous.IGHE.stats <- decisionTree.Hazardous.IGHE.stats / length(folds)

img_name_plot <- paste("IMG_asteroids_model_decisiontree_tree_", "GINI_Hazardous_confusion" ,".png", sep = "")
png(img_name_plot)
  grid.table(decisionTree.Hazardous.GINI.stats)
  dev.off()

img_name_plot <- paste("IMG_asteroids_model_decisiontree_tree_", "IGHE_Hazardous_confusion" ,".png", sep = "")
png(img_name_plot)
  grid.table(decisionTree.Hazardous.IGHE.stats)
  dev.off()

img_name_plot <- paste("IMG_asteroids_model_decisiontree_tree", "Hazardous_fold_split", ".png", sep = "")
png(img_name_plot)
  par(mfrow=c(2,2)) 
  barplot(as.numeric(folds.stats.Hazardous.trainsplit),main="Train",names.arg = folds.stats.namesplit, col='#a71e3b')
  barplot(as.numeric(folds.stats.Hazardous.validsplit),main="Valid",names.arg = folds.stats.namesplit, col='#a71e3b')
  dev.off()

img_name_plot <- paste("IMG_asteroids_model_decisiontree_tree_", "Hazardous_CP_TABLE" ,".png", sep = "")
  png(img_name_plot)
  printcp(decisionTree.Hazardous.GINI)
  dev.off()
  
img_name_plot <- paste("IMG_asteroids_model_decisiontree_tree_", "Hazardous_CP_PLOT" ,".png", sep = "")
  png(img_name_plot)
  plotcp(decisionTree.Hazardous.GINI)
  dev.off()

  #prunedDecisionTree = prune(decisionTree, cp= .011)


img_name_plot <- paste("IMG_asteroids_model_decisiontree_tree_","Hazardous_FANCY_PLOT", ".png", sep = "")
png(img_name_plot)
  fancyRpartPlot(decisionTree.Hazardous.GINI)
  dev.off()

rm(folds.stats.namesplit, folds.stats.Hazardous.trainsplit, folds.stats.Hazardous.validsplit, decisionTree.Hazardous.GINI.stats, decisionTree.Hazardous.IGHE.stats)
rm(decisionTree.Hazardous.GINI, decisionTree.Hazardous.IGHE)
rm(decisionTree.Hazardous.GINI.prediction, decisionTree.Hazardous.IGHE.prediction, decisionTree.Hazardous.GINI.confusion_matrix, decisionTree.Hazardous.IGHE.confusion_matrix)


## Classification Asteroids
folds.stats.namesplit <- list()

folds.stats.Classification.trainsplit.Amor <- list()
folds.stats.Classification.validsplit.Amor <- list()
folds.stats.Classification.trainsplit.Apohele <- list()
folds.stats.Classification.validsplit.Apohele <- list()
folds.stats.Classification.trainsplit.Apollo <- list()
folds.stats.Classification.validsplit.Apollo <- list()
folds.stats.Classification.trainsplit.Aten <- list()
folds.stats.Classification.validsplit.Aten <- list()

decisionTree.Classification.GINI.stats <- NULL
decisionTree.Classification.IGHE.stats <- NULL

mx = matrix(NA, nrow = length(folds))
dt.Classification <- list()
dt.Classification$Accuracy = data.frame(mx)
dt.Classification$MacroSensitivity <- data.frame(mx)
dt.Classification$MacroSpecificity <- data.frame(mx)
dt.Classification$MacroPrecision <- data.frame(mx)
dt.Classification$MacroRecall <- data.frame(mx)
dt.Classification$MacroF1 <- data.frame(mx)

mx = matrix(NA, nrow = 6)
dt.Classification.All <- data.frame(mx)
dt.Classification.All["Performance"] = c("Accuracy","MacroSensitivity","MacroSpecificity","MacroPrecision","MacroRecall","MacroF1")
rm(mx)

dt.Classification.GINI.foldstats <- list()
dt.Classification.IGHE.foldstats <- list()

for (i in 1:length(folds)) {
  fold.valid <- ldply(folds[i], data.frame)
  fold.valid <- fold.valid[ , !names(fold.valid) %in% c(".id")]
  fold.train <- ldply(folds[-i], data.frame)
  fold.train <- fold.train[ , !names(fold.train) %in% c(".id")]
  
  name_fold <- paste("fold", as.character(i), sep = "")
  
  folds.stats.namesplit <- append(folds.stats.namesplit, name_fold)

  folds.stats.Classification.trainsplit.Amor <- append(folds.stats.Classification.trainsplit.Amor, round(sum(fold.train$Classification == "Amor Asteroid")/length(fold.train$Classification)*100, 4))
  folds.stats.Classification.validsplit.Amor <- append(folds.stats.Classification.validsplit.Amor, round(sum(fold.valid$Classification == "Amor Asteroid")/length(fold.valid$Classification)*100, 4))
  
  folds.stats.Classification.trainsplit.Apohele <- append(folds.stats.Classification.trainsplit.Apohele, round(sum(fold.train$Classification == "Apohele Asteroid")/length(fold.train$Classification)*100, 4))
  folds.stats.Classification.validsplit.Apohele <- append(folds.stats.Classification.validsplit.Apohele, round(sum(fold.valid$Classification == "Apohele Asteroid")/length(fold.valid$Classification)*100, 4))
  
  folds.stats.Classification.trainsplit.Apollo <- append(folds.stats.Classification.trainsplit.Apollo, round(sum(fold.train$Classification == "Apollo Asteroid")/length(fold.train$Classification)*100, 4))
  folds.stats.Classification.validsplit.Apollo <- append(folds.stats.Classification.validsplit.Apollo, round(sum(fold.valid$Classification == "Apollo Asteroid")/length(fold.valid$Classification)*100, 4))
  
  folds.stats.Classification.trainsplit.Aten <- append(folds.stats.Classification.trainsplit.Aten, round(sum(fold.train$Classification == "Aten Asteroid")/length(fold.train$Classification)*100, 4))
  folds.stats.Classification.validsplit.Aten <- append(folds.stats.Classification.validsplit.Aten, round(sum(fold.valid$Classification == "Aten Asteroid")/length(fold.valid$Classification)*100, 4))
  
  decisionTree.Classification.GINI <- rpart(Classification ~ Orbit.Axis..AU. + Orbit.Eccentricity + Orbit.Inclination..deg. + Perihelion.Argument..deg. + Node.Longitude..deg. + Mean.Anomoly..deg. + Perihelion.Distance..AU. + Aphelion.Distance..AU. + Orbital.Period..yr. + Minimum.Orbit.Intersection.Distance..AU. + Asteroid.Magnitude,#
    data=fold.train, method="class",  cp= 0.001) #. all var
  
  decisionTree.Classification.IGHE <- rpart(Classification ~ Orbit.Axis..AU. + Orbit.Eccentricity + Orbit.Inclination..deg. + Perihelion.Argument..deg. + Node.Longitude..deg. + Mean.Anomoly..deg. + Perihelion.Distance..AU. + Aphelion.Distance..AU. + Orbital.Period..yr. + Minimum.Orbit.Intersection.Distance..AU. + Asteroid.Magnitude,#
    data=fold.train, method="class", parms = list(split="information"), cp=0.001) #. all var
  
  #salvo anche la comple 
  #altezza
  
  
  decisionTree.Classification.GINI.prediction <- predict(decisionTree.Classification.GINI, fold.valid, type = "class")
  decisionTree.Classification.IGHE.prediction <- predict(decisionTree.Classification.IGHE, fold.valid, type = "class")
  
  decisionTree.Classification.GINI.confusion_matrix_multiclass = confusionMatrix(
    decisionTree.Classification.GINI.prediction, fold.valid$Classification, mode = "prec_recall") 
  decisionTree.Classification.IGHE.confusion_matrix_multiclass = confusionMatrix(
    decisionTree.Classification.IGHE.prediction, fold.valid$Classification, mode = "prec_recall") 
  
  confusion.multi.GINI = decisionTree.Classification.GINI.confusion_matrix_multiclass$byClass
  confusion.multi.IGHE = decisionTree.Classification.IGHE.confusion_matrix_multiclass$byClass
  
  #GINI perf
  sens_Amor.GINI = confusion.multi.GINI["Class: Amor Asteroid","Sensitivity"] 
  spec_Amor.GINI = confusion.multi.GINI["Class: Amor Asteroid","Specificity"]
  prec_Amor.GINI = confusion.multi.GINI["Class: Amor Asteroid","Precision"]
  recal_Amor.GINI = confusion.multi.GINI["Class: Amor Asteroid","Recall"]
  f1_Amor.GINI = confusion.multi.GINI["Class: Amor Asteroid","F1"]
  
  sens_Apohele.GINI = confusion.multi.GINI["Class: Apohele Asteroid","Sensitivity"] 
  spec_Apohele.GINI = confusion.multi.GINI["Class: Apohele Asteroid","Specificity"]
  prec_Apohele.GINI = confusion.multi.GINI["Class: Apohele Asteroid","Precision"]
  recal_Apohele.GINI = confusion.multi.GINI["Class: Apohele Asteroid","Recall"]
  f1_Apohele.GINI = confusion.multi.GINI["Class: Apohele Asteroid","F1"]
  
  sens_Apollo.GINI = confusion.multi.GINI["Class: Apollo Asteroid","Sensitivity"] 
  spec_Apollo.GINI = confusion.multi.GINI["Class: Apollo Asteroid","Specificity"]
  prec_Apollo.GINI = confusion.multi.GINI["Class: Apollo Asteroid","Precision"]
  recal_Apollo.GINI = confusion.multi.GINI["Class: Apollo Asteroid","Recall"]
  f1_Apollo.GINI = confusion.multi.GINI["Class: Apollo Asteroid","F1"]
  
  sens_Aten.GINI = confusion.multi.GINI["Class: Aten Asteroid","Sensitivity"] 
  spec_Aten.GINI = confusion.multi.GINI["Class: Aten Asteroid","Specificity"]
  prec_Aten.GINI = confusion.multi.GINI["Class: Aten Asteroid","Precision"]
  recal_Aten.GINI = confusion.multi.GINI["Class: Aten Asteroid","Recall"]
  f1_Aten.GINI = confusion.multi.GINI["Class: Aten Asteroid","F1"]
  
  #IGHE perf
  sens_Amor.IGHE = confusion.multi.IGHE["Class: Amor Asteroid","Sensitivity"] 
  spec_Amor.IGHE = confusion.multi.IGHE["Class: Amor Asteroid","Specificity"]
  prec_Amor.IGHE = confusion.multi.IGHE["Class: Amor Asteroid","Precision"]
  recal_Amor.IGHE = confusion.multi.IGHE["Class: Amor Asteroid","Recall"]
  f1_Amor.IGHE = confusion.multi.IGHE["Class: Amor Asteroid","F1"]
  
  sens_Apohele.IGHE = confusion.multi.IGHE["Class: Apohele Asteroid","Sensitivity"] 
  spec_Apohele.IGHE = confusion.multi.IGHE["Class: Apohele Asteroid","Specificity"]
  prec_Apohele.IGHE = confusion.multi.IGHE["Class: Apohele Asteroid","Precision"]
  recal_Apohele.IGHE = confusion.multi.IGHE["Class: Apohele Asteroid","Recall"]
  f1_Apohele.IGHE = confusion.multi.IGHE["Class: Apohele Asteroid","F1"]
  
  sens_Apollo.IGHE = confusion.multi.IGHE["Class: Apollo Asteroid","Sensitivity"] 
  spec_Apollo.IGHE = confusion.multi.IGHE["Class: Apollo Asteroid","Specificity"]
  prec_Apollo.IGHE = confusion.multi.IGHE["Class: Apollo Asteroid","Precision"]
  recal_Apollo.IGHE = confusion.multi.IGHE["Class: Apollo Asteroid","Recall"]
  f1_Apollo.IGHE = confusion.multi.IGHE["Class: Apollo Asteroid","F1"]
  
  sens_Aten.IGHE = confusion.multi.IGHE["Class: Aten Asteroid","Sensitivity"] 
  spec_Aten.IGHE = confusion.multi.IGHE["Class: Aten Asteroid","Specificity"]
  prec_Aten.IGHE = confusion.multi.IGHE["Class: Aten Asteroid","Precision"]
  recal_Aten.IGHE = confusion.multi.IGHE["Class: Aten Asteroid","Recall"]
  f1_Aten.IGHE = confusion.multi.IGHE["Class: Aten Asteroid","F1"]
  
  #GINI MACROs
  MacroSensitivity.GINI = (0.25 * sens_Amor.GINI) + (0.25 * sens_Apohele.GINI) + (0.25 * sens_Apollo.GINI) + (0.25 * sens_Aten.GINI)
  MacroSpecificity.GINI = (0.25 * spec_Amor.GINI) + (0.25 * spec_Apohele.GINI) + (0.25 * spec_Apollo.GINI) + (0.25 * spec_Aten.GINI)
  MacroPrecision.GINI = (0.25 * prec_Amor.GINI) + (0.25 * prec_Apohele.GINI) + (0.25 * prec_Apollo.GINI) + (0.25 * prec_Aten.GINI)
  MacroRecall.GINI = (0.25 * recal_Amor.GINI) + (0.25 * recal_Apohele.GINI) + (0.25 * recal_Apollo.GINI) + (0.25 * recal_Aten.GINI)
  MacroF1.GINI = (0.25 * f1_Amor.GINI) + (0.25 * f1_Apohele.GINI) + (0.25 * f1_Apollo.GINI) + (0.25 * prec_Aten.GINI)
  
  #IGHE MACROs
  MacroSensitivity.IGHE = (0.25 * sens_Amor.IGHE) + (0.25 * sens_Apohele.IGHE) + (0.25 * sens_Apollo.IGHE) + (0.25 * sens_Aten.IGHE)
  MacroSpecificity.IGHE = (0.25 * spec_Amor.IGHE) + (0.25 * spec_Apohele.IGHE) + (0.25 * spec_Apollo.IGHE) + (0.25 * spec_Aten.IGHE)
  MacroPrecision.IGHE = (0.25 * prec_Amor.IGHE) + (0.25 * prec_Apohele.IGHE) + (0.25 * prec_Apollo.IGHE) + (0.25 * prec_Aten.IGHE)
  MacroRecall.IGHE = (0.25 * recal_Amor.IGHE) + (0.25 * recal_Apohele.IGHE) + (0.25 * recal_Apollo.IGHE) + (0.25 * recal_Aten.IGHE)
  MacroF1.IGHE = (0.25 * f1_Amor.IGHE) + (0.25 * f1_Apohele.IGHE) + (0.25 * f1_Apollo.IGHE) + (0.25 * prec_Aten.IGHE)
  
  dt.Classification.GINI.foldstats$Accuracy    = append(dt.Classification.GINI.foldstats$Accuracy, decisionTree.Classification.GINI.confusion_matrix_multiclass$overall["Accuracy"])
  dt.Classification.GINI.foldstats$MacroSensitivity = append(dt.Classification.GINI.foldstats$MacroSensitivity, MacroSensitivity.GINI)
  dt.Classification.GINI.foldstats$MacroSpecificity = append(dt.Classification.GINI.foldstats$MacroSpecificity, MacroSpecificity.GINI)
  dt.Classification.GINI.foldstats$MacroPrecision = append(dt.Classification.GINI.foldstats$MacroPrecision, MacroPrecision.GINI)
  dt.Classification.GINI.foldstats$MacroRecall = append(dt.Classification.GINI.foldstats$MacroRecall, MacroRecall.GINI)
  dt.Classification.GINI.foldstats$MacroF1 = append(dt.Classification.GINI.foldstats$MacroF1, MacroF1.GINI)
  
  dt.Classification.IGHE.foldstats$Accuracy    = append(dt.Classification.IGHE.foldstats$Accuracy, decisionTree.Classification.IGHE.confusion_matrix_multiclass$overall["Accuracy"])
  dt.Classification.IGHE.foldstats$MacroSensitivity = append(dt.Classification.IGHE.foldstats$MacroSensitivity, MacroSensitivity.IGHE)
  dt.Classification.IGHE.foldstats$MacroSpecificity = append(dt.Classification.IGHE.foldstats$MacroSpecificity, MacroSpecificity.IGHE)
  dt.Classification.IGHE.foldstats$MacroPrecision = append(dt.Classification.IGHE.foldstats$MacroPrecision, MacroPrecision.IGHE)
  dt.Classification.IGHE.foldstats$MacroRecall = append(dt.Classification.IGHE.foldstats$MacroRecall, MacroRecall.IGHE)
  dt.Classification.IGHE.foldstats$MacroF1 = append(dt.Classification.IGHE.foldstats$MacroF1, MacroF1.IGHE)
  
  
  decisionTree.Classification.GINI.confusion_matrix <- table(fold.valid$Classification, decisionTree.Classification.GINI.prediction)
  decisionTree.Classification.IGHE.confusion_matrix <- table(fold.valid$Classification, decisionTree.Classification.IGHE.prediction)
  
  if (is.null(decisionTree.Classification.GINI.stats)){
    decisionTree.Classification.GINI.stats <- decisionTree.Classification.GINI.confusion_matrix
  }else{
    decisionTree.Classification.GINI.stats <- decisionTree.Classification.GINI.stats + decisionTree.Classification.GINI.confusion_matrix
  }
  
  if (is.null(decisionTree.Classification.IGHE.stats)){
    decisionTree.Classification.IGHE.stats <- decisionTree.Classification.IGHE.confusion_matrix
  }else{
    decisionTree.Classification.IGHE.stats <- decisionTree.Classification.IGHE.stats + decisionTree.Classification.IGHE.confusion_matrix
  }
  rm(name_fold, fold.train, fold.valid)
}

dt.GINI.name <- paste("Classification GINI")
dt.Classification$Accuracy[dt.GINI.name] <- dt.Classification.GINI.foldstats$Accuracy
dt.Classification$MacroSensitivity[dt.GINI.name] <- dt.Classification.GINI.foldstats$MacroSensitivity
dt.Classification$MacroSpecificity[dt.GINI.name] <- dt.Classification.GINI.foldstats$MacroSpecificity
dt.Classification$MacroPrecision[dt.GINI.name] <- dt.Classification.GINI.foldstats$MacroPrecision
dt.Classification$MacroRecall[dt.GINI.name] <- dt.Classification.GINI.foldstats$MacroRecall
dt.Classification$MacroF1[dt.GINI.name] <- dt.Classification.GINI.foldstats$MacroF1
tdist <- list()
tdist_val = confidence_interval(as.vector(dt.Classification.GINI.foldstats$Accuracy),0.95)
  tdist$acc <- paste(as.character(round(tdist_val[2],4))," ± ",as.character(round(tdist_val[1],4)))
tdist_val = confidence_interval(as.vector(dt.Classification.GINI.foldstats$MacroSensitivity),0.95)
  tdist$sens <- paste(as.character(round(tdist_val[2],4))," ± ",as.character(round(tdist_val[1],4)))
tdist_val = confidence_interval(as.vector(dt.Classification.GINI.foldstats$MacroSpecificity),0.95)
  tdist$spec <- paste(as.character(round(tdist_val[2],4))," ± ",as.character(round(tdist_val[1],4)))
tdist_val = confidence_interval(as.vector(dt.Classification.GINI.foldstats$MacroPrecision),0.95)
  tdist$prec <- paste(as.character(round(tdist_val[2],4))," ± ",as.character(round(tdist_val[1],4)))
tdist_val = confidence_interval(as.vector(dt.Classification.GINI.foldstats$MacroRecall),0.95)
  tdist$rec <- paste(as.character(round(tdist_val[2],4))," ± ",as.character(round(tdist_val[1],4)))
tdist_val = confidence_interval(as.vector(dt.Classification.GINI.foldstats$MacroF1),0.95)
  tdist$f1 <- paste(as.character(round(tdist_val[2],4))," ± ",as.character(round(tdist_val[1],4)))
dt.Classification.All[dt.GINI.name] <- c(tdist$acc,tdist$sens,tdist$spec,tdist$prec,tdist$rec,tdist$f1)

dt.IGHE.name <- paste("Classification IGHE")
dt.Classification$Accuracy[dt.IGHE.name] <- dt.Classification.IGHE.foldstats$Accuracy
dt.Classification$MacroSensitivity[dt.IGHE.name] <- dt.Classification.IGHE.foldstats$MacroSensitivity
dt.Classification$MacroSpecificity[dt.IGHE.name] <- dt.Classification.IGHE.foldstats$MacroSpecificity
dt.Classification$MacroPrecision[dt.IGHE.name] <- dt.Classification.IGHE.foldstats$MacroPrecision
dt.Classification$MacroRecall[dt.IGHE.name] <- dt.Classification.IGHE.foldstats$MacroRecall
dt.Classification$MacroF1[dt.IGHE.name] <- dt.Classification.IGHE.foldstats$MacroF1
tdist <- list()
tdist_val = confidence_interval(as.vector(dt.Classification.IGHE.foldstats$Accuracy),0.95)
  tdist$acc <- paste(as.character(round(tdist_val[2],4))," ± ",as.character(round(tdist_val[1],4)))
tdist_val = confidence_interval(as.vector(dt.Classification.IGHE.foldstats$MacroSensitivity),0.95)
  tdist$sens <- paste(as.character(round(tdist_val[2],4))," ± ",as.character(round(tdist_val[1],4)))
tdist_val = confidence_interval(as.vector(dt.Classification.IGHE.foldstats$MacroSpecificity),0.95)
  tdist$spec <- paste(as.character(round(tdist_val[2],4))," ± ",as.character(round(tdist_val[1],4)))
tdist_val = confidence_interval(as.vector(dt.Classification.IGHE.foldstats$MacroPrecision),0.95)
  tdist$prec <- paste(as.character(round(tdist_val[2],4))," ± ",as.character(round(tdist_val[1],4)))
tdist_val = confidence_interval(as.vector(dt.Classification.IGHE.foldstats$MacroRecall),0.95)
  tdist$rec <- paste(as.character(round(tdist_val[2],4))," ± ",as.character(round(tdist_val[1],4)))
tdist_val = confidence_interval(as.vector(dt.Classification.IGHE.foldstats$MacroF1),0.95)
  tdist$f1 <- paste(as.character(round(tdist_val[2],4))," ± ",as.character(round(tdist_val[1],4)))
dt.Classification.All[dt.IGHE.name] <- c(tdist$acc,tdist$sens,tdist$spec,tdist$prec,tdist$rec,tdist$f1)
end_table <- length(dt.Classification$Accuracy)
img_name_plot <- paste("IMG_asteroids_model_DecisionTree_", "Classification_KFOLD_performance_plot_A", ".png", sep = "")
png(img_name_plot,res = 800, height = 10, width = 15, unit='in')
par(mfrow=c(2,2)) 
  boxplot(dt.Classification$Accuracy[2:end_table],vertical = TRUE, pch=19, las = 2, col = 1:length(dt.Classification$Accuracy),main="Accuracy")  
  boxplot(dt.Classification$MacroSensitivity[2:end_table],vertical = TRUE, pch=19, las = 2, col = 1:length(dt.Classification$MacroSensitivity),main="MacroSensitivity")  
  dev.off()
img_name_plot <- paste("IMG_asteroids_model_DecisionTree_", "Classification_KFOLD_performance_plot_B", ".png", sep = "")
png(img_name_plot,res = 800, height = 10, width = 15, unit='in')
par(mfrow=c(2,2))  
  boxplot(dt.Classification$MacroSpecificity[2:end_table], vertical = TRUE, pch=19, las = 2, col = 1:length(dt.Classification$Specificity),main="MacroSpecificity")  
  boxplot(dt.Classification$MacroPrecision[2:end_table], vertical = TRUE, pch=19, las = 2, col = 1:length(dt.Classification$MacroPrecision),main="MacroPrecision")  
  dev.off()
img_name_plot <- paste("IMG_asteroids_model_DecisionTree_", "Classification_KFOLD_performance_plot_C", ".png", sep = "")
png(img_name_plot,res = 800, height = 10, width = 15, unit='in')
par(mfrow=c(2,2))  
  boxplot(dt.Classification$MacroRecall[2:end_table],vertical = TRUE, pch=19, las = 2, col = 1:length(dt.Classification$MacroRecall),main="MacroRecall")  
  boxplot(dt.Classification$MacroF1[2:end_table],vertical = TRUE, pch=19, las = 2, col = 1:length(dt.Classification$MacroF1),main="MacroF1")  
  dev.off()

img_name_plot <- paste("PDF_asteroids_model_DecisionTree_", "Classification_KFOLD_performance_tdist", ".pdf", sep = "")
pdf(img_name_plot, height = 20, width = 46)
  grid.table(t(dt.Classification.All[2:length(dt.Classification.All)]))
  dev.off()

rm(i)

decisionTree.Classification.GINI.stats = decisionTree.Classification.GINI.stats / length(folds)
decisionTree.Classification.IGHE.stats = decisionTree.Classification.IGHE.stats / length(folds)

img_name_plot <- paste("IMG_asteroids_model_decisiontree_tree_", "GINI_Classification_confusion" ,".png", sep = "")
png(img_name_plot)
  grid.table(decisionTree.Classification.GINI.stats)
  dev.off()

img_name_plot <- paste("IMG_asteroids_model_decisiontree_tree_", "IGHE_Classification_confusion" ,".png", sep = "")
png(img_name_plot)
  grid.table(decisionTree.Classification.IGHE.stats)
  dev.off()
 
img_name_plot <- paste("IMG_asteroids_model_decisiontree_tree", "Classification_fold_split", ".png", sep = "")
png(img_name_plot)
  par(mfrow=c(2,4)) 
  barplot(as.numeric(folds.stats.Classification.trainsplit.Amor),main="Amor Train",names.arg = folds.stats.namesplit, col='#a71e3b')
  barplot(as.numeric(folds.stats.Classification.validsplit.Amor),main="Amor Valid",names.arg = folds.stats.namesplit, col='#a71e3b')
  barplot(as.numeric(folds.stats.Classification.trainsplit.Apohele),main="Apohele Train",names.arg = folds.stats.namesplit, col='#a71e3b')
  barplot(as.numeric(folds.stats.Classification.validsplit.Apohele),main="Apohele Valid",names.arg = folds.stats.namesplit, col='#a71e3b')
  barplot(as.numeric(folds.stats.Classification.trainsplit.Apollo),main="Apollo Train",names.arg = folds.stats.namesplit, col='#a71e3b')
  barplot(as.numeric(folds.stats.Classification.validsplit.Apollo),main="Apollo Valid",names.arg = folds.stats.namesplit, col='#a71e3b')
  barplot(as.numeric(folds.stats.Classification.trainsplit.Aten),main="Aten Train",names.arg = folds.stats.namesplit, col='#a71e3b')
  barplot(as.numeric(folds.stats.Classification.validsplit.Aten),main="Aten Valid",names.arg = folds.stats.namesplit, col='#a71e3b')
  dev.off()

img_name_plot <- paste("IMG_asteroids_model_decisiontree_tree_", "Classification_CP_TABLE_GINI" ,".png", sep = "")
png(img_name_plot)
  p<-tableGrob(printcp(decisionTree.Classification.GINI))
  grid.arrange(p)
  dev.off()

img_name_plot <- paste("IMG_asteroids_model_decisiontree_tree_", "Classification_CP_PLOT_GINI" ,".png", sep = "")
png(img_name_plot)
  plotcp(decisionTree.Classification.GINI)
  dev.off()

#prunedDecisionTree = prune(decisionTree, cp= .011)

img_name_plot <- paste("IMG_asteroids_model_decisiontree_tree_","Classification_FANCY_PLOT_GINI", ".png", sep = "")
png(img_name_plot)
  fancyRpartPlot(decisionTree.Classification.GINI)
  dev.off()

  img_name_plot <- paste("IMG_asteroids_model_decisiontree_tree_", "Classification_CP_TABLE_IGHE" ,".png", sep = "")
  png(img_name_plot)
  p<-tableGrob(printcp(decisionTree.Classification.IGHE))
  grid.arrange(p)
  dev.off()
  
  img_name_plot <- paste("IMG_asteroids_model_decisiontree_tree_", "Classification_CP_PLOT_IGHE" ,".png", sep = "")
  png(img_name_plot)
  plotcp(decisionTree.Classification.IGHE)
  dev.off()
  
  #prunedDecisionTree = prune(decisionTree, cp= .011)
  
img_name_plot <- paste("IMG_asteroids_model_decisiontree_tree_","Classification_FANCY_PLOT_IGHE", ".png", sep = "")
png(img_name_plot)
  fancyRpartPlot(decisionTree.Classification.IGHE)
  dev.off()  

rm(folds.stats.namesplit, folds.stats.Classification.trainsplit.Amor, folds.stats.Classification.validsplit.Amor, folds.stats.Classification.trainsplit.Apohele, folds.stats.Classification.validsplit.Apohele, folds.stats.Classification.trainsplit.Apollo, folds.stats.Classification.validsplit.Apollo, folds.stats.Classification.trainsplit.Aten, folds.stats.Classification.validsplit.Aten, )
rm(decisionTree.Classification.GINI)
rm(decisionTree.Classification.GINI.stats, decisionTree.Classification.GINI.prediction, decisionTree.Classification.GINI.confusion_matrix)

rm(asteroids_split, folds, img_name_plot, p)
