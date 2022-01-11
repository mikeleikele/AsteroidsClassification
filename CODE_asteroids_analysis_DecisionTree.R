setwd("~/Github/AsteroidsClassification")

library(rpart)
library(rattle)
library(rpart.plot)
library(RColorBrewer)

#load dataset RObject as asteroids_split
load("DATA_asteroids_dataset_split_0.7.RData")


decisionTree = rpart(Hazardous ~ Orbit.Axis..AU. + Orbit.Eccentricity + Orbit.Inclination..deg. + Perihelion.Argument..deg. + Node.Longitude..deg. + Mean.Anomoly..deg. + Perihelion.Distance..AU. + Aphelion.Distance..AU. + Orbital.Period..yr. + Minimum.Orbit.Intersection.Distance..AU. + Asteroid.Magnitude,
                     data=asteroids_split$train, method="class") #. all var
printcp(decisionTree)

plotcp(decisionTree)




decimg_name_plot <- paste("IMG_asteroids_model_decisiontree_tree", ".png", sep = "")
png(img_name_plot)
  fancyRpartPlot(decisionTree)

asteroids_split$test$pred <- predict(decisionTree, asteroids_split$test, type = "class")
confusion.matrix = table(asteroids_split$test$Hazardous, asteroids_split$test$pred)
sum(diag(confusion.matrix))/sum(confusion.matrix)

prunedDecisionTree = prune(decisionTree, cp= .011)
fancyRpartPlot(prunedDecisionTree)

decisionTreeIG = rpart(Hazardous ~ Absolute.Magnitude+Est.Dia.in.KM.min. +Est.Dia.in.KM.max. +Est.Dia.in.M.min.+Est.Dia.in.M.max.+Est.Dia.in.Miles.min. +Est.Dia.in.Miles.max.+Est.Dia.in.Feet.min. +Relative.Velocity.km.per.sec+Epoch.Date.Close.Approach+Relative.Velocity.km.per.hr+Miles.per.hour+Miss.Dist..Astronomical.+Miss.Dist..lunar.+Miss.Dist..kilometers.+Miss.Dist..miles.+Orbit.Uncertainity+Minimum.Orbit.Intersection+Jupiter.Tisserand.Invariant +Epoch.Osculation+Eccentricity+Semi.Major.Axis+Inclination +Asc.Node.Longitude+Orbital.Period +Perihelion.Distance+Perihelion.Arg,
  data=asteroids_split$train, method="class", #. all var
  parms = list(split = 'information'))
