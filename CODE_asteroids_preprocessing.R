setwd("~/Github/AsteroidsClassification")
library("scales")

#load data
asteroids_data = read.csv("DATASET/orbits.csv", header = TRUE, row.name=1)

#datatype change chars -> fact or int -> boolean
cat_cols <- c("Epoch..TDB.","Classification","Orbital.Reference")
log_cols <- c("Hazardous")
double_cols <- c("Asteroid.Magnitude")

asteroids_data[cat_cols] <- lapply(asteroids_data[cat_cols], as.factor)
asteroids_data[log_cols] <- lapply(asteroids_data[log_cols], as.logical)
asteroids_data[double_cols] <- lapply(asteroids_data[double_cols], as.numeric )

asteroids_data$Hazardous.int = as.factor(asteroids_data$Hazardous)


RescaleFUN <- function(x) *(x-min(x))/(max(x) - min(x))-0.5) * 2
int2rescale <- c("Orbit.Axis..AU.","Orbit.Eccentricity","Orbit.Inclination..deg.","Perihelion.Argument..deg.","Node.Longitude..deg.","Mean.Anomoly..deg.","Perihelion.Distance..AU.","Aphelion.Distance..AU.","Orbital.Period..yr.","Minimum.Orbit.Intersection.Distance..AU.","Asteroid.Magnitude")
nameNewCols <- c("Orbit.Axis..AU._scaled_scaled","Orbit.Eccentricity_scaled","Orbit.Inclination..deg._scaled","Perihelion.Argument..deg._scaled","Node.Longitude..deg._scaled","Mean.Anomoly..deg._scaled","Perihelion.Distance..AU._scaled","Aphelion.Distance..AU._scaled","Orbital.Period..yr._scaled","Minimum.Orbit.Intersection.Distance..AU._scaled","Asteroid.Magnitude_scaled")
for (i in 1:length(int2rescale)) {
  to_scaled = int2rescale[i]
  to_scaled.name = nameNewCols[i]
  asteroids_data[to_scaled.name] = RescaleFUN(asteroids_data[to_scaled])
}

asteroids_datatype <- sapply(asteroids_data, class)



#remove object useless
rm(cat_cols,log_cols,double_cols,asteroids_datatype)
save(asteroids_data, file="DATA_asteroids_dataset.RData")
rm(asteroids_data)

