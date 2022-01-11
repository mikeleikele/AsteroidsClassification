setwd("~/Github/AsteroidsClassification")
#load data
asteroids_data = read.csv("DATASET/orbits.csv", header = TRUE, row.name=1)

#datatype change chars -> fact or int -> boolean
cat_cols <- c("Epoch..TDB.","Classification","Orbital.Reference")
log_cols <- c("Hazardous")
double_cols <- c("Asteroid.Magnitude")

asteroids_data[cat_cols] <- lapply(asteroids_data[cat_cols], as.factor)
asteroids_data[log_cols] <- lapply(asteroids_data[log_cols], as.logical)
asteroids_data[double_cols] <- lapply(asteroids_data[double_cols], as.numeric )

asteroids_datatype <- sapply(asteroids_data, class)

#remove object useless
rm(cat_cols,log_cols,double_cols,asteroids_datatype)
save(asteroids_data, file="DATA_asteroids_dataset.RData")
rm(asteroids_data)
