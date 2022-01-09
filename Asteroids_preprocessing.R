setwd("~/Github/AsteroidsClassification")
#load data
asteroids_data = read.csv("dataset/nasa.csv", header = TRUE)

#datatype change chars -> fact or int -> boolean
cat_cols <- c("Neo.Reference.ID", "Name", "Orbiting.Body", "Orbit.ID" , "Equinox")
log_cols <- c("Hazardous")


asteroids_data[cat_cols] <- lapply(asteroids_data[cat_cols], as.factor)
asteroids_data[log_cols] <- lapply(asteroids_data[log_cols], as.logical)

asteroids_data$Close.Approach.Date <- as.Date(asteroids_data$Close.Approach.Date, format =  "%Y-%m-%d")
asteroids_data$Orbit.Determination.Date <- as.Date(asteroids_data$Orbit.Determination.Date, format =  "%Y-%m-%d %H:%M:%S")


asteroids_datatype <- sapply(asteroids_data, class)

#remove object useless
rm(cat_cols,log_cols,asteroids_datatype)
save(asteroids_data, file="asteroids_dataset.RData")
rm(asteroids_data)
