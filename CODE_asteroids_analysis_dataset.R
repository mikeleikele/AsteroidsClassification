setwd("C:/Users/Gasatoz/Github/AsteroidsClassification")
#load dataset RObject as asteroids_data
load("DATA_asteroids_dataset.RData")

####### Preliminary Analysis ############

dim(asteroids_dataset)


#### get types of each variable

sapply(asteroids_data, class)


### count frequency of each class
levels(asteroids_data$Hazardous)

class_frequencies = table(asteroids_data$Hazardous)


asteroid_type = table(asteroids_data$Classification)

asteroid_type

## pie chart 
pie(class_frequencies)
pie(asteroid_type)

## bar plot
barplot(table(asteroids_data$Hazardous))

barplot(asteroid_type)
# get various statistics of each covariate
### mean, sd, var, min, max, median, range, quantile
numeric_covariates = asteroids_data[,c("Orbit.Axis..AU." , "Orbit.Eccentricity" , "Orbit.Inclination..deg." , "Perihelion.Argument..deg.",
                               "Node.Longitude..deg." , "Mean.Anomoly..deg." , "Perihelion.Distance..AU." , "Aphelion.Distance..AU.",
                               "Orbital.Period..yr.", "Minimum.Orbit.Intersection.Distance..AU.",  "Asteroid.Magnitude"
                                     )]
factor_covariates = asteroids_data[,c("Orbital.Reference", "Classification", "Epoch..TDB." )]
rows = c("mean", "sd" , "var", "min", "max", "median") # "range", "quantile" )

columns = c("Orbit.Axis..AU." , "Orbit.Eccentricity" , "Orbit.Inclination..deg." , "Perihelion.Argument..deg.",
         "Node.Longitude..deg." , "Mean.Anomoly..deg." , "Perihelion.Distance..AU." , "Aphelion.Distance..AU.",
         "Orbital.Period..yr.", "Minimum.Orbit.Intersection.Distance..AU.",  "Asteroid.Magnitude"
)
stats <- data.frame(matrix(nrow = length(rows), ncol = length(columns))) 

# assign column names
colnames(stats) = columns
rownames(stats) = rows

i <- 1

for( x in numeric_covariates ){
  stats['mean', columns[i] ] <- mean((x))
  stats[ 'sd' , columns[i] ] <- sd((x))
  stats[ 'var' , columns[i] ] <- var((x))
  stats[ 'min' , columns[i] ] <- min((x))
  stats['max' , columns[i] ] <- max((x))
  stats[ 'median' , columns[i]] <- median((x))
  #stats[ i, 'range'] <- range((x))
  #stats[columns[i]]['quantile'] = quantile((x))
  i <- i + 1 
  }


stats
## mean
stats[columns[1]]['mean']

#### make boxplot of each covariate

x = numeric_covariates 

par(mfrow = c (2,4))

for (i in 1:8) {
  
  boxplot(x[,i], main=names(x)[i]) 
  
}

par(mfrow = c (1,3))

for (i in 9:11) {
  
  boxplot(x[,i], main=names(x)[i]) 
  
}

### box plot factor variables

x = factor_covariates 


par(mfrow = c (1,3))

for (i in 1:3) {
  
  boxplot(x[,i], main=names(x)[i]) 
  
}



### histogram numeric covariates

x = numeric_covariates 

par(mfrow = c (2,4))

for (i in 1:8) {
  
  hist(x[,i], main=names(x)[i]) 
  
}

par(mfrow = c (1,3))

for (i in 9:11) {
  
  hist(x[,i], main=names(x)[i]) 
  
}



library(caret)

#### multivariate plots
par(mfrow = c (1,1))

plot(x=asteroids_data$Node.Longitude..deg. , y=asteroids_data$Minimum.Orbit.Intersection.Distance..AU., col=asteroids_data$Hazardous)
plot(x=asteroids_data$Mean.Anomoly..deg., y=asteroids_data$Eccentricity, col=asteroids_data$Hazardous)

### ggplot

library(ggplot2) 
library(ggExtra)


# Save the scatter plot in a variable
p <- ggplot(asteroids_data, aes(x = Eccentricity , y = Mean.Motion)) +
  geom_point()
p
# Plot the scatter plot with marginal histograms
ggMarginal(p, type = "histogram")

ggplot(asteroids_data, aes(Perihelion.Argument..deg.)) + geom_density(aes(fill=factor(Hazardous), alpha=0.75))
ggplot(asteroids_data, aes(x=Orbital.Period..yr., y=Hazardous )) + geom_point(alpha=0.7)


### best variable up - Jupiter Tisserand - mean Motion  Miss dist lunar
#pca 1
#  apelion dist , orbital period , semi major axis
# pca 2
# minimum orbit intersection perihelion arg , absolute magnituted uncertainity