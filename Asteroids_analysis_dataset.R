setwd("C:/Users/Gasatoz/Github/AsteroidsClassification")
#load dataset RObject as asteroids_data
load("asteroids_dataset.RData")

####### Preliminary Analysis ############

dim(asteroids_data)


#### get types of each variable

sapply(asteroids_data, class)


### count frequency of each class
levels(asteroids_data$Hazardous)

class_frequencies = table(asteroids_data$Hazardous)

## pie chart 
pie(class_frequencies)


# get various statistics of each covariate
### mean, sd, var, min, max, median, range, quantile
covariates = asteroids_data[,c("Est.Dia.in.KM.min." , "Est.Dia.in.M.min." , "Est.Dia.in.Miles.min." , "Est.Dia.in.Feet.min.",
              "Est.Dia.in.KM.max." , "Est.Dia.in.M.max." , "Est.Dia.in.Miles.max." , "Est.Dia.in.Feet.max.",
              "Relative.Velocity.km.per.sec", "Relative.Velocity.km.per.hr",  "Miles.per.hour",
              "Miss.Dist..Astronomical.", "Miss.Dist..lunar." , "Miss.Dist..miles." , 
              "Minimum.Orbit.Intersection" , "Jupiter.Tisserand.Invariant" , "Epoch.Osculation" ,
              "Eccentricity" , "Semi.Major.Axis" ,"Inclination",
              "Asc.Node.Longitude", "Orbital.Period" , "Perihelion.Distance",
              "Perihelion.Arg", "Mean.Anomaly", "Mean.Motion"        
              )]

for x in covariates ){
  mean((x))
  sd((x))
  var((x))
  min((x))
  max((x))
  median((x))
  range((x))
  quantile((x))
  }

## mean

#### make boxplot of each covariate

hist(asteroids_data$Absolute.Magnitude)

x = asteroids_data[, c("Est.Dia.in.KM.min." , "Est.Dia.in.M.min." , "Est.Dia.in.Miles.min." , "Est.Dia.in.Feet.min.",
                       "Est.Dia.in.KM.max." , "Est.Dia.in.M.max." , "Est.Dia.in.Miles.max." , "Est.Dia.in.Feet.max.",
                       "Relative.Velocity.km.per.sec", "Relative.Velocity.km.per.hr",  "Miles.per.hour",
                       "Miss.Dist..Astronomical.", "Miss.Dist..lunar." , "Miss.Dist..miles." , 
                       "Minimum.Orbit.Intersection" , "Jupiter.Tisserand.Invariant" , 
                       "Epoch.Osculation", "Eccentricity" , "Semi.Major.Axis" ,
                       "Inclination", "Asc.Node.Longitude", "Orbital.Period" , 
                       "Perihelion.Distance", "Perihelion.Arg", "Mean.Anomaly" )]

par(mfrow = c (2,4))

for (i in 1:8) {
  
  boxplot(x[,i], main=names(x)[i]) 
  
  }

x = asteroids_data[, c("Relative.Velocity.km.per.sec", "Relative.Velocity.km.per.hr",  "Miles.per.hour",
                       "Miss.Dist..Astronomical.", "Miss.Dist..lunar." , "Miss.Dist..miles." , 
                       "Minimum.Orbit.Intersection" , "Jupiter.Tisserand.Invariant" , "Epoch.Osculation"
                       )
                   ]

par(mfrow = c (3,3))

for (i in 1:9) {
  
  boxplot(x[,i], main=names(x)[i]) 
  
  }

x = asteroids_data[, c("Mean.Motion", "Eccentricity" , "Semi.Major.Axis" ,
                       "Inclination", "Asc.Node.Longitude", "Orbital.Period" , 
                       "Perihelion.Distance", "Perihelion.Arg", "Mean.Anomaly" 
                        )
                   ]

par(mfrow = c (3,3))

for (i in 1:9) {
  
  boxplot(x[,i], main=names(x)[i]) 
  
}

# histograms of each covariate
hist(asteroids_data$Absolute.Magnitude)

x = asteroids_data[, c("Est.Dia.in.KM.min." , "Est.Dia.in.M.min." , "Est.Dia.in.Miles.min." , "Est.Dia.in.Feet.min.",
                       "Est.Dia.in.KM.max." , "Est.Dia.in.M.max." , "Est.Dia.in.Miles.max." , "Est.Dia.in.Feet.max.")]

par(mfrow = c (2,4))

for (i in 1:8) {
  
  hist(x[,i], main=names(x)[i]) 
  
  }

x = asteroids_data[, c("Relative.Velocity.km.per.sec", "Relative.Velocity.km.per.hr",  "Miles.per.hour",
                       "Miss.Dist..Astronomical.", "Miss.Dist..lunar." , "Miss.Dist..miles." , 
                       "Minimum.Orbit.Intersection" , "Jupiter.Tisserand.Invariant" , "Epoch.Osculation"
                   )]

par(mfrow = c (3,3))

for (i in 1:9) {
  
  hist(x[,i], main=names(x)[i]) 

  }

x = asteroids_data[, c("Eccentricity" , "Semi.Major.Axis" ,"Inclination",
                       "Asc.Node.Longitude", "Orbital.Period" , "Perihelion.Distance",
                       "Perihelion.Arg", "Mean.Anomaly", "Mean.Motion" 
                       
                       )
                   ]


for (i in 1:9) {
  
  hist(x[,i], main=names(x)[i]) 
  
  }



library(caret)

#### multivariate plots
par(mfrow = c (1,1))

plot(x=asteroids_data$Mean.Anomaly, y=asteroids_data$Mean.Motion, col=asteroids_data$Hazardous)
plot(x=asteroids_data$Perielion.Distance, y=asteroids_data$Eccentricity, col=asteroids_data$Hazardous)


### ggplot

library(ggplot2)
library(ggExtra)


# Save the scatter plot in a variable
p <- ggplot(asteroids_data, aes(x = Eccentricity , y = Mean.Motion)) +
  geom_point()
p
# Plot the scatter plot with marginal histograms
ggMarginal(p, type = "histogram")

ggplot(asteroids_data, aes(Orbital.Period)) + geom_density(aes(fill=factor(Hazardous), alpha=0.75))
ggplot(asteroids_data, aes(x=Inclination, y=Hazardous, size = pop )) + geom_point(alpha=0.7)
