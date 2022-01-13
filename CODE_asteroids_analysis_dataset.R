setwd("~/Github/AsteroidsClassification")
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

all_covariates <- cbind(numeric_covariates, factor_covariates)

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


## calculate 4-quantiles for each covariate

#calculate quartiles

sapply(numeric_covariates, function(x) quantile(x, probs = seq(0, 1, 1/4)))


#calculate range for each covariate

sapply(numeric_covariates, function(x) range(x))


#### make boxplot of each covariate


plot_name <- paste("IMG_asteroids_barplot_", "numeric_covariates" ,".png", sep = "")


x = numeric_covariates 


png(plot_name)
par(mfrow = c (1,4))

for (i in 1:4) {
  
  boxplot(x[,i], main=names(x)[i], col='red') 
  
}

dev.off()



plot_name <- paste("IMG_asteroids_barplot_", "numeric_covariates2" ,".png", sep = "")


x = numeric_covariates 
png(plot_name)

par(mfrow = c (1,4))

for (i in 1:8) {
  
  boxplot(x[,i], main=names(x)[i], col = 'red') 
  
}


dev.off()



plot_name <- paste("IMG_asteroids_barplot_", "numeric_covariates3" ,".png", sep = "")

png(plot_name)

par(mfrow = c (1,3))


for (i in 9:11) {
  
  boxplot(x[,i], main=names(x)[i],  col = 'red') 
  
}

dev.off()


### box plot factor variables

x = factor_covariates 

plot_name <- paste("IMG_asteroids_barplot_", "factor_covariates" ,".png", sep = "")

png(plot_name)
par(mfrow = c (1,3))



for (i in 1:3) {
  
  boxplot(x[,i], main=names(x)[i],  col = 'red') 
  
}

dev.off()


### histogram numeric covariates

x = numeric_covariates 

plot_name <- paste("IMG_asteroids_histogram_", "numeric_covariates" ,".png", sep = "")

png(plot_name)

par(mfrow = c (2,2))





for (i in 1:4) {
  
  hist(x[,i], main=names(x)[i] ) 
  
}

dev.off()

plot_name <- paste("IMG_asteroids_histogram_", "numeric_covariates2" ,".png", sep = "")

png(plot_name)


par(mfrow = c (2,2))

for (i in 5:8) {
  
  hist(x[,i], main=names(x)[i] ) 
  
}

dev.off()

plot_name <- paste("IMG_asteroids_histogram_", "numeric_covariates3" ,".png", sep = "")

png(plot_name)

par(mfrow = c (1,3))

for (i in 9:11) {
  
  hist(x[,i], main=names(x)[i]  ) 
  
}

dev.off()


#### correlation matrix

# change fill and outline color manually 
ggplot(asteroids_data, aes(x =Minimum.Orbit.Intersection.Distance..AU. )) +
  geom_histogram(aes(color = Hazardous, fill = Hazardous), 
                 position = "identity", bins = 30, alpha = 0.4) +
  scale_color_manual(values = c("#00AFBB", "#E7B800")) +
  scale_fill_manual(values = c("#00AFBB", "#E7B800"))

# change fill and outline color manually 
ggplot(asteroids_data, aes(x =Orbit.Axis..AU. )) +
  geom_histogram(aes(color = Hazardous, fill = Hazardous), 
                 position = "identity", bins = 30, alpha = 0.4) +
  scale_color_manual(values = c("#00AFBB", "#E7B800")) +
  scale_fill_manual(values = c("#00AFBB", "#E7B800"))


library(caret)

#### multivariate plots
par(mfrow = c (1,1))
x = numeric_covariates[, 1]
y = asteroids_data$Classification
featurePlot(x,y)
featurePlot(x,y, plot = 'density', scales=list(x=list(relation='free'), y= list(relation='free')), auto.key=list(columns=3))

par(mfrow = c (1,1))
ggplot(asteroids_data, aes(x = columns[1], 
                           y = columns[2] )) +
                          geom_point()


plot(x=asteroids_data$Node.Longitude..deg. , y=asteroids_data$Minimum.Orbit.Intersection.Distance..AU., col=asteroids_data$Hazardous)
plot(x=asteroids_data$Mean.Anomoly..deg., y=asteroids_data$Eccentricity, col=asteroids_data$Hazardous)

### ggplot

library(ggplot2) 
library(ggExtra)


# Save the scatter plot in a variable
ggplot(asteroids_data, aes(x = Orbital.Period..yr. , y = Mean.Anomoly..deg., color=Hazardous) ) +
  geom_point()

ggplot(asteroids_data, aes(x = Orbital.Period..yr. , y = Orbit.Eccentricity, color=Classification) ) +
  geom_point()



plot_name <- paste("IMG_asteroids_multivariate_scatter", "period_eccentricity" ,".png", sep = "")

png(plot_name)


ggplot(asteroids_data, aes(y = Orbital.Period..yr. , x = Orbit.Eccentricity, color=Classification) ) +
  geom_point()

dev.off()

library(ggplot2)




ggplot(asteroids_data, aes(x = Orbital.Period..yr. , y = Orbit.Eccentricity, color=Hazardous) ) +
  geom_point()

ggplot(asteroids_data, aes(x =Asteroid.Magnitude , y = Orbit.Inclination..deg., color=Hazardous) ) +
  geom_point()

ggplot(asteroids_data, aes(x = Minimum.Orbit.Intersection.Distance..AU. , y = Orbit.Eccentricity, color=Hazardous) ) +
  geom_point()

plot_name <- paste("IMG_asteroids_multivariate_scatter_", "magintude_orbitintersection" ,".png", sep = "")

png(plot_name)


ggplot(asteroids_data, aes(x = Minimum.Orbit.Intersection.Distance..AU. , y = Asteroid.Magnitude, color=Hazardous) ) +
  geom_point()



dev.off()


ggplot(asteroids_data , aes(x = Asteroid.Magnitude , y = Mean.Anomoly..deg.,  color=Hazardous))



# Plot the scatter plot with marginal histograms
ggMarginal(p, type = "histogram")

ggplot(asteroids_data, aes(Perihelion.Argument..deg.)) + geom_density(aes(fill=factor(Hazardous), alpha=0.75))
ggplot(asteroids_data, aes(x=Orbital.Period..yr., y=Classification )) + geom_point(alpha=0.7)
ggplot(asteroids_data, aes(x=Orbital.Period..yr., y=Classification )) + geom_point(alpha=0.7)



c("Orbit.Axis..AU." , "Orbit.Eccentricity" , "Orbit.Inclination..deg." , "Perihelion.Argument..deg.",
  "Node.Longitude..deg." , "Mean.Anomoly..deg." , "Perihelion.Distance..AU." , "Aphelion.Distance..AU.",
  "Orbital.Period..yr.", "Minimum.Orbit.Intersection.Distance..AU.",  "Asteroid.Magnitude")
## correlation matrix


install.packages("corrplot")
library(corrplot)

numeric_covariates.cor = cor(numeric_covariates, method = c("spearman"))
all_covariates.cor = cor(all_covariates, method = c("spearman"))

corrplot(numeric_covariates.cor)


plot_name <- paste("IMG_asteroids_correlation_", "numeric_covariates" ,".png", sep = "")

png(plot_name)

par(mfrow = c (1,1))

corrplot(numeric_covariates.cor)

dev.off()


palette = colorRampPalette(c("green", "white", "red")) (20)
heatmap(x = mydata.cor, col = palette, symm = TRUE)


install.packages('ggcorrplot')
library('ggcorrplot')


ggcorrplot(cor(numeric_covariates))

plot_name <- paste("IMG_asteroids_correlation_", "numeric_covariates_ggplot_spearman" ,".png", sep = "")

png(plot_name)

ggcorrplot(cor(numeric_covariates , method = c('spearman')))

dev.off()

# p-values and rcorr

library("Hmisc")

numeric_covariates.rcorr = rcorr(as.matrix(numeric_covariates))
numeric_covariates.rcorr


numeric_covariates.coeff = numeric_covariates.rcorr$r
numeric_covariates.p = numeric_covariates.rcorr$P



### best variable up - Jupiter Tisserand - mean Motion  Miss dist lunar
#pca 1
#  apelion dist , orbital period , semi major axis
# pca 2
# minimum orbit intersection perihelion arg , absolute magnituted uncertainity