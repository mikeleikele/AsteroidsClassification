setwd("~/Github/AsteroidsClassification")

library("FactoMineR")
library("factoextra")
library(gridExtra)
library(caret)
library(tidyverse)
#load dataset RObject as asteroids_data
load("DATA_asteroids_dataset.RData")

#PCA feature extration

col_active = c("Orbit.Axis..AU.", "Orbit.Eccentricity", "Orbit.Inclination..deg.", "Perihelion.Argument..deg.", "Node.Longitude..deg.", "Mean.Anomoly..deg.", "Perihelion.Distance..AU.", "Aphelion.Distance..AU.", "Orbital.Period..yr.", "Minimum.Orbit.Intersection.Distance..AU.", "Asteroid.Magnitude")
asteroids_data.active <- asteroids_data[,col_active]

asteroids_pca = PCA(asteroids_data.active, graph = TRUE)

asteroids_pca.eig_val <- get_eigenvalue(asteroids_pca)

img_name_plot <- paste("IMG_asteroids_PCA_", "Components_variance" ,".png", sep = "")
png(img_name_plot,res = 800, height = 10, width = 15, unit='in')
grid.table(asteroids_pca.eig_val)
dev.off()


fviz_eig(asteroids_pca, addlabels = TRUE, ylim = c(0, 50))

img_name_plot <- paste("IMG_asteroids_PCA_", "CoVariable_screenplot",".png", sep = "")
png(img_name_plot,res = 800, height = 10, width = 15, unit='in')
fviz_eig(asteroids_pca, addlabels = TRUE, ylim = c(0, 50))
dev.off()

#var
var <- get_pca_var(asteroids_pca)

img_name_plot <- paste("IMG_asteroids_PCA_", "CoVariable_component_clock" ,".png", sep = "")
png(img_name_plot,res = 800, height = 10, width = 15, unit='in')
  fviz_pca_var(asteroids_pca, col.var = "black")
  dev.off()

img_name_plot <- paste("IMG_asteroids_PCA_", "CoVariable_component_5" ,".png", sep = "")
png(img_name_plot,res = 800, height = 10, width = 15, unit='in')
  grid.table(var$coord)
  dev.off()


var$cos2
var$contrib




#istanze
ind <- get_pca_ind(asteroids_pca)
fviz_pca_ind(asteroids_pca)

img_name_plot <- paste("IMG_asteroids_PCA_", "Istance_","cos2" ,".png", sep = "")
  png(img_name_plot,res = 800, height = 10, width = 15, unit='in')
  fviz_pca_ind(asteroids_pca, col.ind = "cos2",
               gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
               repel = TRUE # Avoid text overlapping (slow if many points)
  )
  dev.off()



img_name_plot <- paste("IMG_asteroids_PCA_", "Istance_","Hazardous" ,".png", sep = "")
png(img_name_plot,res = 800, height = 10, width = 15, unit='in')
  fviz_pca_biplot(asteroids_pca,
                geom.ind = "point",
                  col.ind = asteroids_data$Hazardous,
                  palette = c("#00AFBB", "#E7B800", "#FC4E07"),
                  addEllipses = TRUE,
                  legend.title = "Groups"
  )
  dev.off()

img_name_plot <- paste("IMG_asteroids_PCA_", "Istance_","Classification" ,".png", sep = "")
png(img_name_plot,res = 800, height = 10, width = 15, unit='in')
fviz_pca_biplot(asteroids_pca,
                  geom.ind = "point",
                  col.ind = asteroids_data$Classification,
                  palette = c("#00AFBB", "#E7B800", "#FC4E07", "#02b436"),
                  addEllipses = TRUE,
                  legend.title = "Groups"
  )
  dev.off()
  
var$coord
var$cos2
var$contrib
