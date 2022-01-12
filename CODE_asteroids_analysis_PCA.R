setwd("~/Github/AsteroidsClassification")

library("FactoMineR")
library("factoextra")
library("gridExtra")

#load dataset RObject as asteroids_data
load("DATA_asteroids_dataset.RData")

#PCA feature extration

col_active = c("Orbit.Axis..AU.", "Orbit.Eccentricity", "Orbit.Inclination..deg.", "Perihelion.Argument..deg.", "Node.Longitude..deg.", "Mean.Anomoly..deg.", "Perihelion.Distance..AU.", "Aphelion.Distance..AU.", "Orbital.Period..yr.", "Minimum.Orbit.Intersection.Distance..AU.", "Asteroid.Magnitude")
asteroids_data.active <- asteroids_data[,col_active]

asteroids_pca = PCA(asteroids_data.active, graph = TRUE)
asteroids_pca.eig_val <- get_eigenvalue(asteroids_pca)

img_name_plot <- paste("IMG_asteroids_pca_", "Component_eig_values" ,".png", sep = "")
png(img_name_plot)
  p<-tableGrob(asteroids_pca.eig_val)
  grid.arrange(p)
  dev.off()


img_name_plot <- paste("IMG_asteroids_pca_", "Component_explaind_variances" ,".png", sep = "")
png(img_name_plot)
  fviz_eig(asteroids_pca, addlabels = TRUE)
  dev.off()
  

#old variable comparison
var <- get_pca_var(asteroids_pca)
img_name_plot <- paste("IMG_asteroids_pca_", "covariables_in_pcaSpace" ,".png", sep = "")
png(img_name_plot)
  fviz_pca_var(asteroids_pca, col.var = "blue")
  dev.off()

var$coord
var$cos2
var$contrib


#individual analysis
ind <- get_pca_ind(asteroids_pca)
fviz_pca_ind(asteroids_pca)

img_name_plot <- paste("IMG_asteroids_pca_", "individual_class_cos2" ,".png", sep = "")
png(img_name_plot)
  fviz_pca_ind(asteroids_pca, col.ind = "cos2",
    gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
    repel = FALSE # Avoid text overlapping (slow if many points)
  )
  dev.off()

img_name_plot <- paste("IMG_asteroids_pca_", "individual_class_hazardous" ,".png", sep = "")
png(img_name_plot)
  fviz_pca_biplot(asteroids_pca,
      geom.ind = "point",
      col.ind = asteroids_data$Hazardous,
      palette = c("#00AFBB", "#E7B800", "#FC4E07"),
      addEllipses = TRUE,
      legend.title = "Groups"
  )
  dev.off()

img_name_plot <- paste("IMG_asteroids_pca_", "individual_class_classification" ,".png", sep = "")
png(img_name_plot)
fviz_pca_biplot(asteroids_pca,
                geom.ind = "point",
                col.ind = asteroids_data$Classification,
                palette = c("#00AFBB", "#E7B800", "#FC4E07"),
                addEllipses = TRUE,
                legend.title = "Groups"
)
dev.off()
  
var$coord
var$cos2
var$contrib
