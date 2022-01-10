setwd("~/Github/AsteroidsClassification")
install.packages(c("FactoMineR", "factoextra"))
library("FactoMineR")
library("factoextra")

#load dataset RObject as asteroids_data
load("DATA_asteroids_dataset.RData")

#PCA feature extration

col_active = c("Absolute.Magnitude","Est.Dia.in.KM.min.","Est.Dia.in.KM.max.","Est.Dia.in.M.min.","Est.Dia.in.M.max.","Est.Dia.in.Miles.min.","Est.Dia.in.Miles.max.","Est.Dia.in.Feet.min.","Relative.Velocity.km.per.sec","Epoch.Date.Close.Approach","Relative.Velocity.km.per.hr","Miles.per.hour","Miss.Dist..Astronomical.","Miss.Dist..lunar.","Miss.Dist..kilometers.","Miss.Dist..miles.","Orbit.Uncertainity","Minimum.Orbit.Intersection","Jupiter.Tisserand.Invariant","Epoch.Osculation","Eccentricity","Semi.Major.Axis","Inclination","Asc.Node.Longitude","Orbital.Period","Perihelion.Distance","Perihelion.Arg","Aphelion.Dist","Perihelion.Time","Mean.Anomaly","Mean.Motion")
asteroids_data.active <- asteroids_data[,col_active]

asteroids_pca = PCA(asteroids_data.active, graph = TRUE)

asteroids_pca.eig_val <- get_eigenvalue(asteroids_pca)

fviz_eig(asteroids_pca, addlabels = TRUE, ylim = c(0, 50))
fviz_screeplot(asteroids_pca)

#var
var <- get_pca_var(asteroids_pca)
fviz_pca_var(asteroids_pca, col.var = "black")
var$coord
var$cos2
var$contrib


#istanze
ind <- get_pca_ind(asteroids_pca)
fviz_pca_ind(asteroids_pca)

fviz_pca_ind(asteroids_pca, col.ind = "cos2",
  gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
    repel = TRUE # Avoid text overlapping (slow if many points)
)


fviz_pca_biplot(asteroids_pca,
                geom.ind = "point",
                col.ind = asteroids_data$Hazardous,
                palette = c("#00AFBB", "#E7B800", "#FC4E07"),
                addEllipses = TRUE,
                legend.title = "Groups"
)

var$coord
var$cos2
var$contrib
