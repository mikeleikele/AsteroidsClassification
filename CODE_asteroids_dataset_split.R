setwd("~/Github/AsteroidsClassification")

#load dataset RObject as asteroids_data
load("DATA_asteroids_dataset.RData")

split.data = function(data, p = 0.7, s = 1){
  set.seed(s) 
  index = sample(1:dim(data)[1]) 
  train = data[index[1:floor(dim(data)[1] * p)], ] 
  test = data[index[((ceiling(dim(data)[1] * p)) + 1):dim(data)[1]], ] 
  return(list(train=train, test=test)) 
} 

split_perc <- 0.7
asteroids_split <- split.data(asteroids_data, p = split_perc)

img_name_plot <- paste("IMG_asteroids_dataset_splited_hazardous_", as.character(split_perc), ".png", sep = "")
png(img_name_plot)
  par(mfrow=c(1,2)) 
  pie(table(asteroids_split$train$Hazardous), 
    labels = paste(round(prop.table(table(asteroids_split$train$Hazardous))*100), "% (",table(asteroids_split$train$Hazardous),")" , sep = ""), 
    col = heat.colors(5), main = "Train set - Hazardous class"
  )
  pie(table(asteroids_split$test$Hazardous), 
      labels = paste(round(prop.table(table(asteroids_split$test$Hazardous))*100), "% (",table(asteroids_split$test$Hazardous),")" , sep = ""), 
      col = heat.colors(5), main = "Test set - Hazardous class"
  )
dev.off() 

img_name_plot <- paste("IMG_asteroids_dataset_splited_classification_", as.character(split_perc), ".png", sep = "")
png(img_name_plot)
par(mfrow=c(1,2)) 
pie(table(asteroids_split$train$Classification), 
    labels = paste(round(prop.table(table(asteroids_split$train$Classification))*100), "% (",table(asteroids_split$train$Classification),")" , sep = ""), 
    col = heat.colors(5), main = "Train set - Classification of asteroids "
)
pie(table(asteroids_split$test$Classification), 
    labels = paste(round(prop.table(table(asteroids_split$test$Classification))*100), "% (",table(asteroids_split$test$Classification),")" , sep = ""), 
    col = heat.colors(5), main = "Test set - Classification of asteroids"
)
dev.off() 


  
name_file <- paste("DATA_asteroids_dataset_split_", as.character(split_perc), ".RData", sep = "")

rm(asteroids_data,split.data,split_perc,img_name_plot)
save(asteroids_split, file=name_file)
rm(asteroids_split,name_file)
