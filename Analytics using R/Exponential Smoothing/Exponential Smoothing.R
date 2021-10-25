
library(data.table)
AtlantaTemp =  read.table("D://MS Georgia Tech/Introduction to Analytics/HW4/temps.txt", header = TRUE,sep = '\t')
head(AtlantaTemp)


AtlantaTempVector = as.vector( unlist( AtlantaTemp[ , 2:21 ] ) )
AtlantaTempVectorTS = ts( AtlantaTempVector, start = 1996, frequency = 123 )
plot.ts( AtlantaTempVectorTS )


EsModel= HoltWinters( AtlantaTempVectorTS, alpha = NULL, beta = NULL, gamma = NULL, seasonal =  "multiplicative")
#Plotting moder
EsModel
plot(EsModel)


EsOutput = EsModel$fitted
plot( EsOutput )

EsOutput_Seasonal = matrix(EsOutput[,4],nrow = 123)

write.csv( EsOutput_Seasonal, file = "EsOutput.csv" )
