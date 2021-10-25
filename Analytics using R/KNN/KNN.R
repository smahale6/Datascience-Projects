#Clear the environment
rm(list=ls())
#install.packages("kableExtra")
library(kableExtra)
library(DAAG)
library(kknn)
BreastCancerWisconsin <- read.table("breast-cancer-wisconsin.data.txt", stringsAsFactors = FALSE, sep = ",")
head(BreastCancerWisconsin)


MissingDataValues <- which(BreastCancerWisconsin$V7 == "?")
kable(BreastCancerWisconsin[MissingDataValues,], caption="Wisconsin Breast Cancer Data: Rows with Missing Data Values")
head(MissingDataValues)

## (Total Number Missing Values / Dataset Size) * 100
cat(100*length(MissingDataValues)/nrow(BreastCancerWisconsin),"% of data missing")


## Clean Dataset with no missing values
BreastCancerWisconsin_Clean <- BreastCancerWisconsin[-MissingDataValues,]
head(BreastCancerWisconsin_Clean)

## Dataset with only missing values
BreastCancerWisconsin_Missing <- BreastCancerWisconsin[MissingDataValues,]
BreastCancerWisconsin_Missing


# function to find the mode of a variable
CalculateMode <- function(field) {
  UniqueValues <- unique(field)
  UniqueValues[which.max(tabulate(match(field,UniqueValues)))]
}

# calculating the mode of v7 in the clean data
ModeOfV7 <- as.numeric(CalculateMode(BreastCancerWisconsin_Clean$V7))
cat("Mode of V7: ",ModeOfV7)
#creating a copy of the original dataset
BreastCancerWisconsin_ModeImpute <- BreastCancerWisconsin
# impute missing values for V7 with the mode
BreastCancerWisconsin_ModeImpute[MissingDataValues,]$V7 <- as.integer(ModeOfV7)
kable(BreastCancerWisconsin_ModeImpute[MissingDataValues,], caption = "Mode Imputation of Breast Cancer Winsconsin Missing Data")



# calculating the mode of v7 in the clean data
MeanOfV7 <- mean(as.numeric(BreastCancerWisconsin_Clean$V7))
cat("mean: ",MeanOfV7)
#creating a copy of the original dataset
BreastCancerWisconsin_MeanImpute <- BreastCancerWisconsin
# impute the missing values with the mean
BreastCancerWisconsin_MeanImpute[MissingDataValues,]$V7 <- as.integer(MeanOfV7)
kable(BreastCancerWisconsin_MeanImpute[MissingDataValues,], caption = "Mean Imputation of Breast Cancer Winsconsin Missing Data")



# data set without missing values or response variable
BreastCancerWisconsin_Clean_NoResponse <- BreastCancerWisconsin_Clean[,2:10]
BreastCancerWisconsin_Clean_NoResponse$V7 <- as.integer(BreastCancerWisconsin_Clean_NoResponse$V7)
head(BreastCancerWisconsin_Clean_NoResponse)


BreastCancerWisconsin_Model <- lm(V7~., data = BreastCancerWisconsin_Clean_NoResponse)
summary(BreastCancerWisconsin_Model)

# build a linear regression model with only significant predictors
BreastCancerWisconsin_Model2 <- lm(V7~V2+V4+V5+V8, data = BreastCancerWisconsin_Clean_NoResponse)
summary(BreastCancerWisconsin_Model2)

step(BreastCancerWisconsin_Model)

# Performing 4 fold crossvalidation
BreastCancerWisconsin_Model_CV <- cv.lm(BreastCancerWisconsin_Clean_NoResponse, BreastCancerWisconsin_Model2, m=4)


n = length(BreastCancerWisconsin_Clean$V7)
avg = mean(as.numeric(BreastCancerWisconsin_Clean$V7))

SSE<-0
SSR<-0
SST<-0


for(i in 1:n){
  SST = SST + (as.numeric(BreastCancerWisconsin_Clean$V7[i]) - avg)^2
  SSE = SSE + (as.numeric(BreastCancerWisconsin_Clean$V7[i]) - as.numeric(BreastCancerWisconsin_Model_CV$cvpred[i]))^2
  SSR = SSR + (as.numeric(BreastCancerWisconsin_Model_CV$cvpred) - avg)^2
}
SSE
SST
SSR

R_Squared = 1- (SSE/SST)
R_Squared






# predicted values
V7_predicted <- predict(BreastCancerWisconsin_Model2, newdata = BreastCancerWisconsin_Missing)
# impute the missing values with predicted values
BreastCancerWisconsin_Reg_Impute <- BreastCancerWisconsin
BreastCancerWisconsin_Reg_Impute[MissingDataValues,]$V7 <- as.integer(V7_predicted)
# make sure the data is within the original range of 1 to 10
BreastCancerWisconsin_Reg_Impute$V7[BreastCancerWisconsin_Reg_Impute$V7 > 10] <- 10
BreastCancerWisconsin_Reg_Impute$V7[BreastCancerWisconsin_Reg_Impute$V7 < 1] <- 1
# resulting imputed values
kable(BreastCancerWisconsin_Reg_Impute[MissingDataValues,], caption = "Regression Imputation of Breast Cancer Winsconsin Missing Data")







# set seed for reproducible random results
set.seed(1)
# perturbation using normal distribution of the predicted values
perturbation_values <- rnorm(nrow(BreastCancerWisconsin_Missing), V7_predicted, sd(V7_predicted))
cat("perturbation values: ", perturbation_values)

# imputing the missing values
BreastCancerWisconsin_Petrubation <- BreastCancerWisconsin
BreastCancerWisconsin_Petrubation[MissingDataValues,]$V7 <- perturbation_values
BreastCancerWisconsin_Petrubation[MissingDataValues,]$V7 <- as.integer(BreastCancerWisconsin_Petrubation[MissingDataValues,]$V7)
# Checking if the data is within the range
BreastCancerWisconsin_Petrubation$V7[BreastCancerWisconsin_Petrubation$V7 > 10] <- 10
BreastCancerWisconsin_Petrubation$V7[BreastCancerWisconsin_Petrubation$V7 < 1] <- 1
kable(BreastCancerWisconsin_Petrubation[MissingDataValues,], caption = "Peturbation Regression Imputation of Breast Cancer Winsconsin Missing Data")





# set seed
set.seed(1)
# training and validation sets
training <- sample(nrow(BreastCancerWisconsin), size = floor(nrow(BreastCancerWisconsin) * 0.7))
testing <- setdiff(1:nrow(BreastCancerWisconsin),training)

accuracies_knn <- rep(0,15)



Total_knn_Accuracies = c()
results_knn_mode <- c()
for (k in 1:15) {
  BreastCancerWisconsin_ModeImpute_model <- kknn(V11~V2+V3+V4+V5+V6+V7+V8+V9+V10, BreastCancerWisconsin_ModeImpute[training,], BreastCancerWisconsin_ModeImpute[testing,], k=k)
  pred <- as.integer(fitted(BreastCancerWisconsin_ModeImpute_model)+0.5)
  results_knn_mode[k] <- sum(pred == BreastCancerWisconsin_ModeImpute[testing,]$V11) / nrow(BreastCancerWisconsin_ModeImpute[testing,])
  Total_knn_Accuracies[k] <- sum(pred == BreastCancerWisconsin_ModeImpute[testing,]$V11) / nrow(BreastCancerWisconsin_ModeImpute[testing,])
}
results_knn_mode
Total_knn_Accuracies



results_knn_mean <- c()
for (k in 1:15) {
  BreastCancerWisconsin_MeanImpute_model <- kknn(V11~V2+V3+V4+V5+V6+V7+V8+V9+V10, BreastCancerWisconsin_MeanImpute[training,], BreastCancerWisconsin_MeanImpute[testing,], k=k)
  pred <- as.integer(fitted(BreastCancerWisconsin_MeanImpute_model)+0.5)
  results_knn_mean[k] <- sum(pred == BreastCancerWisconsin_MeanImpute[testing,]$V11) / nrow(BreastCancerWisconsin_MeanImpute[testing,])
  Total_knn_Accuracies[k+15] <- sum(pred == BreastCancerWisconsin_MeanImpute[testing,]$V11) / nrow(BreastCancerWisconsin_MeanImpute[testing,])
}
results_knn_mean
Total_knn_Accuracies


results_knn_regression <- c()
for (k in 1:15) {
  BreastCancerWisconsin_RegImpute_model <- kknn(V11~V2+V3+V4+V5+V6+V7+V8+V9+V10, BreastCancerWisconsin_Reg_Impute[training,], BreastCancerWisconsin_Reg_Impute[testing,], k=k)
  pred <- as.integer(fitted(BreastCancerWisconsin_RegImpute_model)+0.5)
  results_knn_regression[k] <- sum(pred == BreastCancerWisconsin_Reg_Impute[testing,]$V11) / nrow(BreastCancerWisconsin_Reg_Impute[testing,])
  Total_knn_Accuracies[k+30] <- sum(pred == BreastCancerWisconsin_Reg_Impute[testing,]$V11) / nrow(BreastCancerWisconsin_Reg_Impute[testing,])
}
results_knn_regression
Total_knn_Accuracies


results_knn_peturbation <- c()
for (k in 1:15) {
  BreastCancerWisconsin_Petrubation_model <- kknn(V11~V2+V3+V4+V5+V6+V7+V8+V9+V10, BreastCancerWisconsin_Petrubation[training,], BreastCancerWisconsin_Petrubation[testing,], k=k)
  pred <- as.integer(fitted(BreastCancerWisconsin_Petrubation_model)+0.5)
  results_knn_peturbation[k] <- sum(pred == BreastCancerWisconsin_Petrubation[testing,]$V11) / nrow(BreastCancerWisconsin_Petrubation[testing,])
  Total_knn_Accuracies[k+45] <- sum(pred == BreastCancerWisconsin_Petrubation[testing,]$V11) / nrow(BreastCancerWisconsin_Petrubation[testing,])
}
results_knn_peturbation
Total_knn_Accuracies


plot(Total_knn_Accuracies)
