ccdata = read.table("D://MS Georgia Tech/Introduction to Analytics/HW1/credit_card_data-headers.txt", header = T, sep = '\t')
setwd("D:/MS Georgia Tech/Introduction to Analytics/HW1")
install.packages("kknn")

library(data.table)
library(kernlab)
set.seed(42)

model <-  ksvm(as.matrix(ccdata[,1:10]),as.factor(ccdata[,11]),type="C-svc",kernel="vanilladot",C=100,scaled=TRUE)
# calculate a1.am
a <- colSums(model@xmatrix[[1]] * model@coef[[1]])
#a
# calculate a0
a0 <- -model@b
#a0
# see what the model predicts
pred <- predict(model,ccdata[,1:10])
#pred
# see what fraction of the model's predictions match the actual classification
sum(pred == ccdata[,11]) / nrow(ccdata)

myC = seq(1,10, by=1)
results=c()
for(i in 1:length(myC)){
  # call ksvm using  kernel instead of linear
  model <-  ksvm(as.matrix(ccdata[,1:10]),as.factor(ccdata[,11]),type="C-svc",kernel="vanilladot",C= i,scaled=TRUE)
  # calculate a1.am
  a <- colSums(model@xmatrix[[1]] * model@coef[[1]])
  # calculate a0
  a0 <- -model@b
  # see what the model predicts
  pred <- predict(model,ccdata[,1:10])
  # see what fraction of the model's predictions match the actual classification
  results[i]=data.table(sum(pred == ccdata[,11]) / nrow(ccdata))
  #results[i]=sum(pred == ccdata[,11]) / nrow(ccdata)
}
results



myKernels = c("rbfdot","polydot","tanhdot","laplacedot","besseldot","anovadot","splinedot")
results=list()
for(i in 1:length(myKernels)){
  # call ksvm using  kernel instead of linear
  model <-  ksvm(as.matrix(ccdata[,1:10]),as.factor(ccdata[,11]),type="C-svc",kernel=myKernels[[i]],C=10000,scaled=TRUE)
  # calculate a1.am
  a <- colSums(model@xmatrix[[1]] * model@coef[[1]])
  a
  # calculate a0
  a0 <- -model@b
  a0
  # see what the model predicts
  pred <- predict(model,ccdata[,1:10])
  pred
  # see what fraction of the model's predictions match the actual classification
  results[[i]]=data.table(kernel=myKernels[[i]],accuracy=sum(pred == ccdata[,11]) / nrow(ccdata))
}
results

library(kknn)

# initiating total_accurate_values to 0
total_accurate_values = 0
# running for loop for all the datapoints
for (i in 1:654){
  CCmodel = kknn(R1~A1+A2+A3+A8+A9+A10+A11+A12+A14+A15,
                 ccdata[-i,],
                 ccdata[i,],
                 k = 10,
                 distance = 2,
                 kernel = 'optimal',
                 scale = TRUE)
  rounded_fitted_value = round(fitted.values(CCmodel))
  actual_value = ccdata[i,11]
  # checking if fitted value  matches with actual values
  if (actual_value == rounded_fitted_value) {
    total_accurate_values = total_accurate_values+1
  }
}
# see what fraction of the model's predictions match the actual classification
accuracy = total_accurate_values/654







#Initiating a vector
results=c()
# Looping the value of k from 1 to 20 neighbours
for (j in 1:20){
  total_accurate_values = 0
  #looping through all datapoints
  for (i in 1:654){
    CCmodel = kknn(R1~A1+A2+A3+A8+A9+A10+A11+A12+A14+A15,
                   ccdata[-i,],
                   ccdata[i,],
                   k = j,
                   distance = 2,
                   kernel = 'optimal',
                   scale = TRUE)
    rounded_fitted_value = round(fitted.values(CCmodel))
    actual_value = ccdata[i,11]
    if (actual_value == rounded_fitted_value) {
      total_accurate_values = total_accurate_values+1
    }
    # see what fraction of the model's predictions match the actual classification
    accuracy = total_accurate_values/654
  }
  results[j] = accuracy
}


