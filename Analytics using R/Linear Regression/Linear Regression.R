set.seed(33)
uscrime = read.table("D://MS Georgia Tech/Introduction to Analytics/HW5/uscrime.txt", header = TRUE, sep = '\t')
uscrime

##Running Linear Regression Model on all the data
lm_uscrime1 <- lm(Crime~.,data = uscrime)
## Summary of the Model
summary(lm_uscrime1)
##Setting up test points
test_point1 <- data.frame(M = 14.0, So = 0, Ed = 10.0,Po1 = 12.0,Po2 = 15.5,
                         LF = 0.640, M.F = 94.0, Pop = 150, NW = 1.1,
                         U1 = 0.120, U2 = 3.6 , Wealth  = 3200, Ineq = 20.1,
                         Prob = 0.040, Time = 39.0)
##Predicting the Model
Pred_model1 <- predict(lm_uscrime1,test_point1)
Pred_model1
qqnorm(uscrime$Crime)
qqline(uscrime$Crime)

################################
##Running Linear Regression Model on aSignificant values of lm_uscrime1
lm_uscrime2 <- lm(Crime~M+Ed+Ineq+Prob+Po1+U2,data = uscrime)
## Summary of the Model
summary(lm_uscrime2)
##Setting up test points
test_point2 <- data.frame(M = 14.0,  Ed = 10.0, Ineq = 20.1,Po1 = 12.0,U2 = 3.6 ,
                          Prob = 0.040)
##Predicting the Model
Pred_model2 <- predict(lm_uscrime2,test_point2)
Pred_model2
mean(uscrime$Crime)
median(uscrime$Crime)


################################



##Running Linear Regression Model on aSignificant values of lm_uscrime1
lm_uscrime3 <- lm(Crime~Ed+Prob,data = uscrime)
## Summary of the Model
summary(lm_uscrime3)
##Setting up test points
test_point3 <- data.frame( Ed = 10.0,Prob = 0.040)
##Predicting the Model
Pred_model3 <- predict(lm_uscrime3,test_point3)
Pred_model3
mean(uscrime$Crime)
median(uscrime$Crime)

Pred_model <- predict(lm_uscrime,test_point)

Pred_model



##install.packages("DAAG")
library(DAAG)
lm_uscrime_Cv = cv.lm(uscrime,lm_uscrime2,m = 4)

## Length of Crime Set
n = length(uscrime$Crime)
## Mean of the Crime
avg = mean(uscrime$Crime)

SSE<-0
SSR<-0
SST<-0

for(i in 1:n){
  SST = SST + (uscrime$Crime[i] - avg)^2
  SSE = SSE + (uscrime$Crime[i] - lm_uscrime_Cv$cvpred[i])^2
  SSR = SSR + (lm_uscrime_Cv$cvpred[i] - avg)^2
}
SSE
SST
SSR

R_Squared = 1- (SSE/SST)
R_Squared






