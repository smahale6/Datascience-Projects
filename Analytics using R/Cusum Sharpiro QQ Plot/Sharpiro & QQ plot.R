
set.seed(42)
uscrime = read.table("D://MS Georgia Tech/Introduction to Analytics/HW3/uscrime.txt", header = TRUE, sep = '\t')
shapiro.test(uscrime$Crime)



lapply(uscrime,shapiro.test)
qqnorm(uscrime$Crime, col = "blue")
qqline(uscrime$Crime, col = "red")

install.packages("shapiro.test")
library(outliers)
histogram(uscrime$Crime)
boxplot(uscrime$Crime)

grubbs.test(uscrime$Crime, two.sided = TRUE)

grubbs.test(uscrime$Crime, type = 10 )

grubbs.test(tail(uscrime$Crime), type = 20, opposite = FALSE, two.sided = FALSE )

head()

