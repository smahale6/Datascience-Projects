set.seed(1)
uscrime <- read.table("D://MS Georgia Tech/Introduction to Analytics/HW7/uscrime.txt", header = TRUE)
library(randomForest)
install.packages("tree")
library(tree)
library(caret)


TreeModelUSCrime <- tree(Crime ~ ., data = uscrime)
summary(TreeModelUSCrime)


# see how tree was split
TreeModelUSCrime$frame
# Ploting the tree
plot(TreeModelUSCrime)
text(TreeModelUSCrime)
title("USCRIME Classification Tree for Training Set")

# Prune the tree
termnodes <- 5
TreeModelUSCrime <- prune.tree(TreeModelUSCrime, best = termnodes)
plot(prune.TreeModelUSCrime)
text(prune.TreeModelUSCrime)
title("Pruned Tree")

summary(prune.crimeTreeMod)



