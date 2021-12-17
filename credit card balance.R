#The goal of this guide is to show basic credit card balance computations in R using different algorithms in the caret package.

# Load & Inspect ----------------------------------------------------------

# load libraries
library(ISLR)
library(caret)
library(tidyverse)
library(e1071)

data("Credit")
df <- Credit

#shuffle the dataframe 
df <-  df[sample(1:nrow(df)), ]
head(df)

glimpse(df)

#remove IDs
df <- df[,-1]

#transform Cards to a factor variable
summary(df)
df$Cards <- as.factor(df$Cards)

#distribution of factor variables
sapply(df[,c(4,7:10)], table)

#Skewness
# calculate skewness for each numeric variable
skew <- apply(df[,c(1:3,5:6)], 2, skewness)
print(skew)


# Visualise ---------------------------------------------------------------

#Univariate Visualisation
#Use hist to check the distribution of numeric variables
par(mfrow = c(2,3))
for (i in c(1:3,5:6)) {
  hist(df[,i],main = names(df)[i], col = "red")
}

#Box And Whisker Plots
#Detect any outliers
par(mfrow = c(2,3))
for (i in c(1:3,5:6)) {
  boxplot(df[,i], main=names(df)[i])
}

# Feature Selection -------------------------------------------------------

#Correlations
#calculate a correlation matrix for numeric variables
correlations <- cor(df[,c(1:3,5:6)])
#Limit and rating are highly correlated
print(correlations)

#define the control using a random forest selection function
control <- rfeControl(functions=rfFuncs, method="cv", number=10)
seed <- 123
metric <- "RMSE"
#run the RFE algorithm
results <- rfe(df[,1:10], df[,11], metric = metric,
               maximize = ifelse(metric == "RMSE", F, T), sizes=c(1:10), rfeControl=control)
#summarize the results
print(results)
#list the chosen features
predictors(results)
#plot the results
plot(results, type=c("g", "o"))
varImp(results)

#Remove rating attribute and choose top 3 variables for modelling
df <- df[,c("Limit","Student","Income","Balance")]


# Box-Cox Transform numeric variables -------------------------------------------------------------

#reduce the skew of limit and income and make it more Gaussian
#calculate the pre-process parameters from the dataset
preprocessParams <- preProcess(df[,c(1,3)], method=c("BoxCox"))
#summarise transform parameters
print(preprocessParams)
#transform the dataset using the parameters
df_transformed <- predict(preprocessParams, df[,c(1,3)])
#summarise the transformed dataset
summary(df_transformed)

#Confirm the distributions look more Gaussian 
par(mfrow = c(1,2))
for (i in c(1:2)) {
  hist(df_transformed[,i],main = names(df_transformed)[i], col = "red")
}

#Combine other variables
df_transformed <- cbind(df_transformed ,df[,c(2,4)])


# Build Models ------------------------------------------------------------

#Define test harness
control <- trainControl(method="repeatedcv", number=10, repeats = 3)
seed <- 123
metric <- "RMSE"

# Linear Regression
set.seed(seed)
fit.lm <- train(Balance~., data=df_transformed, method="lm", metric=metric, trControl=control)
# GLMNET
set.seed(seed)
fit.glmnet <- train(Balance~., data=df_transformed, method="glmnet", metric=metric, trControl=control)
# kNN
set.seed(seed)
fit.knn <- train(Balance~., data=df_transformed, method="knn", metric=metric, trControl=control)
# SVM Radial
set.seed(seed)
fit.svmRadial <- train(Balance~., data=df_transformed, method="svmRadial", metric = metric, trControl=control, fit=FALSE)
# CART
set.seed(seed)
fit.cart <- train(Balance~., data=df_transformed, method="rpart", metric=metric, trControl=control)
# GBM
set.seed(seed)
fit.gbm <- train(Balance~., data=df_transformed, method="gbm", metric=metric, trControl=control, verbose = FALSE)

results <- resamples(list(lm=fit.lm, glmnet=fit.glmnet,knn=fit.knn, svm=fit.svmRadial,
                          cart=fit.cart, gbm=fit.gbm))

#Table comparison
summary(results)

#Boxplot comparison
par(mfrow=c(3,1))
bwplot(results)


# Tune Best Models -------------------------------------------------------------

#Proceed with tuning top two models

#GBM
print(fit.gbm)

#Grid Search 
control <- trainControl(method="repeatedcv", number=10, repeats=3, search="grid")
set.seed(seed)
gbmGrid <-  expand.grid(interaction.depth = c(1:10), 
                        n.trees = (1:100)*5, 
                        shrinkage = c(0.1,0.05, 0.01),
                        n.minobsinnode = 10)
gbm_gridsearch <- train(Balance~., data=df_transformed, method="gbm", metric=metric, tuneGrid=gbmGrid, trControl=control, verbose = FALSE)

print(gbm_gridsearch)
print(gbm_gridsearch$bestTune)

#Tuning GBM model helped to reduce RMSE from 92.82 to 84.27
print(min(gbm_gridsearch$results$RMSE))



# SVM Radial
print(fit.svmRadial)

#Set tuneLength argument to 10
control <- trainControl(method="repeatedcv", number=10, repeats=3)

set.seed(seed)
svmRadial_gridsearch <- train(Balance~., data=df_transformed, method="svmRadial", metric = metric, trControl=control,
                              tuneLength = 10, fit=FALSE)

#The elbow is formed at Cost = 4  
plot(svmRadial_gridsearch)

#After tuning,  RMSE  decreased from 68.35 to 57.69
print(svmRadial_gridsearch)


