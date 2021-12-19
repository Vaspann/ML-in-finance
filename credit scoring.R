#The goal of this guide is to show credit scoring (Good vs Bad) computation in R using Keras.


# Load & Inspect ----------------------------------------------------------

library(keras)
library(tidyverse)
library(caret)
data("GermanCredit")

glimpse(GermanCredit)
summary(GermanCredit)

#Target variable distribution 
table(GermanCredit$Class)


# Alter feature classes ------------------------------------------------------

#Transform factor features to integers
glimpse(GermanCredit[,1:10])
sapply(GermanCredit[,c(3,4,6,7,10)], table)
summary(GermanCredit[,1:10])
GermanCredit$Class <- as.integer(as.factor(ifelse(GermanCredit$Class == "Good", "1", "0")))-1
GermanCredit$InstallmentRatePercentage <- as.integer(as.factor(GermanCredit$InstallmentRatePercentage))-1
GermanCredit$ResidenceDuration <- as.integer(as.factor(GermanCredit$ResidenceDuration))-1
GermanCredit$NumberExistingCredits <- as.integer(as.factor(GermanCredit$NumberExistingCredits))-1
GermanCredit$NumberPeopleMaintenance <- as.integer(as.factor(GermanCredit$NumberPeopleMaintenance))-1


# Train & Test  -----------------------------------------------------------

set.seed(123)  # for reproducibility
#split into training (80%) and testing set (20%)
splittt <-  createDataPartition(GermanCredit$Class, p = .8, list = F)
train <-  GermanCredit[splittt, ]
test <-  GermanCredit[-splittt, ]


# Viusaluse distributions --------------------------------------------------

#Univariate Visualization
#Histograms
par(mfrow = c(1,3))
for (i in c(1:2,5)) {
  hist(train[,i],main = names(train)[i], col = "red")
}

# Box-Cox Transform numeric variables -------------------------------------------------------------

#reduce the skew and  make it more Gaussian
# calculate the pre-process parameters from the dataset
preprocessParams <- preProcess(train[,c(1:2,5)], method=c("BoxCox"))
# summarize transform parameters
print(preprocessParams)
# transform the dataset using the parameters
train_transformed <- predict(preprocessParams, train[,c(1:2,5)])
# summarise the transformed dataset
summary(train_transformed)

#Confirm the distributions look more Gaussian 
par(mfrow = c(1,3))
for (i in c(1:3)) {
  hist(train_transformed[,i],main = names(train_transformed)[i], col = "red")
}

#Combine new columns
train_transformed <- cbind(train_transformed ,train[,c(-1,-2,-5)])

#Apply to test set
preprocessParams <- preProcess(test[,c(1:2,5)], method=c("BoxCox"))
test_transformed <- predict(preprocessParams, test[,c(1:2,5)])
test_transformed <- cbind(test_transformed ,test[,c(-1,-2,-5)])


# Create matrix and Label -------------------------------------------------

trainlabel <- train_transformed[,"Class"]
trainlabel <- to_categorical(trainlabel,2)
trainx <- as.matrix(train_transformed[,-10])
dimnames(trainx) <- NULL

testlabel <- test_transformed[,"Class"]
testlabel <- to_categorical(testlabel,2)
testx <- as.matrix(test_transformed[,-10])
dimnames(testx) <- NULL

##Normalise matrix values
Ntrainx <- normalize(trainx)
Ntestx <- normalize(testx)

summary(Ntrainx[,1:10])


# Build a Neural Network model with one hidden layer using Keras ----------------------------------------------

#create a sequential model 
model <- keras_model_sequential()

#One hidden layer with 10 neurons
model%>% 
  layer_dense(units = 10, activation = "relu", kernel_initializer = "normal", input_shape = c(61)) %>% 
  layer_dense(units = 2, activation = "sigmoid")

summary(model)

#Compile
model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(),  
  metrics = c("accuracy")
)

#run model
NN <- model%>%
  fit(Ntrainx, trainlabel, epoch = 100, batch_size = 20, validation_split = 0.2)

plot(NN)

#evaluate on Ntestx
#Model Accuracy % 75.5 vs Majority Class Accuracy % 65
table(test$Class)
score <- model %>% evaluate(Ntestx, testlabel)
pred <- model %>% predict_classes(Ntestx)

confusionMatrix(as.factor(pred),as.factor(test$Class))
