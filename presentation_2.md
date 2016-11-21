---
title: "Machine Learning Course Project v2"
author: "Raymond Schneider"
date: "November 21, 2016"
output: md_document
---

## Introduction

This analysis seeks to take Human Activity Recognition data available at http://groupware.les.inf.puc-rio.br/har and use the data to predict errors in weight lifting technique.  The data captures 5 common errors in weightlifting technique.  Two models will be created and compared against known test data to determine the predictive power of each model.

## Load the Data

First we must load in the data for analysis, and load libraries we know will be used:

```{r libraries, warning=FALSE, message=FALSE}
library(caret)
library(randomForest)
setwd("C:/Users/schneiderr/Desktop/MLCourseProject")
trainData <- read.csv("pml-training.csv")
testData <- read.csv("pml-testing.csv")
```

Next we take a look at the data.  For the sake of space, we'll not print the summary here.  The data shows there are a lot of variables with NA, 0, or very few data points.  We'll need to clean this data up a bit before we build any models.

```{r summary, eval= FALSE}
summary(trainData)
```

## Data Cleaning

First we'll find any variables which consist entirely of NAs.  Since we looked at the data, we know we can look for variables that are completely NAs.  We remove the NAs in the training data, and do the same to the test data.

```{r removeNA}
trainNA <- sapply(trainData, function(y) sum(length(which(is.na(y)))))
trainNA <- data.frame(trainNA)
trainNA$name <- rownames(trainNA)
drops <- subset(trainNA, trainNA > 0)
drops <- as.list(drops$name)
trainNoNA <- trainData[,!(names(trainData) %in% drops)]
testNoNA <- testData[,!(names(testData) %in% drops)]
```

Next we'll run the nearZeroVar() function in order to remove variables that have very low variance.
```{r removeNZV}
trainNZV <- nearZeroVar(trainNoNA)
trainDataClean <- trainNoNA[,-c(trainNZV)]
testDataClean <- testNoNA[,-c(trainNZV)]
```

We'll also remove the first six variables, as none of these are relevant to the prediction models we will create.  The variables remved here are user and timestamp data.
```{r removeFirst}
trainDataClean <- trainDataClean[,-c(1:6)]
testDataClean <- testDataClean[,-c(1:6)]
```

## Partition Training Data

Now we will take the cleaned training data, and partition to provide a training subset and testing subset, both from the original training data set.

```{r partition}
set.seed(2626)
inTrain <- createDataPartition(trainDataClean$classe, p=0.6, list=FALSE)
training <- trainDataClean[inTrain,]
testing <- trainDataClean[-inTrain,]
```
We'll further split the training group into four smaller groups for cross validation.
```{r crossval}
set.seed(2626)
trainingAlpha <- createDataPartition(training$classe, p=0.25, list=FALSE)
training01 <- training[trainingAlpha,]
trainingBeta <- training[-trainingAlpha,]
set.seed(2626)
trainingGamma <- createDataPartition(trainingBeta$classe, p=0.33, list=FALSE)
training02 <- trainingBeta[trainingGamma,]
trainingDelta <- trainingBeta[-trainingGamma,]
set.seed(2626)
trainingEpsilon <- createDataPartition(trainingDelta$classe, p=0.5, list=FALSE)
training03 <- trainingDelta[trainingEpsilon,]
training04 <-trainingDelta[-trainingEpsilon,]
```

And we'll partition those into sub training and test groups.
```{r subval}
set.seed(2626)
split01 <- createDataPartition(training01$classe, p=0.6, list = FALSE)
train01 <- training01[split01,]
test01 <- training01[-split01,]
set.seed(2626)
split02 <- createDataPartition(training02$classe, p=0.6, list = FALSE)
train02 <- training02[split02,]
test02 <- training02[-split02,]
set.seed(2626)
split03 <- createDataPartition(training03$classe, p=0.6, list = FALSE)
train03 <- training03[split03,]
test03 <- training03[-split03,]
set.seed(2626)
split04 <- createDataPartition(training04$classe, p=0.6, list = FALSE)
train04 <- training04[split04,]
test04 <- training04[-split04,]
```
## Decision Tree Model

For our first model, we'll attempt to fit a Classification and Regression Tree (CART) model.  This approach was chosen as the variable we are trying to predict for is a class variable.  The model will build a single decision tree and try to fit the best choices at ecah node until there are no choices, or no better fit can be found.

```{r decisionTreeModel}
set.seed(2626)
modelDT <- train(classe~., data=train01, method="rpart")
print(modelDT, digits=5)
print(modelDT$finalModel, digits=5)
```

```{r decisionTreePredict}
predictDT <- predict(modelDT,test01)
confusionMatrix(test01$classe,predictDT)
```

Looking at the results, this model is not very accurate.  At 0.4702 accuracy, this is actually worse than chance (0.5).  It also appears this model is not fitting anything to class D, which seems like it might be a problem.  The model is only fitting on 4 variables.

## Random Forest

Next, we'll try a random forest model.  This model will take advantage of the many variables available, and may produce better results.  The random forest builds the model by taking many rabdom subsets of the data and creating mutlitple decision trees, then uses the results of all the decision trees to fit a final model.  This should allow the model to take advantage of more variables, and may handle continuous data better than the rpart model/

```{r randomForestModel}
set.seed(2626)
modelRF <- randomForest(classe ~ ., data=train01)
print(modelRF, digits=5)
```
This is already looking much better.  Let's extract the top variables of importance to see how they compare to the previous model.  I've suppressed all of the variables and just output the top 7.

```{r varImp, results="hide"}
varImp(modelRF)
```
```{r varImpTop}
imp <- varImp(modelRF)
rownames(imp)[order(imp$Overall, decreasing=TRUE)[1:7]]
```

These are actually quite similar to the previous model, but using more variables is increasing the predictive power of the model.

```{r randomFOrestPredict}
predictRF <- predict(modelRF,test01)
confusionMatrix(test01$classe,predictRF)
```

The accuracy is much better, 0.96, and the model is predicting well across all of the classes of our outcome.  The sensitivity, specificity, positive and negative predictive power are all high.  This model is much more promising.

## Cross Validation

I'll run the first random forest model with cross validation now.

```{r cv1}
set.seed(2626)
modelRF01 <- train(classe ~ ., data=train01, method="rf", trControl=trainControl(method = "cv", number = 4))
print(modelRF, digits=5)
```
This gives an accuracy of 0.939.

Let's test against the testing set test01:

```{r cvp1}
predictRF01 <- predict(modelRF,test01)
confusionMatrix(test01$classe,predictRF01)
z1 <- confusionMatrix(test01$classe,predictRF01)
a1 <- z1$overall[1]
```
Accuracy is 0.95.  Let's run the other three:

```{r morecv}
set.seed(2626)
modelRF02 <- train(classe ~ ., data=train02, method="rf", trControl=trainControl(method = "cv", number = 4))
print(modelRF02, digits=5)
predictRF02 <- predict(modelRF02,test02)
confusionMatrix(test02$classe,predictRF02)
z2 <- confusionMatrix(test02$classe,predictRF02)
a2 <- z2$overall[1]

set.seed(2626)
modelRF03 <- train(classe ~ ., data=train03, method="rf", trControl=trainControl(method = "cv", number = 4))
print(modelRF03, digits=5)
predictRF03 <- predict(modelRF03,test03)
confusionMatrix(test03$classe,predictRF03)
z3 <- confusionMatrix(test03$classe,predictRF03)
a3 <- z3$overall[1]

set.seed(2626)
modelRF04 <- train(classe ~ ., data=train04, method="rf", trControl=trainControl(method = "cv", number = 4))
print(modelRF04, digits=5)
predictRF04 <- predict(modelRF04,test04)
confusionMatrix(test04$classe,predictRF04)
z4 <- confusionMatrix(test04$classe,predictRF04)
a4 <- z4$overall[1]

acc <- c(a1,a2,a3,a4)
names(acc) <- c("Accuracy RF1","Accuracy RF2","Accuracy RF3","Accuracy RF4")
acc
```

We can see the accuracy is between 0.928 and 0.959.  Our cross validation supports this model as being farily consistent, and it appears to be farily accurate.

## Against Test Data
Let's test the random forest model against the course quize (the test data).
```{r RFagainstTest}
predictFinal <- predict(modelRF,testDataClean)
print(predictFinal)
```
When comparing these results to the answers provided, the random forest model correctly predicted 19 of 20 variables.  This is in line with our ~0.95 accuracy estimate.

## In and Out of Sample Error Rates

```{r insample}
insample <- predict(modelRF,training)
confusionMatrix(training$classe,insample)
```

The in-sample error rate is 0.05.

```{r oos}
oos <- 1-acc
names(oos) <- c("Error RF1", "Error RF2", "Error RF3", "Error RF4")
oos
mean(oos)
```

Averaging the out of sample error rate for each test of the RF model, we get an error rate of 0.0595.

## Conclusion

The random forest model was better able to predict the outcomes for this data set, as evidenced by all metrics observed in this analysis.

It is worth noting that if the Random Forest model is run against the initial training/testing partition, the model will predict the final testing set answers with 100% accuracy, for the test set.  this illustrates how using a larger sample size size may increase the predicitive power of a model.
