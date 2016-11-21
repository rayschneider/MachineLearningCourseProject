---
title: "Machine Learning Course Project"
author: "Raymond Schneider"
date: "November 19, 2016"
output: 
  md_document: 
    variant: markdown_github
---
## Introduction

This analysis seeks to take Human Activity Recognition data available at http://groupware.les.inf.puc-rio.br/har and use the data to predict errors in weight lifting technique.  The data captures 5 common errors in weightlifting technique.  Two models will be created and compared against known test data to determine the predictive power of each model.

## Load the Data

First we must load in the data for analysis, and load libraries we know will be used:

```{r libraries, warning=FALSE, message=FALSE}
library(caret)
library(randomForest)
library(rattle)
setwd("C:/Users/Ray/Desktop/MLCourseProject")
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

## Decision Tree Model

For our first model, we'll attempt to fit a Classification and Regression Tree (CART) model.  This appraoch was chosen as the variable we are trying to predict for is a class variable.

```{r decisionTreeModel}
set.seed(2626)
modelDT <- train(classe~., data=training, method="rpart")
print(modelDT, digits=5)
print(modelDT$finalModel, digits=5)
fancyRpartPlot(modelDT$finalModel)
```

```{r decisionTreePredict}
predictDT <- predict(modelDT,testing)
confusionMatrix(testing$classe,predictDT)
```

Looking at the results, this model is not very accurate.  At 0.491 accuracy, this is actually worse than chance (0.50).  It also appears this model is not fitting anything to class D, which seems like it might be a problem.

## Random Forest

Next, we'll try a random forest model.  This model will take advantage of the many variables available, and may preoduve better results.

```{r randomForestModel}
set.seed(2626)
modelRF <- randomForest(classe ~ ., data=training)
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
predictRF <- predict(modelRF,testing)
confusionMatrix(testing$classe,predictRF)
```

The accuracy is much better, 0.99, and the model is predicting well across all of the classes of our outcome.  The sensitivity, specificity, positive and megative predictive power are all high.  This model is much more promising.

## Against Test Data
First, let's test the random forest model.
```{r RFagainstTest}
predictFinal <- predict(modelRF,testDataClean)
print(predictFinal)
```
When comparing these results to the answers provided, the random forest model correctly predicted all 20 variables.

```{r DTagainstTest}
predictFinalDT <- predict(modelDT,testDataClean)
print(predictFinal)
```
This model predicted 8 of 20 results accurately.  Based on these results, we'll use the random forest model.

## In and Out of Sample Error Rates

```{r insample}
insample <- predict(modelRF,training)
confusionMatrix(training$classe,insample)
```

The in-sample error rate is 0, which may indicate overfitting.

As indicated before, the out of sample error rate is 0.0069 (1 - accuracy).

## Conclusion

The random forest model was better able to predict the outcomes for this data set, as evidenced by all metrics observed in this analysis.