library(caret)
library(AppliedPredictiveModeling)
library(kernlab)
library(dplyr)
library(doParallel)
library(randomForest)
library(party)

missClass = function(values,prediction){sum(((prediction > 0.5)*1) != values)/length(values)}


setwd("C:\\Users\\richt\\Documents\\Data Science\\MachineLearning")

training <- read.csv("pml-training.csv")
testing <- read.csv("pml-testing.csv")

inTrain <- createDataPartition(y = training$classe, p=0.25)[[1]]
trainingSet_ <- training[inTrain,]
cvSet_ <- training[-inTrain,]
dim(trainingSet_)
dim(cvSet_)

# check for near-zero-variability
# nearZeroVar(trainingSet_,saveMetrics = TRUE)
nzvCols <- nearZeroVar(trainingSet_)

cols1 <- c(8:160)
cols2 <- cols1[! cols1 %in% nzvCols]

manyNACols <- (colSums(is.na(trainingSet_)) > 1000)
manyNACols_names <- names(manyNACols[manyNACols==1])
manyNAColIndexes <- which(names(trainingSet_) %in% manyNACols_names)

cols <- cols2[! cols2 %in% manyNAColIndexes]

trainingSet <- trainingSet_[,cols]
cvSet <- cvSet_[,cols]

rm(trainingSet_)
rm(cvSet_)

class_col <- which(colnames(trainingSet)=="classe")

set.seed(3433)
#modelFit <- train(classe ~., data=trainingSet, method="rf", metric = "Accuracy", tuneLength =1, trControl=trainControl(method="boot",number=4))
modelFit <- randomForest(classe ~ .,data = trainingSet,importance = TRUE,ntrees = 150)
#missClass(trainingSet$classe, predict(modelFit, newdata = trainingSet[-class_col])) 

plot(modelFit)
varImpPlot(modelFit,cex=.5)  

# test accuracy of the model using the cross-validation set
cvPred <- predict(modelFit, cvSet) 
table(cvSet$classe, cvPred)
confusionMatrix(cvSet$classe, cvPred)
plot(cvPred)







inTrainData <- createFolds(trainingData$classe, k=10)

class_col <- which(colnames(trainingSet)=="classe")

modelFit <- train(y ~., data=trainPC, method="rf", metric = "Accuracy", tuneLength =1, trControl=trainControl(method="boot",number=4))



M <- abs(cor(trainingSet[,-class_col]))
diag(M) <- 0
which(M > 0.8, arr.ind=T)


set.seed(3433)
preProc <- preProcess(trainingSet[,-class_col], method="center","scale")
trainPC <- predict(preProc, trainingSet[,-class_col])
#modelFit <- train(trainingSet$classe ~ ., method="rpart", data=trainingSet[,-class_col])
#modelFit <- train(trainingSet$classe ~ ., method="rf", data=trainingSet)
#modelFit <- train(classe ~., data=trainingSet, method="rf", metric = "Accuracy", tuneLength =1, trControl=trainControl(method="boot",number=4))

#randomForestFit <- train(classe ~ ., method = "rf", data = trainingData,
#                         tuneLength = 5, allowParallel=TRUE,
#                         trControl = trainControl(method = "cv", indexOut = inTrainData))

rf_modelFit <- randomForest(classe ~ .,data = trainPC,importance = TRUE,ntrees = 100)

#cfit <- cforest(classe ~ ., data = trainingSet, controls=cforest_unbiased(ntree=100, mtry=3))

par(mar=c(3,4,4,4))                               
plot(modelFit)
varImpPlot(modelFit,cex=.5)  

# test accuracy of the model using the cross-validation set
testPC <- predict(modelFit, cvSet) 
table(cvSet$classe, testPC)
confusionMatrix(cvSet$classe, testPC)
plot(testPC)

missClass(trainSA$chd, predict(modelFit, newdata = trainSA)) 

#modelFit <- train(classe ~ ., method="glm", data=trainingSet)
#modelFit <- train(classe ~., data=trainingSet, method="rf", trControl=trainControl(method="cv",number=4))
#summary(modelFit.rf)
#modelFit_lda <- train(classe ~ ., method="lda", data=trainingSet)
#modelFit_nb <- train(classe ~ ., method="nb", data=trainingSet)
print(modelFit, digits=3)
testPC <- predict(modelFit, newdata=cvSet[,-class_col])
confusionMatrix(trainingSet$classe, predict(modelFit,testPC))



# find columns in testing set that are all NA because they won't be useful
colsAllNA_test <- sapply(testing, function(x)all(is.na(x)))
colsNamesNotNA_train <- names( colsAllNA_test[colsAllNA_test==0] );
colsNamesNotNA_train <- c(colsNamesNotNA_test, "classe")


class_col_train <- which(colnames(training)=="classe")

# find columns in testing set that are all NA because they won't be useful
colsAllNA_test <- sapply(testing, function(x)all(is.na(x)))
colsNamesNotNA_test <- names( colsAllNA_test[colsAllNA_test==0] );
colsNamesNotNA_train <- c(colsNamesNotNA_test, "classe")

new_train_ <- trainingSet %>% select(roll_belt:classe)
new_train <- new_train_ %>% select(-nzvCols)

#new_train <- new_train_ %>% select(which(names(new_train_) %in% colsNamesNotNA_train))
new_test_ <- testing %>% select(roll_belt:magnet_forearm_z)
new_test <- new_test_ %>% select(which(names(new_test_) %in% colsNamesNotNA_test))

# cols <- c("roll_belt","pitch_belt","yaw_belt","total_accel_belt","gyros_belt_x","gyros_belt_y","gyros_belt_z","accel_belt_x","accel_belt_y","accel_belt_z","magnet_belt_x","magnet_belt_y","magnet_belt_z","roll_arm","pitch_arm","yaw_arm","total_accel_arm","gyros_arm_x","gyros_arm_y","gyros_arm_z","accel_arm_x","accel_arm_y","accel_arm_z","magnet_arm_x","magnet_arm_y","magnet_arm_z","roll_dumbbell","pitch_dumbbell","yaw_dumbbell")
featurePlot(x=trainingSet, y=trainingSet$classe, plot="pairs")

set.seed(3433)
pca <- prcomp(trainingSet[,-class_col],cor=TRUE)
