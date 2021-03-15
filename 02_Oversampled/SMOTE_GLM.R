library(tidyverse)
library(keras)
require(gbm)
require(data.table)
library(pROC)
library(rpart)
library(ROSE)
library(DMwR)   # Loading DMwr to balance the unbalanced class

data <- read.csv('./data/Pre-Processed.csv')
str(data)

levels(data$MaritalStatus)
levels(data$PastNumberOfClaims)
data$PastNumberOfClaims <- factor(data$PastNumberOfClaims, levels = c( "none", "1", "2 to 4", "more than 4"))
levels(data$Days.Policy.Accident)
data$Days.Policy.Accident <- factor(data$Days.Policy.Accident)
levels(data$Days.Policy.Claim)
data$Days.Policy.Claim <- factor(data$Days.Policy.Claim, levels = c("8 to 15", "15 to 30", "more than 30"))
levels(data$AgeOfVehicle)
data$AgeOfVehicle <- factor(data$AgeOfVehicle, levels = c("less than 4 years", "4 to 6 years", "more than 7"))
levels(data$NumberOfSuppliments)
data$NumberOfSuppliments <- factor(data$NumberOfSuppliments, levels = c("none", "1 to 2", "3 to 5", "more than 5"))
levels(data$AddressChange.Claim)
data$AddressChange.Claim <- factor(data$AddressChange.Claim, levels = c("no change", "0 to 3 years", "4 to 8 years"))
levels(data$NumberOfCars)
data$NumberOfCars <- factor(data$NumberOfCars)
data$FraudFound <- as.factor(data$FraudFound)
str(data)
# data$FraudFound <- ifelse(data$FraudFound == "Yes", 1, 0)

###############################################
#########  choosing learning and test sample
###############################################

## Smote : Synthetic Minority Oversampling Technique To Handle Class Imbalancy In Binary Classification
balanced.data <- SMOTE(FraudFound ~., data, perc.over = (14000/923)*100, k = 5, perc.under = 105)
table(balanced.data$FraudFound)

data <- balanced.data

set.seed(100)
ll <- sample(c(1:nrow(data)), round(0.8*nrow(data)), replace = FALSE)
learn <- data[ll,]
test <- data[-ll,]
(n_l <- nrow(learn))
(n_t <- nrow(test))


table(learn$FraudFound)

## Smote : Synthetic Minority Oversampling Technique To Handle Class Imbalancy In Binary Classification
# balanced.data <- SMOTE(FraudFound ~., learn, perc.over = (9000/754)*100, k = 5, perc.under = 135)
# table(balanced.data$FraudFound)
# 
# learn <- balanced.data


##############################################
###############  GLM analysis ###############
##############################################

dataGLM <- data

learn$FraudFound <- ifelse(learn$FraudFound == "Yes", 1, 0)
test$FraudFound <- ifelse(test$FraudFound == "Yes", 1, 0)

learnGLM <- learn
testGLM <- test
(n_l <- nrow(learnGLM))
(n_t <- nrow(testGLM))

{t1 <- proc.time()
d.glm <- glm(FraudFound ~ daysDiff + Deductible + Age + Fault + PastNumberOfClaims + 
               VehiclePrice + AddressChange.Claim + Make + DriverRating + VehicleCategory + 
               NumberOfSuppliments + MaritalStatus + BasePolicy + AccidentArea + PoliceReportFiled,
             data=learnGLM, family=binomial)
(proc.time()-t1)}

summary(d.glm)

learnGLM$fitGLM <- fitted(d.glm)
testGLM$fitGLM <- predict(d.glm, newdata=testGLM, type="response")
dataGLM$fitGLM <- predict(d.glm, newdata=dataGLM, type="response")


result.roc <- roc(testGLM$FraudFound, testGLM$fitGLM)
auc(result.roc)
# plot(result.roc, print.thres="best", print.thres.best.method="closest.topleft")

# Get some more values.
result.coords <- coords(
  result.roc, "best", best.method="closest.topleft", ret=c("threshold", "accuracy"))

print(result.coords)

pred<-prediction(testGLM$fitGLM,testGLM$FraudFound)
perf <- performance(pred,"tpr","fpr")
plot(perf)
abline(a=0,b=1, col="red", lty=2)

# Make prediction using the best top-left cutoff.
result.predicted.label <- ifelse(testGLM$fitGLM > result.coords[1,1], 1, 0)

xtabs(~ result.predicted.label + testGLM$FraudFound)

accuracy.meas(testGLM$FraudFound, result.predicted.label)