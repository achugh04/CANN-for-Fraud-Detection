library(tidyverse)
library(keras)
require(gbm)
require(data.table)
library(pROC)
library(rpart)
library(ROSE)

data <- read.csv('./data/Pre-Processed.csv')
str(data)

data %>% select(-c(Medical.Service.Provider.ID)) -> data


# levels(data$MaritalStatus)
# levels(data$PastNumberOfClaims)
# data$PastNumberOfClaims <- factor(data$PastNumberOfClaims, levels = c( "none", "1", "2 to 4", "more than 4"))
# levels(data$Days.Policy.Accident)
# data$Days.Policy.Accident <- factor(data$Days.Policy.Accident)
# levels(data$Days.Policy.Claim)
# data$Days.Policy.Claim <- factor(data$Days.Policy.Claim, levels = c("8 to 15", "15 to 30", "more than 30"))
# levels(data$AgeOfVehicle)
# data$AgeOfVehicle <- factor(data$AgeOfVehicle, levels = c("less than 4 years", "4 to 6 years", "more than 7"))
# levels(data$NumberOfSuppliments)
# data$NumberOfSuppliments <- factor(data$NumberOfSuppliments, levels = c("none", "1 to 2", "3 to 5", "more than 5"))
# levels(data$AddressChange.Claim)
# data$AddressChange.Claim <- factor(data$AddressChange.Claim, levels = c("no change", "0 to 3 years", "4 to 8 years"))
# levels(data$NumberOfCars)
# data$NumberOfCars <- factor(data$NumberOfCars)
# data$Class <- as.factor(data$Class)
# str(data)

data$Class <- as.factor(data$Class)

###############################################
#########  choosing learning and test sample
###############################################

# set.seed(100)
ll <- sample(c(1:nrow(data)), round(0.8*nrow(data)), replace = FALSE)
learn <- data[ll,]
test <- data[-ll,]
(n_l <- nrow(learn))
(n_t <- nrow(test))
# sum(learn$ClaimNb)/sum(learn$Exposure)

##################################################
# Applying GBM to check for the feature importance
##################################################

dataGBM <- data
dataGBM$Class <- ifelse(dataGBM$Class == "0", 0, 1)

learn <- dataGBM[ll,]
test <- dataGBM[-ll,]

param = c(100, 5, 0.03)
train <- as.data.table(learn)
g = gbm(Class ~ ., data=train[,with=FALSE], n.trees = param[1], interaction.depth = param[2], shrinkage = param[3], distribution = "bernoulli", verbose=T)
summary(g)

pf = predict(g, test, n.trees = param[1], type = "response")
hist(pf)

## Calculating ROC Curve for model
library(ROCR)
require(verification)

pred<-prediction(pf,test$Class)
perf <- performance(pred,"tpr","fpr")
plot(perf)
abline(a=0,b=1, col="red", lty=2)


##############################################
###############  GLM analysis ###############
##############################################

dataGLM <- data

dataGBM$Class <- ifelse(dataGBM$Class == "0", 0, 1)

learnGLM <- dataGLM[ll,]
testGLM <- dataGLM[-ll,]
(n_l <- nrow(learnGLM))
(n_t <- nrow(testGLM))

{t1 <- proc.time()
d.glm <- glm(Class ~ .,
             data=learnGLM, family=binomial())
(proc.time()-t1)}

summary(d.glm)

learnGLM$fitGLM <- fitted(d.glm)

testGLM$fitGLM <- predict(d.glm, newdata=testGLM, type="response")
dataGLM$fitGLM <- predict(d.glm, newdata=dataGLM, type="response")


result.roc <- roc(testGLM$Class, testGLM$fitGLM)
auc(result.roc)
# plot(result.roc, print.thres="best", print.thres.best.method="closest.topleft")

# Get some more values.
result.coords <- coords(
  result.roc, "best", best.method="closest.topleft", ret=c("threshold", "accuracy"))

print(result.coords)

pred<-prediction(testGLM$fitGLM,testGLM$Class)
perf <- performance(pred,"tpr","fpr")
plot(perf)
abline(a=0,b=1, col="red", lty=2)

# Make prediction using the best top-left cutoff.
result.predicted.label <- ifelse(testGLM$fitGLM > result.coords[1,1], 1, 0)

xtabs(~ result.predicted.label + testGLM$Class)

accuracy.meas(testGLM$Class, result.predicted.label)