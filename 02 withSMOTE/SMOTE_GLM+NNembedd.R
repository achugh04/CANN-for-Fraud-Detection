library(tidyverse)
library(keras)
require(gbm)
require(data.table)
library(pROC)
library(rpart)
library(ROSE)
library(DMwR)   # Loading DMwr to balance the unbalanced class
# data <- read.csv('./data/carclaims.csv')
# glimpse(data)

data <- read.csv('./data/Pre-Processed.csv')
str(data)

# data$MakeGLM <- as.integer(data$Make)
# data$AccidentAreaGLM <- as.integer(data$AccidentArea)
# data$SexGLM <- as.integer(data$Sex)
# data$MaritalStatusGLM <- as.integer(data$MaritalStatus)
# data$FraudFound <- ifelse(data$FraudFound == "Yes", 1, 0)
# 
# levels(data$MaritalStatus)
# levels(data$PastNumberOfClaims)
# data$PastNumberOfClaims <- ordered(data$PastNumberOfClaims, levels = c( "none", "1", "2 to 4", "more than 4"))
# levels(data$Days.Policy.Accident)
# data$Days.Policy.Accident <- ordered(data$Days.Policy.Accident)
# levels(data$Days.Policy.Claim)
# data$Days.Policy.Claim <- ordered(data$Days.Policy.Claim, levels = c("8 to 15", "15 to 30", "more than 30"))
# levels(data$AgeOfVehicle)
# data$AgeOfVehicle <- ordered(data$AgeOfVehicle, levels = c("less than 4 years", "4 to 6 years", "more than 7"))
# levels(data$NumberOfSuppliments)
# data$NumberOfSuppliments <- ordered(data$NumberOfSuppliments, levels = c("none", "1 to 2", "3 to 5", "more than 5"))
# levels(data$AddressChange.Claim)
# data$AddressChange.Claim <- ordered(data$AddressChange.Claim, levels = c("no change", "0 to 3 years", "4 to 8 years"))
# levels(data$NumberOfCars)
# data$NumberOfCars <- ordered(data$NumberOfCars)

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
# sum(learn$ClaimNb)/sum(learn$Exposure)


##############################################
###############  GLM analysis ###############
##############################################

dataGLM <- data

dataGLM$FraudFound <- ifelse(dataGLM$FraudFound == "Yes", 1, 0)

learnGLM <- dataGLM[ll,]
testGLM <- dataGLM[-ll,]
(n_l <- nrow(learnGLM))
(n_t <- nrow(testGLM))

{t1 <- proc.time()
d.glm <- glm(FraudFound ~ daysDiff + Deductible + Age + Fault + PastNumberOfClaims + 
               VehiclePrice + AddressChange.Claim + Make + DriverRating + VehicleCategory + 
               NumberOfSuppliments + MaritalStatus + BasePolicy + AccidentArea + PoliceReportFiled,
             data=learnGLM, family=binomial())
(proc.time()-t1)}

summary(d.glm)

learnGLM$fitGLM <- fitted(d.glm)
testGLM$fitGLM <- predict(d.glm, newdata=testGLM, type="response")
dataGLM$fitGLM <- predict(d.glm, newdata=dataGLM, type="response")

######################################################
#########  feature pre-processing for (CA)NN Embedding
######################################################

PreProcess.Continuous <- function(var1, dat2){
  names(dat2)[names(dat2) == var1]  <- "V1"
  dat2$X <- as.numeric(dat2$V1)
  dat2$X <- 2*(dat2$X-min(dat2$X))/(max(dat2$X)-min(dat2$X))-1
  names(dat2)[names(dat2) == "V1"]  <- var1
  names(dat2)[names(dat2) == "X"]  <- paste(var1,"X", sep="")
  dat2
}

Features.PreProcess <- function(dat2){
  dat2 <- PreProcess.Continuous("daysDiff", dat2)   
  dat2 <- PreProcess.Continuous("Deductible", dat2)
  dat2 <- PreProcess.Continuous("Age", dat2)
  dat2 <- PreProcess.Continuous("Fault", dat2)
  dat2 <- PreProcess.Continuous("PastNumberOfClaims", dat2)
  # dat2 <- PreProcess.Continuous("VehiclePrice", dat2)
  dat2$VehiclePriceX <- as.integer(dat2$VehiclePrice)-1
  dat2 <- PreProcess.Continuous("AddressChange.Claim", dat2)
  # dat2 <- PreProcess.Continuous("Make", dat2)
  dat2$MakeX <- as.integer(dat2$Make)-1
  dat2 <- PreProcess.Continuous("DriverRating", dat2)
  dat2 <- PreProcess.Continuous("VehicleCategory", dat2)
  dat2 <- PreProcess.Continuous("NumberOfSuppliments", dat2)
  dat2 <- PreProcess.Continuous("MaritalStatus", dat2)
  dat2 <- PreProcess.Continuous("BasePolicy", dat2)
  dat2 <- PreProcess.Continuous("AccidentArea", dat2)
  dat2 <- PreProcess.Continuous("PoliceReportFiled", dat2)
  dat2
}

dataNN <- Features.PreProcess(dataGLM)     

###############################################
#########  choosing learning and test sample
###############################################
table(dataNN$FraudFound)
# dataNN$FraudFound <- ifelse(dataNN$FraudFound == "Yes", 1, 0)

# data_balanced_under <- ovun.sample(FraudFound ~ ., data = dataNN, method = "under", N = 923*2, seed = 1)$data
# table(data_balanced_under$FraudFound)

# set.seed(100)
# ll <- sample(c(1:nrow(data_balanced_under)), round(0.8*nrow(data_balanced_under)), replace = FALSE)

learnNN <- dataNN[ll,]
testNN <- dataNN[-ll,]
(n_l <- nrow(learnNN))
(n_t <- nrow(testNN))



#######################################################
#########  neural network definitions for model (3.11)
#######################################################

learnNN.x <- list(as.matrix(learnNN[,c("VehicleCategoryX", "AgeX", "FaultX", "DriverRatingX",
                                       "MaritalStatusX", "PoliceReportFiledX")]),
                  as.matrix(learnNN[,"VehiclePriceX"]),
                  as.matrix(learnNN[,"MakeX"]),
                  as.matrix(learnNN[,c("daysDiffX", "DeductibleX", "PastNumberOfClaimsX",
                                       "AddressChange.ClaimX", "NumberOfSupplimentsX", "BasePolicyX",
                                       "AccidentAreaX")]),
                  as.matrix(learnNN$fitGLM) )

testNN.x <- list(as.matrix(testNN[,c("VehicleCategoryX", "AgeX", "FaultX", "DriverRatingX",
                                      "MaritalStatusX", "PoliceReportFiledX")]),
                 as.matrix(testNN[,"VehiclePriceX"]),
                 as.matrix(testNN[,"MakeX"]),
                 as.matrix(testNN[,c("daysDiffX", "DeductibleX", "PastNumberOfClaimsX",
                                      "AddressChange.ClaimX", "NumberOfSupplimentsX", "BasePolicyX",
                                      "AccidentAreaX")]),
                 as.matrix(testNN$fitGLM) )

neurons <- c(20,15,10)
# No.Labels <- length(unique(learn$VehBrandX))

###############################################
#########  definition of neural network (3.11)
###############################################

model.2IA <- function(){
  Cont1 <- layer_input(shape = c(6), dtype = 'float32', name='Cont1')
  VehiclePrice <- layer_input(shape = c(1), dtype = 'int32', name = 'Cat1')
  Make <- layer_input(shape = c(1), dtype = 'int32', name = 'Cat2')
  # Cont2 <- layer_input(shape = c(5), dtype = 'float32', name='Cont2')
  Cont3 <- layer_input(shape = c(7), dtype = 'float32', name='Cont3')
  GLM   <- layer_input(shape = c(1), dtype = 'float32', name = 'GLM')     
  x.input <- c(Cont1, VehiclePrice, Make, Cont3, GLM)

  VePrEmb = VehiclePrice %>%
    layer_embedding(input_dim = 4, output_dim = 2, trainable=TRUE,
                    input_length = 1, name = 'VePrEmb') %>%
    layer_flatten(name='VePr_flat')
  
  MakeEmb = Make %>%
    layer_embedding(input_dim = 4, output_dim = 2, trainable=TRUE,
                    input_length = 1, name = 'MakeEmb') %>%
    layer_flatten(name='Make_flat')

  NNetwork1 = list(Cont1, VePrEmb, MakeEmb) %>% layer_concatenate(name='concate') %>%
    layer_dense(units=neurons[1], activation='tanh', name='hidden1') %>%
    layer_dense(units=neurons[2], activation='tanh', name='hidden2') %>%
    layer_dense(units=neurons[3], activation='tanh', name='hidden3') %>%
    layer_dense(units=1, activation='tanh', name='NNetwork1')
                # weights=list(array(0, dim=c(neurons[3],1)), array(0, dim=c(1))))
  # NNetwork1 = Cont1 %>%
  #   layer_dense(units=neurons[1], activation='tanh', name='hidden1') %>%
  #   layer_dense(units=neurons[2], activation='tanh', name='hidden2') %>%
  #   layer_dense(units=neurons[3], activation='tanh', name='hidden3') %>%
  #   layer_dense(units=1, activation='tanh', name='NNetwork1') 
  #               # weights=list(array(0, dim=c(neurons[3],1)), array(0, dim=c(1))))
  #
  # NNetwork2 = Cont2 %>%
  #   layer_dense(units=neurons[1], activation='tanh', name='hidden4') %>%
  #   layer_dense(units=neurons[2], activation='tanh', name='hidden5') %>%
  #   layer_dense(units=neurons[3], activation='tanh', name='hidden6') %>%
  #   layer_dense(units=1, activation='tanh', name='NNetwork2') 
  #               # weights=list(array(0, dim=c(neurons[3],1)), array(0, dim=c(1))))
  #
  NNetwork3 = Cont3 %>%
    layer_dense(units=neurons[1], activation='tanh', name='hidden7') %>%
    layer_dense(units=neurons[2], activation='tanh', name='hidden8') %>%
    layer_dense(units=neurons[3], activation='tanh', name='hidden9') %>%
    layer_dense(units=1, activation='tanh', name='NNetwork3')
                # weights=list(array(0, dim=c(neurons[3],1)), array(0, dim=c(1))))
  #
  NNoutput = list(NNetwork1, NNetwork3, GLM) %>% layer_add(name='Add') %>%
    layer_dense(units=1, activation='sigmoid', name = 'NNoutput')
                 # trainable=TRUE, weights=list(array(c(1), dim=c(1,1)), array(0, dim=c(1))))
  
  model <- keras_model(inputs = x.input, outputs = c(NNoutput))
  model %>% compile(optimizer = optimizer_nadam(), loss = 'binary_crossentropy')        
  model
}

model <- model.2IA()
summary(model)

# may take a couple of minutes if epochs is more than 100
{t1 <- proc.time()
  fit <- model %>% fit(learnNN.x, as.matrix(learnNN$FraudFound), epochs=400, batch_size=500, verbose=0, 
                       validation_data=list(testNN.x, as.matrix(testNN$FraudFound)))
  (proc.time()-t1)}

# This plot should not be studied because in a thorough analyis one should not track 
# out-of-sample losses on the epochs, however, it is quite illustrative, here. 
# oos <- 200* fit[[2]]$val_loss + 200*(-mean(test$ClaimNb)+mean(log(test$ClaimNb^test$ClaimNb)))
# plot(oos, type='l', ylim=c(31.5,32.1), xlab="epochs", ylab="out-of-sample loss", cex=1.5, cex.lab=1.5, main=list(paste("Model GAM+ calibration", sep=""), cex=1.5) )
# abline(h=c(32.07597, 31.50136), col="orange", lty=2)

learn0 <- learnNN     
learn0$fitGANPlus <- as.vector(model %>% predict(learnNN.x))
test0 <- testNN
test0$fitGANPlus <- as.vector(model %>% predict(testNN.x))

pred<-prediction(test0$fitGANPlus,test0$FraudFound)
perf <- performance(pred,"tpr","fpr")
plot(perf)
abline(a=0,b=1, col="red", lty=2)

# Draw ROC curve.

result.roc <- roc(test0$FraudFound, test0$fitGANPlus)
auc(result.roc)
# plot(result.roc, print.thres="best", print.thres.best.method="closest.topleft")


# Get some more values.
result.coords <- coords(
  result.roc, "best", best.method="closest.topleft", ret=c("threshold", "accuracy"))

print(result.coords)

# Make prediction using the best top-left cutoff.
result.predicted.label <- ifelse(test0$fitGANPlus > result.coords[1,1], 1, 0)

xtabs(~  result.predicted.label + test0$FraudFound)

accuracy.meas(test0$FraudFound, result.predicted.label)