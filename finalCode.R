## Analysis of Credit Approval Data Set
## Source of Dataset : UCI Machine Learning Respository
## Author of Analysis: Pragna Bhatt
# ---------------------------------------------------------------------------------------------------------------

## Reading in dataset, and some exploratory data analysis. 

library(mice)
library(VIM)
library(ggplot2)
library(faraway)
library(MASS)
#library(safeBinaryRegression)

crx <- read.csv("crx.data", header = FALSE)

dim(crx) # 690 rows and 16 columns 
head(crx)
str(crx)

# Missing values in the data were denoted with '?' - I replaced them with NA.
lapply(crx, unique) 
crx <- replace(crx, crx == "?", NA) # Replace

# Now I converted V2 to numeric, renamed V16 (our response variable) as 'approval', and converted it to 0s and 1s from - and +.
# I also converted all other 'character' variables to factors.
crx$V2 <- as.numeric(crx$V2)
colnames(crx)[16] <- "approval"
crx$approval <- ifelse(crx$approval == "+", 1, 0)
crx$V1 <- factor(crx$V1)
crx$V4 <- factor(crx$V4)
crx$V5 <- factor(crx$V5)
crx$V6 <- factor(crx$V6)
crx$V7 <- factor(crx$V7)
crx$V9 <- factor(crx$V9)
crx$V10 <- factor(crx$V10)
crx$V12 <- factor(crx$V12)
crx$V13 <- factor(crx$V13)
crx$V14 <- factor(crx$V14)
crx$approval <- factor(crx$approval)
str(crx)

# Check out missing values.
sum(is.na(crx))/prod(dim(crx)) # About 0.6% of data is missing. 
summary(crx) # shows number of NA's for numeric variables.
md.pattern(crx) 
aggr(crx, col=c('navyblue','red'), numbers=TRUE, sortVars=TRUE, labels=names(crx), cex.axis=.7, gap=3, ylab=c("Histogram of missing data","Pattern"))

# So according to these plots, V14 seems to have the most missing values (13 missing values).
# For V14, I decided to completely remove this variable from the dataset because it has 170 levels 
# and will cause problems when fitting a logistic regression model. If we had more information
# on what V14 was we could create smaller groups from it or modify it, but unfortunately since there is 
# no information on what the variables are, I have decided it best to throw the variable out.

sum(is.na(crx$V14))
str(factor(crx$V14))  
crx$V14 <- NULL

# Creating holdout sample for later (before doing multiple imputation).
smp_size <- floor(0.7 * nrow(crx))
smp_size

set.seed(123456)
train_ind <- sample(seq_len(nrow(crx)), size = smp_size)
train <- crx[train_ind,]
test <- crx[-train_ind,]

# I will use multiple imputation to fill in the rest of the missing values. 

sum(is.na(train))/prod(dim(train)) # About 0.6% missing values in train set.
imptrain <- mice(train, nnet.MaxNWts = 3000, seed=500)
summary(imptrain)
completedtrain <- complete(imptrain,1)
completedtrain2 <- complete(imptrain, 2)
completedtrain3 <- complete(imptrain, 3)
completedtrain4 <- complete(imptrain, 4)
completedtrain5 <- complete(imptrain, 5)

# Before I start modeling, it may be helpful to see the distributions of our variables so here
# are some visualizations. We notice here that the numeric variables are strongly skewed to the right.

p <- ggplot(completedtrain, aes(fill = factor(approval)))

# Visualizing Categorical/Binary Variables

p + geom_bar(aes(x = V1))
p + geom_bar(aes(x = V4)) # Low 'l' category count, but it seems to only have approvals.
p + geom_bar(aes(x = V5)) # Low 'gg' category count, but only approvals.
p + geom_bar(aes(x = V6))
p + geom_bar(aes(x = V7))
p + geom_bar(aes(x = V9)) # Wow! Only two categories here, and 't' seems to majority approvals and 'f' majority rejections.
p + geom_bar(aes(x = V10))
p + geom_bar(aes(x = V12))
p + geom_bar(aes(x = V13))

# Visualizing Numeric Variables 

p + geom_histogram(aes(x = V2), bins = 20)
p + geom_histogram(aes(x = V3), bins = 20)
p + geom_histogram(aes(x = V8), bins = 20)
p + geom_histogram(aes(x = V11), bins = 20)
p + geom_histogram(aes(x = V15), bins = 20) # Very strong skew to the right. 
# All numeric variables have strong skew to the right but especially V15. 


# Using Agresti's Purposeful Selection Method to build logistic model on each imputed data set.

# No known main effects so I tested each effect individually to see which one is effective individually (at 0.2 level).
# (similar results on each imputated dataset)
summary(glm(formula = approval~V1, family = binomial(link = "logit"), completedtrain)) # not significant
summary(glm(formula = approval~V2, family = binomial(link = "logit"), completedtrain)) # significant
summary(glm(formula = approval~V3, family = binomial(link = "logit"), completedtrain)) # significant
summary(glm(formula = approval~V4, family = binomial(link = "logit"), completedtrain)) # not significant
summary(glm(formula = approval~V5, family = binomial(link = "logit"), completedtrain)) # significant
summary(glm(formula = approval~V6, family = binomial(link = "logit"), completedtrain)) # signicant
summary(glm(formula = approval~V7, family = binomial(link = "logit"), completedtrain)) # signicant
summary(glm(formula = approval~V8, family = binomial(link = "logit"), completedtrain)) # significant
summary(glm(formula = approval~V9, family = binomial(link = "logit"), completedtrain)) # significant
summary(glm(formula = approval~V10, family = binomial(link = "logit"), completedtrain)) # significant
summary(glm(formula = approval~V11, family = binomial(link = "logit"), completedtrain)) # significant
summary(glm(formula = approval~V12, family = binomial(link = "logit"), completedtrain)) # significant
summary(glm(formula = approval~V13, family = binomial(link = "logit"), completedtrain)) # significant
summary(glm(formula = approval~V15, family = binomial(link = "logit"), completedtrain)) # significant
         
fullModel <- approval ~ V2 + V3 + V5 + V6 + V7 + V8 + V9 + V10 + V11 + V12 + V13 + V15

#Backwards selection on each imputated dataset using above formula. 

sets <- list(completedtrain, completedtrain2, completedtrain3, completedtrain4, completedtrain5)
formulas <- c()
for (i in seq(1, length(sets))){
  fullLogisticModel <- glm(formula = fullModel, family = binomial(link = "logit"), data=sets[[i]])
  logisticModel_back <- step(fullLogisticModel, direction = "back")
  formulas <- append(formulas, values = logisticModel_back$formula)
}

formulas

# We get two distinct formulas from the backwards selection.  

fit <- with(data = imptrain, glm(approval ~ V5 + V7 + V9 + V11 + V13 + V15, family = binomial(link = "logit")))
fit$analyses
fit2 <- with(data = imptrain, glm(approval ~ V5 + V6 + V9 + V11 + V13 + V15, family = binomial(link = "logit")))
fit2$analyses
fit3 <- with(data = imptrain, glm(approval ~ V5 + V6 + V7 + V9 + V11 + V13 + V15, family = binomial(link = "logit")))
fit3$analyses

# Looking at the above models, fit2 gives lowest AIC for all five imputed data sets.
# To this model I add V1 and V4 which I took out before and see if AIC is lower.

fit4 <- with(data = imptrain, glm(approval ~ V1 + V4 + V5 + V6 + V9 + V11 + V13 + V15, family = binomial(link = "logit")))
fit4$analyses # Not much improvement in model. So don't keep V1 and V4.

# Predictions - I only kept one table here since all five models gave same results.

imptest <- mice(test, nnet.MaxNWts = 3000, seed=500)
completedtest <- complete(imptest, 1)
completedtest2 <- complete(imptest, 2)
completedtest3 <- complete(imptest, 3)
completedtest4 <- complete(imptest, 4)
completedtest5 <- complete(imptest, 5)

thresh  <- 0.5      # threshold for categorizing predicted probabilities
pred <- predict(fit2$analyses[[1]], newdata=completedtest, type="response")
predFac <- cut(pred, breaks=c(-Inf, thresh, Inf), labels=c("0", "1"))
cTab    <- table(completedtest$approval, predFac, dnn=c("actual", "predicted"))
addmargins(cTab)

# AUC plot
library(pROC)
rocplot <- roc(approval ~ fitted(fit2$analyses[[1]]), data = completedtrain)
plot.roc(rocplot, legacy.axis=TRUE)
auc(rocplot)

cor(as.numeric(completedtrain$approval), fitted(fit2$analyses[[1]]))

# Now for inferences.

combine <- pool(fit2)
summary(combine, conf.int = TRUE)
summary(combine, conf.int = TRUE, exponentiate=TRUE)

# Profile Likelihood Intervals

fit <- fit2$analyses[[1]]
summary(fit)
library(car)
Anova(fit)
library(profileModel)
exp(confintModel(fit, objective = "ordinaryDeviance", method = "zoom", endpoint.tolerance = 1e-08))


# ---------------------------------------------------------------------------------------------------------------------------------

### NEXT MODEL: Classification Tree and Random Forest

# Similar results on all 5 imputated data sets so I've only included one here.

completedtrain$approval <- factor(completedtrain$approval)
library(rpart)
fit <- rpart(approval ~ V1 + V2 + V3 + V4 + V5 + V6 + V7 + V8 + V9 + V10 + V11 + V12 + V13 + V15, method = "class", data = completedtrain)
plotcp(fit)
p.fit <- prune(fit, cp = 0.03)
library(rpart.plot)
rpart.plot(p.fit, extra=1, digits=4, box.palette = 0)

pred <- predict(p.fit, newdata = completedtest, type = "class")
pred <- as.numeric(pred)
pred <- ifelse(pred == 2, 1, 0)
predFac <- cut(pred, breaks=c(-Inf, thresh, Inf), labels=c("0", "1"))
cTab <- table(completedtest$approval, predFac, dnn=c("actual", "predicted"))
addmargins(cTab)

# Random Forest

set.seed(12345)
library(randomForest)
fit <- randomForest(approval ~ V1 + V2 + V3 + V4 + V5 + V6 + V7 + V8 + V9 + V10 + V11 + V12 + V13 + V15, data = completedtrain, ntree=2000, importance = TRUE)
varImpPlot(fit)

pred <- predict(fit, newdata = completedtest, type = "class")
pred <- as.numeric(pred)
pred <- ifelse(pred == 2, 1, 0)
predFac <- cut(pred, breaks=c(-Inf, thresh, Inf), labels=c("0", "1"))
cTab <- table(completedtest$approval, predFac, dnn=c("actual", "predicted"))
addmargins(cTab)

