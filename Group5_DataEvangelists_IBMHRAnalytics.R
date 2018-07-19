# Group 5: Shipra Shinde, Mayuri Deshpande, Vishal Budhani, Omkar Bapat, Sachin Singh
# Dataset Source: Kaggle - https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset. 

library("rpart")
library("rpart.plot")
library("C50")
library("caret")
library("class")
library("gmodels")
library("randomForest")
library("neuralnet")
library("rattle")
library("varhandle")
library("ggplot2")
library("arules")
library("arulesViz")
library("Hmisc")
library("ClustOfVar")

# set the working directory
setwd("E:/KDD/RIntroduction/Project")

# Read in the hr_analytics_data
hr_analytics_data <- read.csv(file = "HREmployeeAttrition.csv", stringsAsFactors=TRUE)
# correct the column name
colnames(hr_analytics_data)[1] <- 'Age'

# keep copy of the data
data <- hr_analytics_data
hr_data <- hr_analytics_data

# remove the unary columns
cols.dont.want <- c("Over18", "EmployeeCount", "EmployeeNumber", "StandardHours")
# remove multiple columns
hr_analytics_data <- hr_analytics_data[, ! names(hr_analytics_data) %in% cols.dont.want, drop = F]

# convert the categorical variable into flag variable. Attrition is the target variable
hr_analytics_data$Attrition <- ifelse(hr_analytics_data$Attrition == "No", 0, 1)

# normalize the numeric variables using min max normalization
hr_analytics_data$DailyRate <- (hr_analytics_data$DailyRate - min(hr_analytics_data$DailyRate))/(max(hr_analytics_data$DailyRate) - min(hr_analytics_data$DailyRate))
hr_analytics_data$HourlyRate <- (hr_analytics_data$HourlyRate - min(hr_analytics_data$HourlyRate))/(max(hr_analytics_data$HourlyRate) - min(hr_analytics_data$HourlyRate))
hr_analytics_data$MonthlyRate <- (hr_analytics_data$MonthlyRate - min(hr_analytics_data$MonthlyRate))/(max(hr_analytics_data$MonthlyRate) - min(hr_analytics_data$MonthlyRate))
hr_analytics_data$MonthlyIncome <- (hr_analytics_data$MonthlyIncome - min(hr_analytics_data$MonthlyIncome))/(max(hr_analytics_data$MonthlyIncome) - min(hr_analytics_data$MonthlyIncome))
hr_analytics_data$Age <- (hr_analytics_data$Age - min(hr_analytics_data$Age))/(max(hr_analytics_data$Age) - min(hr_analytics_data$Age))
hr_analytics_data$DistanceFromHome <- (hr_analytics_data$DistanceFromHome - min(hr_analytics_data$DistanceFromHome))/(max(hr_analytics_data$DistanceFromHome) - min(hr_analytics_data$DistanceFromHome))
hr_analytics_data$YearsAtCompany <- (hr_analytics_data$YearsAtCompany - min(hr_analytics_data$YearsAtCompany))/(max(hr_analytics_data$YearsAtCompany) - min(hr_analytics_data$YearsAtCompany))
hr_analytics_data$YearsSinceLastPromotion <- (hr_analytics_data$YearsSinceLastPromotion - min(hr_analytics_data$YearsSinceLastPromotion))/(max(hr_analytics_data$YearsSinceLastPromotion) - min(hr_analytics_data$YearsSinceLastPromotion))
hr_analytics_data$YearsWithCurrManager <- (hr_analytics_data$YearsWithCurrManager - min(hr_analytics_data$YearsWithCurrManager))/(max(hr_analytics_data$YearsWithCurrManager) - min(hr_analytics_data$YearsWithCurrManager))
hr_analytics_data$TotalWorkingYears <- (hr_analytics_data$TotalWorkingYears - min(hr_analytics_data$TotalWorkingYears))/(max(hr_analytics_data$TotalWorkingYears) - min(hr_analytics_data$TotalWorkingYears))

# convert the categorical variables into numeric
hr_analytics_data$BusinessTravel = as.numeric(hr_analytics_data$BusinessTravel)
hr_analytics_data$Department = as.numeric(hr_analytics_data$Department)
hr_analytics_data$BusinessTravel = as.numeric(hr_analytics_data$BusinessTravel)
hr_analytics_data$EducationField = as.numeric(hr_analytics_data$EducationField)
hr_analytics_data$Gender = as.numeric(hr_analytics_data$Gender)
hr_analytics_data$JobRole = as.numeric(hr_analytics_data$JobRole)
hr_analytics_data$MaritalStatus = as.numeric(hr_analytics_data$MaritalStatus)
hr_analytics_data$OverTime = as.numeric(hr_analytics_data$OverTime)

write.csv(hr_analytics_data, file = "ModelingHREmployeeAttrition.csv", row.names = FALSE)

# Partition the data set using two fold cross fold validation
inTrain <- createDataPartition(hr_analytics_data$Attrition,p=0.75,list = FALSE)
train_data <- hr_analytics_data[inTrain,]
write.csv(train_data, file = "TrainHREmployeeAttrition.csv", row.names = FALSE)
test_data <- hr_analytics_data[-inTrain,]
write.csv(test_data, file = "TestHREmployeeAttrition.csv", row.names = FALSE)
train_cl <- hr_analytics_data[inTrain, 2] 
test_cl <-hr_analytics_data[-inTrain, 2]

#  ************************ Prediction Using Classification and Regression Trees (CART)  ****************************
train_data$Attrition <- ifelse(train_data$Attrition == 0, "No", "Yes")
cartfit <- rpart(train_data$Attrition ~ .,data = train_data)
# predict the values for the test data based on the training
p <- predict(cartfit, test_data, method = "class")
print(cartfit)
# Plot the decision tree
rpart.plot(cartfit)
# print the confusion matrix
confusionMatrix(round(p),test_cl)
train_data$Attrition <- ifelse(train_data$Attrition == "No", 0, 1)
# Accuracy : 0.8201635 

#  ************************ Prediction Using C5.0 Algorithm ****************************
# Put the predictors into 'x', the response into 'y'
x <- train_data[, c(1, 3:31)]
y <- as.factor(train_data$Attrition)
# use the C5.0 algorithm to train the model
c50fit <- C5.0(x, y)
# print the summary
summary(c50fit)
# predict the values for the test data
p <- predict(c50fit, test_data, method = "class")
# print the confusion matrix
confusionMatrix(p,test_cl)
# Accuracy : 0.8256131  

#  ************************ Prediction Using k-Nearest Neighbor Algorithm ****************************
# train the model
model_knn <- knn(train = train_data, test = test_data, cl = train_cl, k = 10)
# predict the  values for the test data
CrossTable(x=test_cl,y=model_knn,prop.chisq=FALSE)
# print the confusion matrix
confusionMatrix(model_knn,test_cl)
# Accuracy : 0.8474114  

# ************************ Prediction Using Random Forest ****************************
# train the model
fit <- randomForest(as.factor(train_data$Attrition) ~ .,data=train_data,importance=TRUE,  ntree=2000)
varImpPlot(fit)
# predict the values for the test data
Prediction <- predict(fit, test_data)
# print the confusion matrix
confusionMatrix(Prediction,test_cl)
# Accuracy : 0.8610354   

# ************************ Prediction Using Neural Network ****************************
model_neuralnet <- train(train_data, train_cl, method='neuralnet')
Predictions_neuralnet <- predict(model_neuralnet, test_data)
confusionMatrix(round(Predictions_neuralnet),test_cl)
plot(model_neuralnet)
# Accuracy : 1 

# ***************************** Hierarchical Classification ***************************************
# run variable clustering excluding the target variable (churn) 
tree <- hclustvar(X.quanti = train_data[,1:30])
plot(tree)

# requesting for 25 bootstrap samplings and a plot
stab <- stability(tree, B=25) 
plot(stab, main="Stability of the partitions")
boxplot(stab$matCR, main = "Dispersion of the adjusted Rand index")

# cut the variable tree into 9(?) groups of variables 
clus<-cutreevar(tree,9,matsim=TRUE)
# print the list of variables in each cluster groups
print(clus$var)

# ***************************** ASSOCIATION ***************************************

cols.dont.want <- c("Over18", "EmployeeCount", "EmployeeNumber", "StandardHours")
hr_data <- hr_data[, ! names(hr_data) %in% cols.dont.want, drop = F]

# bin the income variable using k-means
incomeCluster <- binning(hr_data$MonthlyIncome, bins=5, method="kmeans", ordered=FALSE)
hr_data$MonthlyIncome <- as.numeric(as.factor(incomeCluster))
hr_data$MonthlyIncome <- as.factor(hr_data$MonthlyIncome)

# bin the age variable using k-means
ageCluster <- binning(hr_data$Age, bins=5, method="kmeans", ordered=FALSE)
hr_data$Age <- as.numeric(as.factor(ageCluster))
hr_data$Age <- as.factor(hr_data$Age)
match("Age",names(hr_data))

# bin the PercentSalaryHike variable using k-means
hikeCluster <- binning(hr_data$PercentSalaryHike, bins=5, method="kmeans", ordered=FALSE)
hr_data$PercentSalaryHike <- as.numeric(as.factor(hikeCluster))
hr_data$PercentSalaryHike <- as.factor(hr_data$PercentSalaryHike)

# bin the DailyRate variable using k-means
DailyRateCluster <- binning(hr_data$DailyRate, bins=5, method="kmeans", ordered=FALSE)
hr_data$DailyRate <- as.numeric(as.factor(DailyRateCluster))
hr_data$DailyRate <- as.factor(hr_data$DailyRate)

# bin the HourlyRate variable using k-means
HourlyRateCluster <- binning(hr_data$HourlyRate, bins=5, method="kmeans", ordered=FALSE)
hr_data$HourlyRate <- as.numeric(as.factor(HourlyRateCluster))
hr_data$HourlyRate <- as.factor(hr_data$HourlyRate)

# bin the MonthlyRate variable using k-means
MonthlyRateCluster <- binning(hr_data$MonthlyRate, bins=5, method="kmeans", ordered=FALSE)
hr_data$MonthlyRate <- as.numeric(as.factor(MonthlyRateCluster))
hr_data$MonthlyRate <- as.factor(hr_data$MonthlyRate)

# bin the DistanceFromHome variable using k-means
DistanceFromHomeCluster <- binning(hr_data$DistanceFromHome, bins=5, method="kmeans", ordered=FALSE)
hr_data$DistanceFromHome <- as.numeric(as.factor(DistanceFromHomeCluster))
hr_data$DistanceFromHome <- as.factor(hr_data$DistanceFromHome)

# bin the YearsAtCompany variable using k-means
YearsAtCompanyCluster <- binning(hr_data$YearsAtCompany, bins=5, method="kmeans", ordered=FALSE)
hr_data$YearsAtCompany <- as.numeric(as.factor(YearsAtCompanyCluster))
hr_data$YearsAtCompany <- as.factor(hr_data$YearsAtCompany)

# bin the YearsSinceLastPromotion variable using k-means
YearsSinceLastPromotionCluster <- binning(hr_data$YearsSinceLastPromotion, bins=5, method="kmeans", ordered=FALSE)
hr_data$YearsSinceLastPromotion <- as.numeric(as.factor(YearsSinceLastPromotionCluster))
hr_data$YearsSinceLastPromotion <- as.factor(hr_data$YearsSinceLastPromotion)

# bin the YearsWithCurrManager variable using k-means
YearsWithCurrManagerCluster <- binning(hr_data$YearsWithCurrManager, bins=5, method="kmeans", ordered=FALSE)
hr_data$YearsWithCurrManager <- as.numeric(as.factor(YearsWithCurrManagerCluster))
hr_data$YearsWithCurrManager <- as.factor(hr_data$YearsWithCurrManager)

# bin the TotalWorkingYears variable using k-means
TotalWorkingYearsCluster <- binning(hr_data$TotalWorkingYears, bins=5, method="kmeans", ordered=FALSE)
hr_data$TotalWorkingYears <- as.numeric(as.factor(TotalWorkingYearsCluster))
hr_data$TotalWorkingYears <- as.factor(hr_data$TotalWorkingYears)

write.csv(hr_data, file = "AssociationHREmployeeAttrition.csv", row.names = FALSE)

# use apriori algorithm for association rule mining
rules <- apriori(testing, parameter = list(minlen=2, supp=0.011, conf=0.8),appearance = list(rhs=c("Attrition=Yes"), default="lhs"), control = list(verbose=F))
rules.sorted <- sort(rules, by="lift")
inspect(rules.sorted)

# prune the association rules
subset.matrix <- is.subset(rules.sorted, rules.sorted)
#dgCMatrix is a standard numeric matrix (description), while ngTMatrix is a binary TRUE/FALSE, logical matrix (description)
subset.matrix <- subset.matrix * 1
subset.matrix[lower.tri(subset.matrix, diag=T)] <- NA
redundant <- colSums(subset.matrix, na.rm=T) >= 1
which(redundant)
rules.pruned <- rules.sorted[!redundant]
inspect(rules.pruned)

# plot the rules
plot(rules.pruned)
# plot the graph for the rules
plot(rules.pruned, method="graph", control=list(type="items"))
# plot the parallel coordinates
plot(rules.pruned, method="paracoord", control=list(reorder=TRUE))