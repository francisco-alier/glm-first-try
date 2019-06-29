## GLM TESTING 

##### Analyze and predict insurance cost based on a medical dataset #####
##### https://www.kaggle.com/mirichoi0218/insurance

# Load packages
library(xgboost)
library(tidyverse)
library(dplyr)
library(caret)
library(broom)
library(vtreat)

# Import data

df <- read.csv("./datasets/insurance.csv")

# get a glimpse of the dataset
glimpse(df)

#-------------------- Correct classes ----------------
df$children <- as.factor(df$children)

sapply(df, FUN = class)

#-------------------- Missing values ----------------
for (var in names(df)) {
  print(
    paste("The variable ", var, "has ", sum(is.na(df[, var])), "missing values")
  )
} 

str(df)
#-------------------- Visualization ----------------

par(mfrow=c(2,2))
categorical<- c("sex","children","smoker","region")

#Bar-Plot
for(i in categorical) {
  counts <- table(df[,i])
  barplot(counts, main=i)
}
#---------------------- Box-Plot ----------------

continous<- c("age","bmi","charges")
contseq<-which(colnames(df) %in% continous)
# Create separate boxplots for each attribute
par(mfrow=c(1,3))
for(i in contseq) {
  boxplot(df[,i], main=names(df)[i])
}

#---------------------- Histogram ------------------

par(mfrow=c(1,3))
for(i in contseq) {
  hist(df[,i], main=names(df)[i])
}

#-------------------- Density-plot -----------------

par(mfrow=c(1,3))
for(i in contseq) {
  plot(density(df[,i]), main=names(df)[i])
}

par(mfrow=c(1,1))# Stop de multiplot


#-------------------- View Data --------------------
View(df)

#-------------------- Split Data --------------------

#Split test/train
set.seed(123)
idx <- sample(seq(1, 2), size = nrow(df), replace = TRUE, prob = c(.7, .3))

train.data <- df[idx == 1,]
test.data  <- df[idx == 2,]

#Other split methods - k-way cross validation
splitPlan <- kWayCrossValidation(nrow(df), 3, NULL, NULL)



#-------------------- GLM --------------------

fmla <- charges ~ age + sex + bmi + children + smoker + region

glm1 <- glm(fmla, data = train.data, family = Gamma(link = "log")  
            )

summary(glm1)

# predict over test data
test.data$pred <- predict.glm(glm1, test.data, type = "response")
View(test.data)

# Calculate error
err <- test.data$pred - test.data$charges
# Square the error vector
err2 <- err^2
# Take the mean, and sqrt it
(rmse <- sqrt(mean(err2)))
# compare with SD
(sd <- sd(test.data$charges))

# Get RMSE
df %>% 
  gather(key = modeltype, value = pred, pred_add, pred_interaction) %>%
  mutate(residuals = Metabol - pred) %>%
  group_by(modeltype) %>%
  summarize(rmse = sqrt(mean(residuals^2)))

# Visualize predictions vs real
ggplot(test.data, aes(x = pred, y = charges)) +
  geom_point() +
  geom_abline(color = "darkblue") +
  ggtitle("GLM prediction vs true charges")

