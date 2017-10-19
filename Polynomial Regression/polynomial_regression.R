# Polynomial Regression
# Data Preprocessing Template

# Importing the dataset
dataset = read.csv('Position_salaries.csv')
dataset = dataset[,2:3]

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$DependentVariable, SplitRatio = 0.8)
# training_set = subset(dataset, split == TRUE)
# test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

#fitting linear regression to the dataset
lin_reg = lm(formula = Salary ~ Level, 
             data = dataset)

#fitting Polynomial Regression to the dataset
dataset$Level2 = dataset$Level^2
dataset$Level = dataset$Level^3
poly_reg = lm(formula = Salary ~ .,
              data = dataset)

#visulizing linear regression model
library(ggplot2)
ggplot() + 
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(lin_reg, newdata = dataset)),
            colour = 'blue') +
  ggtitle('Bluff or truth(Linear Model)') +
  xlab('Level') +
  ylab('Salary')

#visualizing polynomial regression model
ggplot() + 
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(poly_reg, newdata = dataset)),
            colour = 'blue') +
  ggtitle('Bluff or truth(Polynomial Model)') +
  xlab('Level') +
  ylab('Salary')