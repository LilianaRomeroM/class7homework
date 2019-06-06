import pandas as pd
from sklearn.datasets import load_diabetes

diabetes = load_diabetes()
columns_names = diabetes.feature_names
y = diabetes.target
X = diabetes.data
print(columns_names)

#a) For a new patient diagnosed with diabetes, could we use this information to determine one or more
#key variables to control in order to mitigate its progression?
#*Which variables have the greatest effect on the diabetes progression? In technical terms, the idea is to
#analyze whether there is a significant incidence of any factor on progression, in order to focus treatment
#efforts to prevent major complications.

# Splitting features and target datasets into: train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)

# Training a Linear Regression model with fit()
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)


# Output of the training is a model: a + b*X0 + c*X1 + d*X2 ...
print(f"Intercept: {lm.intercept_}\n")
print(f"Coeficients: {lm.coef_}\n")
print(f"Named Coeficients: {pd.DataFrame(lm.coef_, columns_names)}")
