import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes

diabetes = load_diabetes()
columns_names = diabetes.feature_names
y = diabetes.target
X = diabetes.data


# Training a Linear Regression model with fit()
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso

lasso= Lasso (alpha=0.2, normalize=True)
lasso_coef= lasso.fit(X,y).coef_
print(lasso_coef)
_ = plt.plot(range(len(columns_names)), lasso_coef)
_ = plt.xticks(range(len(columns_names)), columns_names, rotation=45)
_ = plt.ylabel('coeficients')
plt.margins(0.02)
plt.show()

lasso= Lasso (alpha=0.5, normalize=True)
lasso_coef= lasso.fit(X,y).coef_
print(lasso_coef)
_ = plt.plot(range(len(columns_names)), lasso_coef)
_ = plt.xticks(range(len(columns_names)), columns_names, rotation=45)
_ = plt.ylabel('coeficients')
plt.margins(0.02)
plt.show()

lasso= Lasso (alpha=0.8, normalize=True)
lasso_coef= lasso.fit(X,y).coef_
print(lasso_coef)
_ = plt.plot(range(len(columns_names)), lasso_coef)
_ = plt.xticks(range(len(columns_names)), columns_names, rotation=45)
_ = plt.ylabel('coeficients')
plt.margins(0.02)
plt.show()



