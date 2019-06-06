#regression analysis on the DIABETES DATASET


from sklearn.datasets import load_diabetes
data = load_diabetes()

# Print a histogram of the quantity to predict: diabetes progression
import matplotlib.pyplot as plt
plt.figure(figsize=(4, 3))
plt.hist(data.target)
plt.xlabel('diabetes progression')
plt.ylabel('count')
plt.tight_layout()

# Print the join histogram for each feature

for index, feature_name in enumerate(data.feature_names):
    plt.figure(figsize=(4, 3))
    plt.scatter(data.data[:, index], data.target)
    plt.ylabel('diabetes progression', size=15)
    plt.xlabel(feature_name, size=15)
    plt.tight_layout()



# Simple prediction

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)

from sklearn.linear_model import LinearRegression
clf = LinearRegression()
clf.fit(X_train, y_train)
predicted = clf.predict(X_test)
expected = y_test

plt.figure(figsize=(4, 3))
plt.scatter(expected, predicted)
plt.plot([0, 50], [0, 50], '--k')
plt.axis('tight')
plt.xlabel('True diabetes progression')
plt.ylabel('Predicted diabetes progression')
plt.tight_layout()


# Prediction with gradient boosted tree

from sklearn.ensemble import GradientBoostingRegressor

clf = GradientBoostingRegressor()
clf.fit(X_train, y_train)

predicted = clf.predict(X_test)
expected = y_test

plt.figure(figsize=(4, 3))
plt.scatter(expected, predicted)
plt.plot([0, 50], [0, 50], '--k')
plt.axis('tight')
plt.xlabel('True diabetes progression')
plt.ylabel('Predicted diabetes progression')
plt.tight_layout()

# Print the error rate
import numpy as np
print("RMS: %r " % np.sqrt(np.mean((predicted - expected) ** 2)))

plt.show()


