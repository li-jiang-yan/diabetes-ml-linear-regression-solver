from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import linear_model
import math
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Load diabetes dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# Split data points for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Compute the correlation matrix
# It looks like only the variables with indexes 0, 2, 3, 4, 6, 7, 8, 9 have a correlation coefficient greater than 0.2 (weak correlation)
corrcoef = np.corrcoef(X_train, y=y_train, rowvar=False)
print(np.argwhere(np.abs(corrcoef[-1,:-1])>0.2)) # [[0], [2], [3], [4], [6], [7], [8], [9]]

# Redefine features to be considered
X = X[:, np.array([0, 2, 3, 4, 6, 7, 8, 9])]

# Split data points for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Create linear regressor
reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)
coef_ = reg.coef_
c = reg.intercept_

# Plot linear regression graphs
feature_names = [diabetes.feature_names[i] for i in [0, 2, 3, 4, 6, 7, 8, 9]]
nsubplots = len(feature_names)
ncols = 4
nrows = math.ceil(nsubplots / ncols)
fig = plt.figure()
for feature_num in range(nsubplots):
    row_num = feature_num // ncols
    col_num = feature_num % ncols
    feature_name = feature_names[feature_num]
    ax = plt.subplot2grid((nrows, ncols), (row_num, col_num))
    x_train = X_train[:, feature_num]
    x_test = X_test[:, feature_num]
    m = coef_[feature_num]
    ax.plot(x_train, y_train, "bo", label="training data")
    ax.plot(x_test, y_test, "go", label="testing data")
    ax.plot(x_train, m * x_train + c, "r", label="model")
    ax.set_xlabel(feature_name)
    ax.set_ylabel("target")
    ax.legend()

fig.suptitle("Diabetes bivariable linear regression, R2 = {:.2f}".format(r2_score(y_test, reg.predict(X_test))))

plt.show()
