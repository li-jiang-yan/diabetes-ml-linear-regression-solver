from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn import linear_model

# Load diabetes dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# Split data points for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Create linear regressor
reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)
m = reg.coef_
c = reg.intercept_
