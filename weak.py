from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import numpy as np

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
