import numpy as np

# Normal equation
def normal_eq(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y

# R-squared
def r_squared(y, y_pred):
    return 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Binary cross-entropy loss
def binary_crossentropy(m, y, y_pred):
    return -1/m * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
