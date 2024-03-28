import numpy as np

# Normal equation
def normal_eq(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y

# R-squared
def r_squared(y, y_pred):
    # Coefficient of determination
    ss_res = np.sum((y - y_pred) ** 2)
    # Total sum of squares
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot == 0:
        return float('nan')
    return 1 - ss_res / ss_tot

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Euclidean distance
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# Binary cross-entropy loss
def binary_crossentropy(m, y, y_pred):
    return -1/m * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

# Entropy
def caculate_entropy(y):
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return -np.sum(p * np.log2(p))

# Select the most common element
def most_common(y):
    return np.bincount(y).argmax()

# Gini impurity
def gini(y):
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return 1 - np.sum(p ** 2)

# Minimum geometric margin
def geometric_margin(X, y, w, b):
    return np.min(y * (X @ w + b))

# Hinge loss
def hinge_loss(m, y, y_pred):
    return 1/m * np.sum(np.maximum(0, 1 - y * y_pred))

# Gaussian probability density
def gaussian_pdf(x, mean, var):
    exponent = np.exp(-((x - mean) ** 2) / (2 * var))
    return (1 / np.sqrt(2 * np.pi * var)) * exponent