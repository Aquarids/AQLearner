import numpy as np

class LinearRegression:
    def __init__(self) -> None:
        self.w = None
        self.b = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.w = np.linalg.inv(X.T @ X) @ X.T @ y
        self.b = np.mean(y - X @ self.w)

    def predict(self, X):
        return X @ self.w + self.b
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
    
