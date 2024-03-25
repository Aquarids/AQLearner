import numpy as np
import learn_math

class LogisticRegression():
    def __init__(self, alpha=0.01, n_iter=1000):
        self.alpha = alpha
        self.n_iter = n_iter
        self.w = None

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        for _ in range(self.n_iter):
            self.w -= self.alpha * self._gradient(X, y)

    def _gradient(self, X, y):
        y_pred = learn_math.sigmoid(X @ self.w)
        error = y_pred - y
        return X.T @ error / X.shape[0]

    def predict(self, X):
        y_pred = learn_math.sigmoid(X @ self.w)
        return [1 if i > 0.5 else 0 for i in y_pred]
    
    def accuracy_score(self, y_test, y_pred):
        return np.sum(y_test == y_pred) / len(y_test)