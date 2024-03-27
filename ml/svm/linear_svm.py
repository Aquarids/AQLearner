import numpy as np
import learn_math

class LinearSVM:
    def __init__(self, learning_rate=0.01, lambda_param=0.01, n_iter=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iter = n_iter
        self.w = None
        self.b = None

    def fit(self, X, y):
        y = np.where(y <= 0, -1, 1)
        self.w = np.zeros(X.shape[1])
        self.b = 0
        for _ in range(self.n_iter):
            for i, x in enumerate(X):
                if y[i] * (x @ self.w + self.b) >= 1:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w - y[i] * x)
                    self.b -= self.learning_rate * y[i]

    def predict(self, X):
        return [1 if x @ self.w + self.b > 0 else -1 for x in X]
    
    def accuracy_score(self, y_test, y_pred):
        return np.sum(y_test == y_pred) / len(y_test)