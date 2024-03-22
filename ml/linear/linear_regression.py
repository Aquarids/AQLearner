import numpy as np
import learn_math

class LinearRegression:
    def __init__(self) -> None:
        self.w = None
        self.b = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.w = learn_math.normal_eq(X, y)
        self.b = np.mean(y - X @ self.w)

    def predict(self, X):
        return X @ self.w + self.b
    
    def r_squared_score(self, y_test, y_pred):
        return learn_math.r_squared(y_test, y_pred)
    
