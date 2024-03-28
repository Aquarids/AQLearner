import numpy as np
import learn_math

class KNN():
    def __init__(self, k=3):
        self.k = k
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    def _predict(self, x):
        distances = [learn_math.euclidean_distance(x, x_train) for x_train in self.X]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y[i] for i in k_indices]
        return learn_math.most_common(k_nearest_labels)
    
    def accuracy_score(self, y_test, y_pred):
        return np.sum(y_test == y_pred) / len(y_test)