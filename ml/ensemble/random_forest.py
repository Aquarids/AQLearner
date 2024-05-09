import numpy as np
from sklearn.tree import DecisionTreeClassifier
import learn_math


class RandomForest:

    def __init__(self, n_trees=100, max_depth=5, max_feature=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_feature = max_feature
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.n_trees):
            tree = DecisionTreeClassifier(max_depth=self.max_depth,
                                          max_features=self.max_feature)
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return [learn_math.most_common(row) for row in tree_preds.T]

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[idxs], y[idxs]

    def accuracy_score(self, y_test, y_pred):
        return np.sum(y_test == y_pred) / len(y_test)
