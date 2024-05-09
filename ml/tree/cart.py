import numpy as np
import learn_math


class CART():

    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):

        if len(np.unique(y)) == 1 or depth >= self.max_depth:
            return {'is_leaf': True, 'label': learn_math.most_common(y)}

        best_feature, best_threshold = self._best_criteria(X, y)
        left_idxs, right_idxs = self._split(X[:, best_feature], best_threshold)
        left = self._build_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._build_tree(X[right_idxs], y[right_idxs], depth + 1)
        tree = {
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left,
            'right': right,
            'is_leaf': False
        }
        return tree

    def _best_criteria(self, X, y):
        best_gain = -1
        best_feature = None
        best_threshold = None
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gain = self._gini_gain(X, y, feature, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold

    def _gini_gain(self, X, y, feature, threshold):
        parent_impurity = learn_math.gini(y)
        left_idxs, right_idxs = self._split(X[:, feature], threshold)
        n = len(y)
        n_left, n_right = len(left_idxs), len(right_idxs)
        if n_left == 0 or n_right == 0:
            return 0
        left_impurity = learn_math.gini(y[left_idxs])
        right_impurity = learn_math.gini(y[right_idxs])
        weighted_impurity = (n_left / n) * left_impurity + (n_right /
                                                            n) * right_impurity
        gain = parent_impurity - weighted_impurity
        return gain

    def _split(self, feature_values, threshold):
        left_idxs = np.where(feature_values < threshold)[0]
        right_idxs = np.where(feature_values >= threshold)[0]
        return left_idxs, right_idxs

    def predict(self, X):
        predicted_labels = [
            self._predict_single_input(x, self.tree) for x in X
        ]
        return np.array(predicted_labels)

    def _predict_single_input(self, x, tree):
        if tree['is_leaf']:
            return tree['label']
        feature = tree['feature']
        threshold = tree['threshold']
        if x[feature] < threshold:
            return self._predict_single_input(x, tree['left'])
        else:
            return self._predict_single_input(x, tree['right'])

    def accuracy_score(self, y_test, y_pred):
        return np.sum(y_test == y_pred) / len(y_test)

    def get_tree(self):
        return self.tree
