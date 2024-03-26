import numpy as np
import learn_math

class C45():
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y):
        if len(np.unique(y)) == 1:
            return {'is_leaf': True, 'label': y[0]}
        if X.shape[1] == 0:
            return {'is_leaf': True, 'label': learn_math.most_common(y)}
        best_feature, best_threshold = self._best_criteria(X, y)
        left_idxs = X[:, best_feature] < best_threshold
        right_idxs = X[:, best_feature] >= best_threshold
        if len(X[left_idxs]) == 0 or len(X[right_idxs]) == 0:
            return {'is_leaf': True, 'label': learn_math.most_common(y)}
        
        left = self._build_tree(X[left_idxs], y[left_idxs])
        right = self._build_tree(X[right_idxs], y[right_idxs])
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
                gain = self._information_gain_ratio(X, y, feature, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold
    
    def _information_gain(self, X, y, feature, threshold):
        left_idxs = X[:, feature] < threshold
        right_idxs = X[:, feature] >= threshold
        n = len(y)
        n_left = np.sum(left_idxs)
        n_right = np.sum(right_idxs)

        total_entropy = learn_math.caculate_entropy(y)
        left_entropy = learn_math.caculate_entropy(y[left_idxs])
        right_entropy = learn_math.caculate_entropy(y[right_idxs])

        return total_entropy - (n_left / n) * left_entropy - (n_right / n) * right_entropy

    def _intrinsic_value(self, X, y, feature, threshold):
        left_idxs = X[:, feature] < threshold
        right_idxs = X[:, feature] >= threshold
        n = len(y)
        n_left = np.sum(left_idxs)
        n_right = np.sum(right_idxs)

        left_entropy = learn_math.caculate_entropy(y[left_idxs])
        right_entropy = learn_math.caculate_entropy(y[right_idxs])

        return (n_left / n) * left_entropy + (n_right / n) * right_entropy

    
    def _information_gain_ratio(self, X, y, feature, threshold):
    
        information_gain = self._information_gain(X, y, feature, threshold)
        intrinsic_value = self._intrinsic_value(X, y, feature, threshold)

        if intrinsic_value == 0:
            return 0
        
        return information_gain / intrinsic_value

    def predict(self, X):
        return np.array([self._predict_single_input(x, self.tree) for x in X])

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