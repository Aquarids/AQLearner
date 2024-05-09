from sklearn.tree import DecisionTreeClassifier
import numpy as np


class AdaBoost:

    def __init__(self, n_learners=50):
        self.n_learners = n_learners
        self.learners = []
        self.learner_weights = []

    def fit(self, X, y):
        n_samples, _ = X.shape
        w = np.full(n_samples, (1 / n_samples))

        for _ in range(self.n_learners):
            learner = DecisionTreeClassifier(max_depth=1)
            learner.fit(X, y, sample_weight=w)
            learner_pred = learner.predict(X)

            miss = [int(x) for x in (learner_pred != y)]
            error = sum(w * miss) / sum(w)

            learner_weight = 0.5 * np.log((1 - error) / (error + 1e-10))

            w *= np.exp(-learner_weight * y * learner_pred)
            w /= np.sum(w)

            self.learners.append(learner)
            self.learner_weights.append(learner_weight)

    def predict(self, X):
        learner_preds = np.array(
            [learner.predict(X) for learner in self.learners])
        weighted_preds = np.dot(self.learner_weights, learner_preds)
        y_pred = np.sign(weighted_preds)
        return y_pred

    def accuracy_score(self, y_test, y_pred):
        return np.sum(y_test == y_pred) / len(y_test)
