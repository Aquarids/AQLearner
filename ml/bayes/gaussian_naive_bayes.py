import numpy as np
import learn_math


class GaussianNaiveBayesClassifier:

    def __init__(self):
        self.means = {}
        self.variances = {}
        self.priors = {}

    def fit(self, X, y):
        self.classes = np.unique(y)
        for cls in self.classes:
            X_cls = X[y == cls]
            self.means[cls] = X_cls.mean(axis=0)
            self.variances[cls] = X_cls.var(axis=0)
            self.priors[cls] = X_cls.shape[0] / X.shape[0]

    def predict(self, X):
        predictions = []
        for x in X:
            posteriors = []
            for cls in self.classes:
                prior = np.log(self.priors[cls])
                conditional = np.sum(
                    np.log(
                        learn_math.gaussian_pdf(x, self.means[cls],
                                                self.variances[cls])))
                posterior = prior + conditional
                posteriors.append(posterior)
            predictions.append(self.classes[np.argmax(posteriors)])
        return predictions

    def accuracy_score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
