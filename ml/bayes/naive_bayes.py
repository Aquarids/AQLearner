import numpy as np

class NaiveBayesClassifier:
    def __init__(self):
        self.class_prior_probabilities = {}
        self.conditional_probabilities = {}
    
    def fit(self, X, y):
        self.classes = np.unique(y)
        sample_count = X.shape[0]
        
        for cls in self.classes:
            X_cls = X[y == cls]
            self.class_prior_probabilities[cls] = X_cls.shape[0] / sample_count
            self.conditional_probabilities[cls] = {}
            for j in range(X.shape[1]):
                feature_values = np.unique(X[:, j])
                probabilities = {}
                for value in feature_values:
                    probability = ((X_cls[:, j] == value).sum() + 1) / (X_cls.shape[0] + len(feature_values))
                    probabilities[value] = probability
                self.conditional_probabilities[cls][j] = probabilities
    
    def predict(self, X):
        predictions = []
        for x in X:
            class_probabilities = {}
            for cls in self.classes:
                total_probability = np.log(self.class_prior_probabilities[cls])
                for j, value in enumerate(x):
                    probabilities = self.conditional_probabilities[cls][j]
                    if value in probabilities:
                        total_probability += np.log(probabilities[value])
                    else:
                        total_probability += np.log(1 / (sum(x == x[j]) + len(np.unique(x))))
                class_probabilities[cls] = total_probability
            predicted_class = max(class_probabilities, key=class_probabilities.get)
            predictions.append(predicted_class)
        return predictions
    
    def accuracy_score(self, y_test, y_pred):
        return np.sum(y_test == y_pred) / len(y_test)
