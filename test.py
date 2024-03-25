import unittest
import numpy as np

from ml.linear.linear_regression import LinearRegression
from ml.linear.logistic_regression import LogisticRegression

class TestLinearRegression(unittest.TestCase):
    def test_linear_regression(self):
        X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
        y = np.array([2, 4, 6, 8, 10])
        
        model = LinearRegression()
        model.fit(X, y)

        test_X = np.array([6, 7, 8, 9, 10]).reshape(-1, 1)
        test_y = np.array([12, 14, 16, 18, 20])
        y_pred = model.predict(test_X)
        self.assertEqual(y_pred.tolist(), test_y.tolist())

        print('R-squared:', model.r_squared_score(test_y, y_pred))

class TestLogisticRegression(unittest.TestCase):
    def test_logistic_regression(self):
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        y = np.array([0, 0, 1, 1, 1])
        
        model = LogisticRegression()
        model.fit(X, y)

        test_X = np.array([[6, 7], [7, 8], [8, 9], [9, 10], [10, 11]])
        test_y = np.array([0, 0, 1, 1, 1])
        y_pred = model.predict(test_X)
        print('Accuracy:', model.accuracy_score(test_y, y_pred))

if __name__ == '__main__':
    unittest.main()