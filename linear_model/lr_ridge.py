# TODO Ridge Regression
# 1.least square
# TODO 2.SVD
# TODO 3.cholesky
# TODO 4.lsqr

import numpy as np

class RidgeRegression:
    def __init__(self, alpha=1.0):
        self.W = None
        self.b = None
        self.alpha = alpha

    def fit(self, X, y):
        sample_num, feature_num = X.shape
        X_1 = X.copy()
        ones = np.ones(sample_num)[:, np.newaxis]
        X_1 = np.hstack([X_1, ones])
        W = np.linalg.inv(np.dot(X_1.T, X_1)+self.alpha*np.eye(feature_num+1)).dot(X_1.T).dot(y)
        self.W = W[:-1]
        self.b = W[-1]

    def predict(self, X):
        return np.dot(X, self.W) + self.b

    def score(self, X, y):
        return 1 - np.sum(np.square(self.predict(X)-np.array(y))) / np.sum(np.square(np.mean(y)-np.array(y)))

