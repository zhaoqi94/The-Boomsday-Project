# TODO Linear Regression with ordinary least square
# 1.使用最简单的实现。 least square
# TODO 2.参考skelarn,对数据进行预处理，归一化以后，就不需要在X最后一列添加1了，最后计算出b即可

import numpy as np

class LinearRegression:

    def __init__(self):
        self.W = None
        self.b = None

    def fit(self, X, y):
        X_1 = X.copy()
        ones = np.ones(X_1.shape[0])[:, np.newaxis]
        X_1 = np.hstack([X_1, ones])
        W = np.linalg.inv(np.dot(X_1.T, X_1)).dot(X_1.T).dot(y)
        self.W = W[:-1]
        self.b = W[-1]

    def predict(self, X):
        return np.dot(X, self.W) + self.b

    def score(self, X, y):
        return 1 - np.sum(np.square(self.predict(X)-np.array(y))) / np.sum(np.square(np.mean(y)-np.array(y)))

