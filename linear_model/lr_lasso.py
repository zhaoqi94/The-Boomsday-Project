# TODO Lasso Regression
# 1.coordinate descent
# TODO 2.Least Angle Regression

import numpy as np


"""
    sklearn里Lasso的基类ElasticNet使用的训练方法注释
    可以看出它的目标函数其实不太一样！！！
    注意什么是MSE啊啊啊啊！
    这样就解释了self.alpha*X.shape[0]这一项了！！！
    否则效果会很差
    The elastic net optimization function varies for mono and multi-outputs.

    For mono-output tasks it is::

        1 / (2 * n_samples) * ||y - Xw||^2_2
        + alpha * l1_ratio * ||w||_1
        + 0.5 * alpha * (1 - l1_ratio) * ||w||^2_2

    For multi-output tasks it is::

        (1 / (2 * n_samples)) * ||Y - XW||^Fro_2
        + alpha * l1_ratio * ||W||_21
        + 0.5 * alpha * (1 - l1_ratio) * ||W||_Fro^2
"""

class LassoRegression:
    def __init__(self, alpha=1.0, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.W = None
        self.b = None

    # 注意在lasso中bias项是不需要正则化的
    def fit(self, X, y):
        W = np.zeros(X.shape[1])
        b = np.sum(y - np.dot(X, W)) / X.shape[0]

        for i in range(self.max_iter):
            for j in range(X.shape[1]):
                W[j] = 0
                rho_j = np.dot(X[:,j], y - np.dot(X, W) - b)
                z_j = np.sum(X[:, j]**2)
                W[j] = self._soft_thresholding(rho_j, z_j, self.alpha*X.shape[0])
            b = np.sum(y - np.dot(X, W)) / X.shape[0]

        self.W = W
        self.b = b

    def _soft_thresholding(self, rho, z, threshold):
        if rho < -threshold:
            return (rho + threshold) / z
        elif rho > threshold:
            return (rho - threshold) / z
        else:
            return 0.0

    def predict(self, X):
        return np.dot(X, self.W) + self.b

    def score(self, X, y):
        return 1 - np.sum(np.square(self.predict(X)-np.array(y))) / np.sum(np.square(np.mean(y)-np.array(y)))

