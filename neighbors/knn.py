# TODO KNN 算法
# TODO 1. Base KNN 作为分类器和回归器的基类，直接保存样本或使用KDTree BallTree
# TODO 2. KNN分类器 投票
# TODO 3. KNN回归 平均
# TODO 4. 不同的距离度量

import numpy as np
from neighbors.kdtree import KDTree

class BaseKNeighbors:
    def __init__(self, n_neighbors=5,
            algorithm='brute'):
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = X
        self.y = y
        if self.algorithm == "brute":
            return
        elif self.algorithm == "kd_tree":
            self.tree = KDTree(self.X)

    def predict(self, X):
        n_samples,n_features = X.shape
        y_pred = np.empty(n_samples)
        if self.algorithm == "brute":
            for i in range(n_samples):
                # 针对每一个测试样本计算它到训练集每个样本的距离
                distances = np.linalg.norm(self.X - X[i], axis=1)
                # 找到k个最近邻的样本
                top_k = np.argsort(distances)[0:self.n_neighbors]
                # print(np.sort(distances_i))
                # 使用这k个样本的y值进行预测
                # 分类：投票 回归：平均
                y_pred[i] = self._aggregate(self.y[top_k])
        elif self.algorithm == "kd_tree":
            for i in range(n_samples):
                top_k = self.tree.kd_nearest_n(X[i], self.n_neighbors)
                # print(top_k)
                y_pred[i] = self._aggregate(self.y[top_k])

        return y_pred

    def score(self, X, y):
        pass

    def _aggregate(self, y):
        pass


class KNeighborsClassifier(BaseKNeighbors):
    def __init__(self, n_neighbors=5,
            algorithm='brute'):
        super(KNeighborsClassifier, self).__init__(n_neighbors, algorithm)

    def score(self, X, y):
        return np.mean(self.predict(X)==y)

    def _aggregate(self, y):
        unique_item, unique_counts = np.unique(y, return_counts=True)
        return unique_item[np.argmax(unique_counts)]
        # 这真的是一个睿智错误啊！！！
        # return unique_item[np.argmax(unique_item)]



class KNeighborsRegressor(BaseKNeighbors):
    def __init__(self, n_neighbors=5,
            algorithm='brute'):
        super(KNeighborsRegressor, self).__init__(n_neighbors, algorithm)

    def score(self, X, y):
        return 1 - np.sum(np.square(self.predict(X)-np.array(y))) / np.sum(np.square(np.mean(y)-np.array(y)))

    def _aggregate(self, y):
        return np.mean(y)

