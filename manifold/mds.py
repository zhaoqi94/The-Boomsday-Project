# TODO 多维缩放 MDS: Multiple Dimensional Scaling
# TODO 1.Classical multidimensional scaling 最经典的MDS算法
# 除此之外，还有mMDS,nMDS,GMD

import numpy as np

class MDS:
    def __init__(self, n_dims=2, dissimilarity="euclidean"):
        self.n_dims = n_dims
        self.dissimilarity = dissimilarity
        self.lamda = None
        self.v = None

    def fit(self, X, y=None):
        self.fit_transform(X)
        return self

    def fit_transform(self, X, y=None):
        n_samples, n_features = X.shape
        # 计算距离矩阵
        if self.dissimilarity == "euclidean":
            dist_matrix = np.empty((n_samples, n_samples))
            for i in range(n_samples):
                dist_matrix[i] = np.linalg.norm(X[i]-X, axis=1)
        elif self.dissimilarity == "precomputed":
            dist_matrix = X
        # 计算书中的B矩阵，即降维后的内积矩阵
        dist_sq = dist_matrix**2
        dist_i_sq = np.mean(dist_sq, axis=1, keepdims=True)
        dist_j_sq = np.mean(dist_sq, axis=0)
        dist_avg_sq = np.mean(dist_sq)
        B = -0.5*(dist_sq-dist_i_sq-dist_j_sq+dist_avg_sq)
        # 计算特征值，特征向量，并取最大的k个特征值对应的特征向量
        lamda, v = np.linalg.eig(B)
        indices = np.argsort(lamda)[::-1]
        self.lamda = lamda[indices[:self.n_dims]]
        self.v = v[:, indices[:self.n_dims]]

        return np.dot(self.v, np.diag(np.sqrt(self.lamda)))