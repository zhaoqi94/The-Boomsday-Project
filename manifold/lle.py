# TODO 局部线性嵌入 LLE: Locally Linear Embedding

import numpy as np
from sklearn.neighbors import kneighbors_graph
from scipy.linalg import solve

class LocallyLinearEmbedding:
    def __init__(self, n_dims=2, n_neighbors=10):
        self.n_dims = n_dims
        self.n_neighbors = n_neighbors

    def fit(self, X, y=None):
        self.fit_transform(X)
        return self

    def fit_transform(self, X, y=None):
        n_samples, n_features = X.shape
        # 寻找k近邻
        kng = kneighbors_graph(X, self.n_neighbors)
        kng = kng.toarray().astype(np.bool)
        # # 针对每个样本，计算w_i，并得到整个W矩阵
        W = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            # 计算X[i]-（X[i]的近邻）的协方差矩阵
            reg = 1e-3
            z = X[i] - X[kng[i]]
            cov = np.dot(z, z.T)
            trace = np.trace(cov)
            # 非常关键！！！！！! cov不一定是可逆的啊！！！
            if trace > 0:
                R = reg * trace
            else:
                R = reg

            local_cov_inv = np.linalg.inv(cov+np.eye(self.n_neighbors)*(reg * trace))
            W[i, kng[i]] = np.sum(local_cov_inv, axis=1) / np.sum(local_cov_inv)

            # sklearn的做法，貌似也是原论文的做法,解线性方程组
            # cov = np.dot(z, z.T)
            # reg = 1e-3
            # trace = np.trace(cov)
            # if trace > 0:
            #     R = reg * trace
            # else:
            #     R = reg
            # cov.flat[::self.n_neighbors + 1] += R
            # w = solve(cov, np.ones(self.n_neighbors),sym_pos=True)
            # W[i, kng[i]] = w / w.sum()

        # 通过特征分解求解embedding优化问题
        I = np.eye(n_samples)
        M = np.dot((I - W).T, I - W)
        # eigh针对对称矩阵求特征值和特征向量
        # 并且特征值已经是排好序的了
        lamda, V = np.linalg.eigh(M)

        # 非常关键 好像就是要去掉最小的特征值
        # 如果是0:self.n_dims，就是一根线
        return V[:, 1:self.n_dims+1]




