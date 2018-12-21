# TODO 核主成分分析 Kernel PCA 非线性降维
# 1. 初步搞定KPCA 注意点：1)特征向量缩放以使参数W是正交矩阵 2)核中心化，使转换后的特征隐式地中心化了
# 2. TODO 自己写KernelCenterer核中心化 放在utils.preprocessing里面

import svm.kernels as kernels
import scipy
import numpy as np
from sklearn.preprocessing.data import KernelCenterer

class KernelPCA:
    def __init__(self, n_dims, kernel="linear", copy=True):
        self.n_dims = n_dims
        self.copy = copy
        self.kernel = kernels.KERNEL_TYPES[kernel]   # 将字符串转化为相应的核函数
        self.X = None   # 样本要保存 因为这是核方法
        self.alphas = None
        self.lambdas = None
        self.centerer = KernelCenterer()

    def fit(self, X, y=None):
        self._fit(X)
        return self

    def transform(self, X):
        K = self.kernel(X, self.X)
        K = self.centerer.transform(K)
        return np.dot(K, self.alphas) / np.sqrt(self.lambdas)

    def fit_transform(self, X, y=None):
        self._fit(X)
        return self.alphas * np.sqrt(self.lambdas)

    def _fit(self, X):
        self.X = X
        # 计算核矩阵
        K = self.kernel(X, X)
        # 中心化核矩阵 目的是为了隐式地让转换后的特征是归一化的！
        K = self.centerer.fit_transform(K)
        # 求核矩阵的特征值,特征向量
        self.lambdas, self.alphas = scipy.linalg.eigh(K, eigvals=(K.shape[0] - self.n_dims, K.shape[0] - 1))
        # 对特征值，特征向量进行处理
        indices = self.lambdas.argsort()[::-1]
        self.lambdas = self.lambdas[indices]
        self.alphas = self.alphas[:, indices]
        # 去除特征值等于0的特征向量
        self.alphas = self.alphas[:, self.lambdas > 0]
        self.lambdas = self.lambdas[self.lambdas > 0]

