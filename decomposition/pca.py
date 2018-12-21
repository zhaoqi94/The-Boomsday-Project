# TODO 主成分分析PCA(Principle Component Analysis)
# TODO 1. SVD分解

import numpy as np
from six.moves import xrange

class PCA:
    def __init__(self, n_dims, copy=True):
        self.n_dims = n_dims
        self.copy = copy
        self.W = None

    def fit(self, X, y=None):
        self._fit(X)
        return self

    def transform(self, X):
        return np.dot(X, self.W)

    def fit_transform(self, X, y=None):
        U,S,V = self._fit(X)
        U = U[:, :self.n_dims]

        # X_new = X * V = U * S * V^T * V = U * S
        U *= S[:self.n_dims]

        return U

    def _fit(self, X):
        # 一般情况复制一下数据
        if self.copy:
            X = X.copy()

        # PCA之前一定要中心化啊啊啊！！！
        self.means = np.mean(X, axis=0)
        X -= self.means

        U, S, V = np.linalg.svd(X, full_matrices=False)
        U, V = self._svd_flip(U, V)
        self.W = V[:self.n_dims].T

        return U, S, V

    '''
    抄一哈sklearn的实现
    SVD分解得到的U和V可能和真正的特征向量差一个正负符号！！！
    其实也没有所谓的“真正的特征向量”，特征向量的线性组合仍然是特征向量
    这个方法其实是为了保证在数据集打乱后PCA得到的结果和打乱前是一样的
    '''
    def _svd_flip(self, u, v, u_based_decision=True):
        """Sign correction to ensure deterministic output from SVD.

        Adjusts the columns of u and the rows of v such that the loadings in the
        columns in u that are largest in absolute value are always positive.

        Parameters
        ----------
        u, v : ndarray
            u and v are the output of `linalg.svd` or
            `sklearn.utils.extmath.randomized_svd`, with matching inner dimensions
            so one can compute `np.dot(u * s, v)`.

        u_based_decision : boolean, (default=True)
            If True, use the columns of u as the basis for sign flipping.
            Otherwise, use the rows of v. The choice of which variable to base the
            decision on is generally algorithm dependent.


        Returns
        -------
        u_adjusted, v_adjusted : arrays with the same dimensions as the input.

        """
        if u_based_decision:
            # 找到每一列绝对值最大的那个数，让它的符号为正
            # columns of u, rows of v
            max_abs_cols = np.argmax(np.abs(u), axis=0)
            signs = np.sign(u[max_abs_cols, xrange(u.shape[1])])
            u *= signs
            v *= signs[:, np.newaxis]
        else:
            # rows of v, columns of u
            max_abs_rows = np.argmax(np.abs(v), axis=1)
            signs = np.sign(v[xrange(v.shape[0]), max_abs_rows])
            u *= signs
            v *= signs[:, np.newaxis]
        return u, v
