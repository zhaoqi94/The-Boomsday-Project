# TODO Gaussian Mixture 高斯混合模型
# TODO 1.EM算法简单的实现
# TODO 2.利用Cholesky分解提高效率

import numpy as np
from cluster.kmeans import KMeans

class GaussianMixture:
    def __init__(self, n_components=1,tol=1e-8, max_iter=100,
                 n_init=1, init_params='kmeans'):
        self.n_components = n_components
        self.tol = tol
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.weights = None
        self.means = None
        self.covariances = None
        self.labels = None
        self.joint_log_likelihood = None

    def fit(self, X, y=None):
        # 1.初始化参数
        self._initilize_parameters(X)
        # 2.EM迭代 E步:计算统计量的期望 M:最大化以计算参数
        for i in range(self.max_iter):
            old_jll = self.joint_log_likelihood
            # E step
            resp, self.joint_log_likelihood = self._e_step(X)
            # M step
            self._m_step(X, resp)
            if np.abs(old_jll - self.joint_log_likelihood) <= self.tol:
                print(i)
                break

        # 3.使用得到的参数计算标签
        resp, _ = self._e_step(X)
        print(resp[0:20])
        self.labels = np.argmax(resp, axis=1)

    def predict(self, X):
        pass

    def score(self, X):
        pass

    def _initilize_parameters(self, X):
        # resp指的是后验分布
        n_samples,_ = X.shape
        if self.init_params == "kmeans":
            resp = np.zeros((n_samples, self.n_components))
            label = KMeans(n_clusters=self.n_components).fit(X).labels
            resp[np.arange(n_samples), label] = 1
        elif self.init_params == 'random':
            resp = np.random.rand(n_samples, self.n_components)
            resp /= resp.sum(axis=1)[:, np.newaxis]
        else:
            raise ValueError("Unimplemented initialization method '%s'"
                             % self.init_params)

        self.weights, self.means, self.covariances = self._estimate_gaussian_parameters(X, resp)

        self.joint_log_likelihood = -np.inf


    def _estimate_gaussian_parameters(self, X, resp):
        # resp：后验推断，即EM算法中所谓的“猜测”
        # 利用计算好的resp估计参数
        n_samples, n_features = X.shape
        nk = np.sum(resp, axis=0)
        # 混合系数 alpha
        weights = nk / n_samples
        # 均值
        means = np.dot(resp.T, X) / nk[:, np.newaxis]
        # 协方差
        covariances = np.empty((self.n_components, n_features, n_features)) # 注意np.empty和np.zeros的区别：np.empty不会进行初始化，不清理内存
        for k in range(self.n_components):
            diff = X - means[k]
            covariances[k] = np.dot(resp[:, k] * diff.T, diff) / nk[k]

        return weights, means, covariances

    def _e_step(self, X):
        n_samples, n_features = X.shape
        resp = np.empty((n_samples, self.n_components))
        # 计算后验分布（似然?）
        cov_det = np.linalg.det(self.covariances)   # 协方差矩阵的行列式
        cov_inv = np.linalg.inv(self.covariances)   # 协方差矩阵的逆矩阵
        for k in range(self.n_components):
            diff = X - self.means[k]
            exp_term = np.exp(-0.5 * np.sum((np.dot(diff, cov_inv[k]) * diff), axis=1))
            # exp_term = np.exp(-0.5 * diff.dot(cov_inv[k]).dot(diff))
            const_term = 1.0 / (np.power(2*np.pi, n_features/2) * np.power(cov_det[k],0.5))
            resp[:, k] = self.weights[k] * const_term * exp_term

        joint_log_likelihood = np.mean(np.log(np.sum(resp, axis=1)))
        resp = resp / np.sum(resp, axis=1)[:, np.newaxis]

        return resp, joint_log_likelihood

    def _m_step(self, X, resp):
        self.weights, self.means, self.covariances = self._estimate_gaussian_parameters(X, resp)