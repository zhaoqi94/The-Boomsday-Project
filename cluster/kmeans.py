# k-means聚类
# 1.距离是可以一步算完的，但是需要把平方误差拆开，而且无法支持别的距离函数；也可以遍历K，针对每个中心点单独算
# 2.初始化方法：k-means++；随机
# 3.设置一个初始化尝试次数，多次初始化，取最好的结果

import numpy as np

class KMeans:
    def __init__(self, n_clusters=8, init='k-means++', n_init=10 ,max_iter=1000, tol=1e-4):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.labels = None
        self.cluster_centers = None

    def fit(self, X, y=None):
        best_score = np.inf
        best_labels = None
        best_cluster_centers = None
        for i in range(self.n_init):
            current_score = self._fit_single(X, y)
            # print(current_score)
            if current_score < best_score:
                best_score = current_score
                best_labels = self.labels
                best_cluster_centers = self.cluster_centers
        self.labels = best_labels
        self.cluster_centers = best_cluster_centers

        return self

    def predict(self, X):
        distances = self._compute_distance(X, self.cluster_centers)
        return np.argmin(distances, axis=1)

    def score(self, X):
        distances = self._compute_distance(X, self.cluster_centers)
        return np.sum(np.min(distances, axis=1) ** 2)

    def _init_cluster_centers(self, X):
        n_samples, n_features = X.shape
        if self.init == "random":
            self.cluster_centers = X[np.random.choice(n_samples, self.n_clusters, replace=False)]
        elif self.init == "k-means++":
            # https://www.cnblogs.com/wang2825/articles/8696830.html
            # step 1:第一个中心随机找
            cluster_centers_indice = [np.random.choice(n_samples,size=1)[0]]
            # step 2:计算每个样本和当前已有的聚类中心之间的最短距离D(x),
            # 则每个点被选择为下一个聚类中心的概率为D(x)^2/sum_x(D(x)^2)，接下来按这个概率抽样出下一个聚类中心
            for k in range(self.n_clusters-1):
                distances = self._compute_distance(X, X[cluster_centers_indice])
                D = np.min(distances, axis=1)
                prob = D**2 / np.sum(D**2)
                next = np.random.choice(n_samples,size=1,p=prob)[0]
                cluster_centers_indice.append(next)

            self.cluster_centers = X[cluster_centers_indice]

    def _compute_distance(self, X, cluster_centers):
        n_samples = X.shape[0]
        n_clusters = cluster_centers.shape[0]
        distances = np.zeros((n_samples, n_clusters))
        for k in range(n_clusters):
            distances[:, k] = np.linalg.norm(X - cluster_centers[k], axis=1)

        return distances


    def _cluster_centers_changes(self, old, new):
        # return np.sum(np.linalg.norm(old - new, axis=1))
        return np.sum(np.square(old-new))

    def _fit_single(self, X, y=None):
        # 使用原始样本进行初始化
        self._init_cluster_centers(X)
        # EM迭代
        for i in range(self.max_iter):
            # 计算距离&&划分
            distances = self._compute_distance(X, self.cluster_centers)
            labels = np.argmin(distances, axis=1)
            # 计算新的均值点（聚类中心）
            cluster_centers_old = self.cluster_centers.copy()
            for k in range(self.n_clusters):
                self.cluster_centers[k] = np.mean(X[labels==k], axis=0)
            # 聚类中心不再变化，就提前停止
            if self._cluster_centers_changes(cluster_centers_old, self.cluster_centers) <= self.tol:
                break
        # 针对新的聚类中心，在划分一次数据集
        distances = self._compute_distance(X, self.cluster_centers)
        self.labels = np.argmin(distances, axis=1)

        return self.score(X)