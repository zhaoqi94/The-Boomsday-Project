# TODO 学习向量量化 Learning Vector Quantization
# 1.LVQ1.0
# 这究竟是聚类还是分类 感觉更像是有监督的分类
# 严格来说应该不属于聚类
import numpy as np

class LVQ:
    def __init__(self, max_iter=1000, eta=0.01, tol=1e-4):
        self.max_iter = max_iter
        self.eta = eta
        self.tol = tol
        self.labels = None
        self.cluster_centers = None
        self.cluster_centers_labels = None

    def fit(self, X, y):
        # 1.初始化聚类中心
        n_samples, n_features = X.shape
        self.cluster_centers_labels = np.unique(y)
        self.cluster_centers = np.zeros((self.cluster_centers_labels.shape[0], n_features))
        for i, label_i in enumerate(self.cluster_centers_labels):
            sub_X = X[y==label_i]
            choice = np.random.randint(0, sub_X.shape[0])
            self.cluster_centers[i] = X[choice]
        # 2.聚类
        for iter in range(self.max_iter):
            cluster_centers_old = self.cluster_centers.copy()
            for i in range(n_samples):
                # 计算样本与聚类中心的距离
                distance = self._distance(X[i], self.cluster_centers)
                closest = np.argmin(distance)
                if y[i] == self.cluster_centers_labels[closest]:
                    self.cluster_centers[closest] = self.cluster_centers[closest] + self.eta * (
                            X[i] - self.cluster_centers[closest])
                else:
                    self.cluster_centers[closest] = self.cluster_centers[closest] - self.eta * (
                                X[i] - self.cluster_centers[closest])
            if self._cluster_centers_changes(cluster_centers_old, self.cluster_centers) <= self.tol:
                break
        # 3.根据获得的中心，计算样本的标签（本来就有标签的，不一定要算）
        distances = self._compute_distance(X, self.cluster_centers)
        self.labels = self.cluster_centers_labels[np.argmin(distances, axis=1)]

    # X1和X2其中一个是矩阵，另一个是向量
    def _distance(self, X1, X2):
        return np.linalg.norm(X1 - X2, axis=1)

    def _cluster_centers_changes(self, old, new):
        # return np.sum(np.linalg.norm(old - new, axis=1))
        return np.sum(np.square(old-new))

    def _compute_distance(self, X, cluster_centers):
        n_samples = X.shape[0]
        n_clusters = cluster_centers.shape[0]
        distances = np.zeros((n_samples, n_clusters))
        for k in range(n_clusters):
            distances[:, k] = np.linalg.norm(X - cluster_centers[k], axis=1)

        return distances

