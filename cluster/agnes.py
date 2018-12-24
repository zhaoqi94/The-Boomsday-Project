# AGNES:agglomerative hierarchical clustering
# 层次聚类：试图在不同的“层次”上对样本数据集进行划分，一层一层地进行聚类
# 主要方法分为：自底向上的凝聚方法AGNES 和 自上向下的分裂方法DIANA

import numpy as np

class AGNES:
    def __init__(self, n_clusters=2, metric="euclidean", linkage="average"):
        self.n_clusters = n_clusters
        self.metric = metric
        self.linkage = linkage

    def fit(self, X, y=None):
        n_samples,n_features = X.shape
        clusters = [{i} for i in range(n_samples)]
        cluster_distances = self._total_cluster_distances(X, clusters)

        # 循环：找到最近的两个簇合并，重新计算距离 直到合并要求的簇个数为止
        while(len(clusters) > self.n_clusters):
            # 找到最近的两个簇
            i, j = self._get_closest_cluster_pair(cluster_distances)
            # 合并簇
            clusters[i].update(clusters[j])
            # 删除簇C_j
            clusters.pop(j)
            # 更新距离 删除第j行和列，以及更新第i行和第i列
            cluster_distances = self._update_cluster_distances(X, clusters, cluster_distances, i, j)

        # 标签设置一下
        self.labels = np.empty(n_samples, dtype=np.int32)
        for label in range(self.n_clusters):
            self.labels[list(clusters[label])] = label

        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.labels

    # 计算两个簇之间的距离
    # 首先根据metric度量算出Ci中每个点到Cj中每个点之间的距离
    # 然后根据不同的linkage计算出两个簇的距离
    def _cluster_distance(self, X, Ci, Cj):
        n_samples_Ci = len(Ci)
        n_samples_Cj = len(Cj)
        distances = np.empty((n_samples_Ci, n_samples_Cj))
        for i,x in enumerate(Ci):
            for j,y in enumerate(Cj):
                distances[i,j] = np.linalg.norm(X[x] - X[y])
        if self.linkage == "average":
            return np.mean(distances)
        elif self.linkage == "max":
            return np.max(distances)
        elif self.linkage == "min":
            return np.min(distances)

    def _total_cluster_distances(self, X, clusters):
        n_clusters = len(clusters)
        cluster_distances = np.empty((n_clusters, n_clusters))
        for i in range(n_clusters):
            for j in range(i, n_clusters):
                if i == j:
                    cluster_distances[i, j] = np.inf    # 同一个簇不能和自己合并，所以把和自己的距离设置为无穷大
                else:
                    cluster_distances[i, j] = self._cluster_distance(X, clusters[i], clusters[j])
                    cluster_distances[j, i] = np.inf # 这是为了后面计算出i,j时 i一定小于j
        return cluster_distances

    def _update_cluster_distances(self, X, clusters, cluster_distances, i, j):
        # 删除j
        cluster_distances = np.delete(np.delete(cluster_distances, j, axis=0), j, axis=1)
        # 更新i
        n_clusters = len(clusters)
        for k in range(n_clusters):
            if k < i:
                cluster_distances[k, i] = self._cluster_distance(X, clusters[k], clusters[i])
            elif k > i:
                cluster_distances[i, k] = self._cluster_distance(X, clusters[i], clusters[k])

        return cluster_distances

    def _get_closest_cluster_pair(self, cluster_distances):
        n_clusters = cluster_distances.shape[0]
        min_i_j = np.argmin(cluster_distances)
        i, j = min_i_j // n_clusters, min_i_j % n_clusters

        return i, j