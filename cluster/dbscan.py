# DBSCAN(Density-Based Spatial Clustering of Applications with Noise)
# DBSCAN是基于密度的聚类的一种，相比于K-means，DBSCAN可以聚类非凸的样本
# DBSCAN聚出来的类的数量不是手动定的，而是算法自动算出来的
# DBSCAN第一步是计算

import numpy as np
import random

class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps  # 邻域半径
        self.min_samples = min_samples  # 定义核心对象所需的领域半径内最少的点数
        self.labels = None  # 样本标签
        self.core_sample_indices = None # 核心样本数量


    def fit(self, X, y=None):
        n_samples, n_features = X.shape
        eps_neignbors = []
        indices = np.arange(0, n_samples)
        # 计算每个样本点的eps邻域
        for i in range(n_samples):
            # 针对每一个测试样本计算它到训练集每个样本的距离
            # 这里只用欧式距离
            distances = np.linalg.norm(X - X[i], axis=1)
            eps_neignbors.append(indices[distances<=self.eps])
        # 过滤出核心对象
        n_neighbors = np.array([len(neighbors)
                                for neighbors in eps_neignbors])
        is_core_samples = n_neighbors >= self.min_samples
        self.core_sample_indices = indices[is_core_samples]
        # 随机选取核心对象并从其出发进行扩充聚类，直到没有核心对象；剩下来的样本的为噪声点/离群点
        # 不加self的core_sample_indices只是个临时对象
        core_sample_indices = set(self.core_sample_indices)
        # clusters = [] # 记录聚类簇 # 这个东西好像并不需要
        k = 0   # 表示第几个类
        labels = np.ones(n_samples, dtype=np.int32) * -1 # 所有样本都设置成-1，后面就不用为噪声点再遍历一次了
        is_scanned = np.zeros(n_samples, dtype=bool) # 记录是否已经扫描过了
        while(core_sample_indices):
            core = random.sample(core_sample_indices,1)[0] # 随机选取一个核心对象
            q = [core] # 用核心对象初始化一个空队列(这里就利用列表来做)
            is_scanned[core] = True
            while(q):
                current = q.pop(0)
                labels[current] = k  # 设置标签
                if is_core_samples[current]:
                    core_sample_indices.remove(current)
                    extended = []
                    for x in eps_neignbors[current]:
                        if not is_scanned[x]:
                            extended.append(x)
                            is_scanned[x] = True
                    q.extend(extended) # 如果是核心对象就扩展
            k += 1

        self.labels = labels

        return self

    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.labels