# TODO 等度量映射 Isomap: Isometric Mapping
# TODO 1. Isomap+MDS (周志华书上的方案) （Isomap的实现可以采用sklearn提供的一些函数）
# TODO 2. 自己整理出knn的接口+邻近图+最短路径算法（复习算法与数据结构的时候）
# TODO 3. 实现sklearn的方案 Isomap+KPCA

import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.utils.graph import graph_shortest_path
from manifold.mds import MDS

class Isomap:
    def __init__(self, n_dims=2, n_neighbors=10):
        self.n_dims = n_dims
        self.n_neighbors = n_neighbors

    def fit(self, X, y=None):
        self.fit_transform(X)
        return self

    def fit_transform(self, X, y=None):
        kng = kneighbors_graph(X, self.n_neighbors, mode='distance')
        dist_matrix = graph_shortest_path(kng, method="auto", directed=False)
        mds = MDS(self.n_dims, dissimilarity="precomputed")
        return mds.fit_transform(dist_matrix)

    def transform(self, X):
        pass

