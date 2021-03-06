#  K-means无法聚类非凸数据，而DBSCAN可以

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

plt.figure(figsize=(10,6))

X1, y1=datasets.make_circles(n_samples=5000, factor=.6,
                                      noise=.05)
X2, y2 = datasets.make_blobs(n_samples=1000, n_features=2, centers=[[1.2,1.2]], cluster_std=[[.1]],
               random_state=9)

X = np.concatenate((X1, X2))
plt.subplot(231)
plt.scatter(X[:, 0], X[:, 1], marker='o')
plt.title("原数据")

from sklearn.cluster import KMeans
y_pred = KMeans(n_clusters=3, random_state=9).fit_predict(X)
plt.subplot(232)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("KMeans")

from sklearn.cluster import DBSCAN
y_pred = DBSCAN().fit_predict(X)
plt.subplot(233)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("DBSCAN (0.5, 5)")

y_pred = DBSCAN(eps = 0.1).fit_predict(X)
plt.subplot(234)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("DBSCAN (0.1, 5)")

y_pred = DBSCAN(eps = 0.1, min_samples = 10).fit_predict(X)
plt.subplot(235)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("DBSCAN (0.1, 10)")

from cluster.dbscan import DBSCAN
y_pred = DBSCAN(eps = 0.1, min_samples = 5).fit_predict(X)
plt.subplot(236)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("我的DBSCAN (0.1, 5)")
plt.show()















'''
# 第二个例子
# sklearn中的DBSCAN的例子
# 加上自己的DBSCANde的例子，进行对比

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler


def print_score(labels, labels_true):
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(labels_true, labels))
    print("Adjusted Mutual Information: %0.3f"
          % metrics.adjusted_mutual_info_score(labels_true, labels))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(X, labels))

def plot_cluster(labels, core_samples_mask):
    # #############################################################################
    # Plot result
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()

# sklearn DBSCAN
# #############################################################################
# Generate sample data
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
                            random_state=0)
X = StandardScaler().fit_transform(X)

# #############################################################################
# Compute DBSCAN
db = DBSCAN(eps=0.3, min_samples=10).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

print_score(labels, labels_true)
plot_cluster(labels, core_samples_mask)



from cluster.dbscan import DBSCAN
# #############################################################################
# Compute DBSCAN
db = DBSCAN(eps=0.3, min_samples=10).fit(X)
core_samples_mask = np.zeros_like(db.labels, dtype=bool)
core_samples_mask[db.core_sample_indices] = True
labels = db.labels

print_score(labels, labels_true)
plot_cluster(labels, core_samples_mask)
'''