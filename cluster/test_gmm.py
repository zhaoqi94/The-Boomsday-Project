from cluster.gmm import GaussianMixture
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Generate dataset
X, y = make_blobs(centers=3, n_samples=500, random_state=1)

# 使用random的初始化方式，很容易就聚不好；
cls = GaussianMixture(n_components=3, max_iter=100, init_params="random", tol=1e-8)
cls.fit(X)

group_colors = ['skyblue', 'coral', 'lightgreen']
colors = [group_colors[j] for j in cls.labels]
fig, ax = plt.subplots(figsize=(4,4))
ax.scatter(X[:,0], X[:,1], color=colors, alpha=0.5)
ax.scatter(cls.means[:,0], cls.means[:,1], color=['blue', 'darkred', 'green'], marker='o', lw=2)
ax.set_xlabel('$x_0$')
ax.set_ylabel('$x_1$')
plt.show()


# from sklearn.mixture import GaussianMixture
# cls = GaussianMixture(n_components=3, max_iter=100, init_params="random", tol=1e-8)
# cls.fit(X)
# labels = cls.predict(X)
#
# group_colors = ['skyblue', 'coral', 'lightgreen']
# colors = [group_colors[j] for j in labels]
# fig, ax = plt.subplots(figsize=(4,4))
# ax.scatter(X[:,0], X[:,1], color=colors, alpha=0.5)
# ax.scatter(cls.means_[:,0], cls.means_[:,1], color=['blue', 'darkred', 'green'], marker='o', lw=2)
# ax.set_xlabel('$x_0$')
# ax.set_ylabel('$x_1$')
# plt.show()
