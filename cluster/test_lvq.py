from cluster.lvq import LVQ
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Generate dataset
X, y = make_blobs(centers=3, n_samples=500, random_state=1)

# 需要重复运行10次k-means，否则就会以一定几率出现很差的情况
cls = LVQ()
cls.fit(X, y)

group_colors = ['skyblue', 'coral', 'lightgreen']
colors = [group_colors[j] for j in cls.labels]
fig, ax = plt.subplots(figsize=(4,4))
ax.scatter(X[:,0], X[:,1], color=colors, alpha=0.5)
ax.scatter(cls.cluster_centers[:,0], cls.cluster_centers[:,1], color=['blue', 'darkred', 'green'], marker='o', lw=2)
ax.set_xlabel('$x_0$')
ax.set_ylabel('$x_1$')
plt.show()
