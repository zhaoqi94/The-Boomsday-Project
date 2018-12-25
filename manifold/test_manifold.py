# Author: Jake Vanderplas -- <vanderplas@astro.washington.edu>
# sklearn的一个例子的改写

print(__doc__)

from time import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter

from sklearn import manifold, datasets

# Next line to silence pyflakes. This import is needed.

n_points = 1000
X, color = datasets.samples_generator.make_s_curve(n_points, random_state=0)
n_neighbors = 10
n_components = 2

fig = plt.figure(figsize=(15, 8))
plt.suptitle("Manifold Learning with %i points, %i neighbors"
             % (1000, n_neighbors), fontsize=14)


ax = fig.add_subplot(251, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
ax.view_init(4, -72)


# sklearn MDS
t0 = time()
mds = manifold.MDS(n_components, max_iter=100, n_init=1)
Y = mds.fit_transform(X)
t1 = time()
print("sklearn MDS: %.2g sec" % (t1 - t0))
ax = fig.add_subplot(252)
plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
plt.title("sklearn MDS (%.2g sec)" % (t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')

# 我的MDS
from manifold.mds import MDS
t0 = time()
mds = MDS(n_components)
Y = mds.fit_transform(X)
t1 = time()
print("My MDS: %.2g sec" % (t1 - t0))
ax = fig.add_subplot(253)
plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
plt.title("My MDS (%.2g sec)" % (t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')


from manifold.isomap import Isomap
t0 = time()
imp = Isomap(n_components)
Y = imp.fit_transform(X)
t1 = time()
print("Isomap+MDS: %.2g sec" % (t1 - t0))
ax = fig.add_subplot(254)
plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
plt.title("Isomap+MDS (%.2g sec)" % (t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')

# 效果和sklearn不一样！！！
from manifold.lle import LocallyLinearEmbedding
t0 = time()
lle = LocallyLinearEmbedding(n_components, n_neighbors)
Y = lle.fit_transform(X)
t1 = time()
print("LLE: %.2g sec" % (t1 - t0))
ax = fig.add_subplot(255)
plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
plt.title("LLE (%.2g sec)" % (t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')



plt.show()

