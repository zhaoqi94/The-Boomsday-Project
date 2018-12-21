from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import seaborn
from mpl_toolkits.mplot3d import Axes3D

mnist = datasets.load_digits()
X = mnist.data
y = mnist.target

# X[0], X[-1] = X[-1], X[0]
from sklearn.decomposition import PCA
pca = PCA(n_components=2, svd_solver="full")
X_1 = pca.fit_transform(X)
print(np.var(X_1, axis=0))

from decomposition.pca import PCA
pca = PCA(n_dims=2)
X_2 = pca.fit_transform(X)
print(np.var(X_2, axis=0))

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.scatter(new_X[:, 0], new_X[:, 1], new_X[:, 2], c=y, cmap=plt.cm.Spectral)
# plt.show()

fig = plt.figure(figsize=(4,8))
plt.subplot(211)
plt.scatter(X_1[:, 0], X_1[:, 1], c=y, cmap=plt.cm.Spectral)

plt.subplot(212)
plt.scatter(X_2[:, 0], X_2[:, 1], c=y, cmap=plt.cm.Spectral)
plt.show()