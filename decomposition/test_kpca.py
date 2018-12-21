# TODO 可能需要一个更具代表性的数据集 比如说瑞士卷》》》

from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import seaborn
from mpl_toolkits.mplot3d import Axes3D

mnist = datasets.load_digits()
X = mnist.data
y = mnist.target

print(X.shape)

from sklearn.decomposition import KernelPCA
pca = KernelPCA(n_components=2,eigen_solver="dense",kernel="poly")
X_1 = pca.fit_transform(X)

from decomposition.kpca import KernelPCA
pca = KernelPCA(n_dims=2, kernel="poly")
X_2 = pca.fit_transform(X)


fig = plt.figure(figsize=(4,8))
plt.subplot(211)
plt.scatter(X_1[:, 0], X_1[:, 1], c=y, cmap=plt.cm.Spectral)

plt.subplot(212)
plt.scatter(X_2[:, 0], X_2[:, 1], c=y, cmap=plt.cm.Spectral)
plt.show()