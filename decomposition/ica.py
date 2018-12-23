# TODO 独立成分分析 ICA Independent Component Analysis
# 和PCA不同，不是为了降维，而是为了从混合信号中提取出有实际意义的信息
# 盲源分离问题 BBS Bind Source Seperation
# 用于信号的解混，比如从n个麦克风中解码出n个独立的声音
# x=As s:信号源 x:混合后的信号（麦克风）
# s各个分量都是独立的 不能为高斯分布
# 解混 s=inv(A)s=Wx
# TODO 1. 假设混合矩阵是方阵，假设s的先验分布，并使用最大似然估计求解参数 (不知道为什么，效果就是不行啊！)
# TODO 2. FastICA算法，并且使混合矩阵可以不是方阵

import numpy as np
from utils.activation import sigmoid
from sklearn.utils import shuffle

class ICA:
    def __init__(self, max_epochs=10, lr=0.01):
        self.max_epochs = max_epochs
        self.lr = lr
        self.W = None


    def fit(self, X, y=None):
        # 初始化
        n_samples, n_features = X.shape
        # W = np.eye(n_features) + 0.1 * np.random.randn(n_features,n_features)
        W = np.random.randn(n_features,n_features)
        '''
        batch_size = 1
        for i in range(self.max_epochs):
            X = shuffle(X)
            for j in range(int(n_samples/batch_size)):
                grad_W = np.zeros((n_features,n_features))
                for k in range(batch_size):
                    x = X[j*batch_size+k]
                    grad_W += np.outer((1 - 2*sigmoid(np.dot(W, x))), x) + np.linalg.inv(W.T)

                W += self.lr * grad_W / batch_size
        '''
        for i in range(self.max_epochs):
            X = shuffle(X)
            for x in X:
                grad_W = np.outer((1 - 2*1.0/(1.0+np.exp(-np.dot(W, x)))), x) + np.linalg.inv(W.T)
                W += self.lr * grad_W
        self.W = W

        return self

    def transform(self, X):
        return np.dot(X, self.W.T)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)