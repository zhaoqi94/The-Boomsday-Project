# TODO Naive Bayes Classifier
# TODO 1.Gaussian Naive Bayes
# TODO 2.Bernulli Naive Bayes
# TODO 3.Multinomial Naive Bayes
# TODO 4.Incremental learning

import numpy as np
from sklearn.preprocessing import LabelBinarizer
from scipy.sparse import issparse

class GaussianNB:
    def __init__(self):
        self.class_prior = None
        self.theta = None
        self.sigma = None

    def fit(self, X, y):
        # 1.初始化参数
        self.classes = np.unique(y)
        n_classes = self.classes.shape[0]
        n_features = X.shape[1]
        self.theta = np.zeros((n_classes, n_features))
        self.sigma = np.zeros((n_classes, n_features))
        self.class_prior = np.zeros(n_classes)

        # 2.遍历y，估计统计量
        for y_i in self.classes:
            i = self.classes.searchsorted(y_i)
            X_i = X[y == y_i, :]
            # 3.针对每一个y和feature，计算均值和方差
            theta = np.mean(X_i, axis=0)
            sigma = np.var(X_i, axis=0)
            self.theta[i, :] = theta
            self.sigma[i, :] = sigma
            self.class_prior[i] = X_i.shape[0] / X.shape[0]

    def predict(self, X):
        jll = self._joint_log_likelihood(X)
        return self.classes[np.argmax(jll, axis=1)]

    def score(self, X, y):
        return np.mean(self.predict(X)==y)

    def _joint_log_likelihood(self, X):
        joint_log_likelihood = []
        for i in range(np.size(self.classes)):
            log_class_density = np.log(self.class_prior[i])
            log_conditional_density = - 0.5 * np.sum(np.log(2. * np.pi * self.sigma[i, :]))
            log_conditional_density -= 0.5 * np.sum(((X - self.theta[i, :]) ** 2) /
                                 (self.sigma[i, :]), axis=1)
            joint_log_likelihood.append(log_class_density+log_conditional_density)

        joint_log_likelihood = np.array(joint_log_likelihood).T
        return joint_log_likelihood



# TODO BernoulliNB
class BernoulliNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.classes = None
        self.class_count = None
        self.feature_count = None
        self.feature_log_prob = None
        self.class_log_prior = None

    def fit(self, X, y):
        # TODO 1.初始化参数&标签转换成one_hot方便计算
        lb = LabelBinarizer()
        y = lb.fit_transform(y)
        self.classes = lb.classes_
        # 如果只有两类，那fit_transform就只会产生1维编码
        if y.shape[1] == 1:
            y = np.concatenate((1 - y, y), axis=1)
        # TODO 2.数据集，进行计数
        self.feature_count = safe_sparse_dot(y.T, X) # (n_classes, n_featrues)
        self.class_count = np.sum(y, axis=0)

        # TODO 3.计算log概率值
        smoothed_fc = self.feature_count + self.alpha
        smoothed_cc = self.class_count + self.alpha * 2
        self.feature_log_prob = (np.log(smoothed_fc) -
                                  np.log(smoothed_cc.reshape(-1, 1)))
        self.class_log_prior = (np.log(self.class_count) -
                                 np.log(np.sum(self.class_count)))

    def predict(self, X):
        jll = self._joint_log_likelihood(X)
        return self.classes[np.argmax(jll, axis=1)]

    def score(self, X, y):
        return np.mean(self.predict(X)==y)

    def _joint_log_likelihood(self, X):
        neg_prob = np.log(1 - np.exp(self.feature_log_prob))
        jll = safe_sparse_dot(X, (self.feature_log_prob - neg_prob).T)
        jll += self.class_log_prior + neg_prob.sum(axis=1)

        return jll


# TODO MultinomialNB:
class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.classes = None
        self.class_count = None
        self.feature_count = None
        self.feature_log_prob = None
        self.class_log_prior = None

    def fit(self, X, y):
        # TODO 1.初始化参数&标签转换成one_hot方便计算
        lb = LabelBinarizer()
        y = lb.fit_transform(y)
        self.classes = lb.classes_
        # 如果只有两类，那fit_transform就只会产生1维编码
        if y.shape[1] == 1:
            y = np.concatenate((1 - y, y), axis=1)
        # TODO 2.数据集，进行计数
        self.feature_count = safe_sparse_dot(y.T, X)  # (n_classes, n_featrues)
        self.class_count = np.sum(y, axis=0)

        # TODO 3.计算log概率值
        smoothed_fc = self.feature_count + self.alpha
        smoothed_cc = np.sum(smoothed_fc, axis=1)
        self.feature_log_prob = (np.log(smoothed_fc) -
                                 np.log(smoothed_cc.reshape(-1, 1)))
        self.class_log_prior = (np.log(self.class_count) -
                                np.log(np.sum(self.class_count)))

    def predict(self, X):
        jll = self._joint_log_likelihood(X)
        return self.classes[np.argmax(jll, axis=1)]

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

    def _joint_log_likelihood(self, X):
        jll = safe_sparse_dot(X, self.feature_log_prob.T) + self.class_log_prior

        return jll


def safe_sparse_dot(a, b, dense_output=False):
    if issparse(a) or issparse(b):
        return a * b
    else:
        return np.dot(a, b)