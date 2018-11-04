import numpy as np
import copy
from collections import Counter

class BaggingClassifier:
    def __init__(self, base_estimator, n_estimators,
                 sample_rate, bootstrap=True):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.sample_rate = sample_rate
        self.bootstrap = bootstrap

        # 获得基学习器的深拷贝
        self.estimators = [copy.deepcopy(self.base_estimator) for _ in range(self.n_estimators)]

    def fit(self, X, y):
        for i in range(self.n_estimators):
            sample_X_indices = np.random.randint(0, X.shape[0], int(X.shape[0] * self.sample_rate))
            sample_X = X[sample_X_indices]
            sample_y = y[sample_X_indices]
            # 从整个训练集中采样出子集
            self.estimators[i].fit(sample_X, sample_y)

    # 投票
    def predict(self, X):
        results = []
        for i in range(self.n_estimators):
            results.append(self.estimators[i].predict(X))
        results = np.array(results).T
        y_pred = np.array(list(map(lambda v: Counter(v).most_common(1)[0][0], results)))

        return y_pred

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred==y)

    # 对所有分类器的决策函数求平均
    def decision_function(self, X):
        dcf = []
        for i in range(self.n_estimators):
            dcf.append(self.estimators[i].decision_function(X))
        avg = np.mean(dcf, axis=1)

        return avg