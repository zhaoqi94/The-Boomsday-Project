import numpy as np
import copy
from collections import Counter

class BaseBagging:
    def __init__(self, base_estimator, n_estimators,
                 sample_rate, bootstrap=True):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.sample_rate = sample_rate
        self.bootstrap = bootstrap      # bootstrap表示有放回的采样

        # 获得基学习器的深拷贝
        self.estimators = [copy.deepcopy(self.base_estimator) for _ in range(self.n_estimators)]

    def fit(self, X, y):
        for i in range(self.n_estimators):
            sample_X_indices = np.random.randint(0, X.shape[0], int(X.shape[0] * self.sample_rate))
            sample_X = X[sample_X_indices]
            sample_y = y[sample_X_indices]
            # 从整个训练集中采样出子集
            self.estimators[i].fit(sample_X, sample_y)


class BaggingClassifier(BaseBagging):
    def __init__(self, base_estimator, n_estimators,
                 sample_rate, bootstrap=True):
        super(BaggingClassifier, self).__init__(
            base_estimator, n_estimators,
            sample_rate, bootstrap
        )

    # 投票 Majority voting
    def predict(self, X):
        results = []
        for i in range(self.n_estimators):
            results.append(self.estimators[i].predict(X))
        results = np.array(results).T
        y_pred = np.array(list(map(lambda v: Counter(v).most_common(1)[0][0], results)))

        return y_pred

    # 准确率
    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred==y)

    # 对所有分类器的决策函数求平均
    def decision_function(self, X):
        dcf = []
        for i in range(self.n_estimators):
            dcf.append(self.estimators[i].decision_function(X))
        avg = np.mean(dcf, axis=0)

        return avg

class BaggingRegressor(BaseBagging):
    def __init__(self, base_estimator, n_estimators,
                 sample_rate, bootstrap=True):
        super(BaggingRegressor, self).__init__(
            base_estimator, n_estimators,
            sample_rate, bootstrap
        )

    # 预测回归值
    def predict(self, X):
        results = []
        for i in range(self.n_estimators):
            results.append(self.estimators[i].predict(X))
        y_pred = np.mean(results, axis=0)

        return y_pred

    # 参考sklearn 回归的指标
    def score(self, X, y):
        return 1 - np.sum(np.square(self.predict(X) - np.array(y))) / np.sum(np.square(np.mean(y) - np.array(y)))


