import numpy as np
import copy

class AdaBoostClassifier:
    def __init__(self, base_estimator, n_estimators=50):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators

        # 获得基学习器的深拷贝
        self.estimators = [copy.deepcopy(self.base_estimator) for _ in range(self.n_estimators)]
        # 记录样本权重
        self.w = None
        # 分类器权重
        self.alpha = None
        # 类别
        self.classes = None

    def fit(self, X, y):
        num = X.shape[0]
        self.w = np.ones(num)
        self.alpha = np.ones(self.n_estimators)
        self.classes = np.unique(y)

        for i in range(self.n_estimators):
            self.estimators[i].fit(X, y, sample_weight=self.w)
            y_pred = self.estimators[i].predict(X)
            # 1:正确 0:错误
            correct_error = (y_pred == y).astype(np.int32)
            # 1:正确 -1:错误
            correct_error[correct_error==0] = -1
            # 加权错误率
            weighted_error = np.dot(self.w, correct_error==-1) / np.sum(self.w)
            self.alpha[i] = 0.5 * np.log( (1 - weighted_error) / weighted_error)
            self.w = self.w * np.exp(-correct_error*self.alpha[i])

    def predict(self, X):
        dcf = self.decision_function(X)
        y_pred = self.classes[np.argmax(dcf, axis=1)]
        return y_pred

    def score(self, X, y):
        return np.mean(self.predict(X)==y)

    def decision_function(self, X):
        total_dcf = []
        for i in range(self.n_estimators):
            dcf_i = (self.estimators[i].predict(X) == self.classes[:, np.newaxis]).T * self.alpha[i]
            total_dcf.append(dcf_i)
        sum_dcf = np.sum(total_dcf, axis=0)
        sum_dcf /= np.sum(self.alpha)
        return sum_dcf

    # 输出前i个分类器的组合，可用于显示集成的效果
    # 以及做模型选择
    def staged_decision_function(self, X):
        dcf = None
        norm = 0
        for i in range(self.n_estimators):
            current_dcf = (self.estimators[i].predict(X) == self.classes[:, np.newaxis]).T * self.alpha[i]
            norm += self.alpha[i]
            if dcf is None:
                dcf = current_dcf
            else:
                dcf += current_dcf

            yield dcf / norm

    def staged_predict(self, X):
        for dcf in self.staged_decision_function(X):
            yield self.classes[np.argmax(dcf, axis=1)]

    def staged_score(self, X, y):
        for y_pred in self.staged_predict(X):
            yield np.mean(y_pred==y)