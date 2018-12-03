import numpy as np
from abc import ABCMeta, abstractmethod


# 把计算impurity的判别准则抽象成一个类
# 模仿sklearn
class Criterion(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        pass

    def node_impurity(self, y):
        pass

    def node_value(self, y):
        pass
    # def children_impurity(self):
    #     pass

    # def init(self, y, ):
    #     self.y = y
    #
    # def reset(self, y):
    #     self.y = y

    # def impurity_gain(self):
    #     impurity_left, impurity_right = self.children_impurity()
    #
    #     return  (impurity - (self.weighted_n_right /
    #                          self.weighted_n_node_samples * impurity_right)
    #                       - (self.weighted_n_left /
    #                          self.weighted_n_node_samples * impurity_left)))


class Gini(Criterion):
    def __init__(self):
        super(Gini, self).__init__()

    def node_impurity(self, y):
        if y.shape[0] == 0:
            return 0.0
        gini = 1.0
        unique_item, unique_counts = np.unique(y, return_counts=True)
        N = y.shape[0]
        gini -= np.sum(np.square(unique_counts)) / (N * N)
        return gini

    def node_value(self, y):
        unique_item, unique_counts = np.unique(y, return_counts=True)
        return unique_item[np.argmax(unique_item)]

class MSE(Criterion):
    def __init__(self):
        super(MSE, self).__init__()

    def node_impurity(self, y):
        if y.shape[0] == 0:
            return 0.0
        return np.mean(np.square(y - np.mean(y)))

    def node_value(self, y):
        return np.mean(y)