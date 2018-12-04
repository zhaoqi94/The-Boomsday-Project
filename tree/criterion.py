import numpy as np
from abc import ABCMeta, abstractmethod


# 把计算impurity的判别准则抽象成一个类
# 模仿sklearn
class Criterion(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        self.y = None   # 计算Criterion, 只需要y，不需要X
        self.indices = None # 利用indices保存样本的划分，这样就不需要频繁地对X进行复制了，也不需要对原始数据进行排序，只需要排序indices即可
        self.start = None   # 子树的开始位置
        self.end = None     # 子树的结束位置
        self.pos = None     # 游标

        self.impurity = None

    def init(self, y, indices, start, end):
        self.y = y
        self.indices = indices
        self.start = start
        self.end = end

        # 初始化
        self.pos = start

        self.impurity = self.node_impurity()

    def reset(self):
        self.pos = self.start
        self.impurity = self.node_impurity()
    # 更新pos
    def update(self, new_pos):
        self.pos = new_pos

    def node_impurity(self):
        return 0.0

    def node_value(self):
        return 0.0

    def children_impurity(self):
        return 0.0, 0.0

    def impurity_gain(self):
        impurity_left, impurity_right = self.children_impurity()
        sample_num = self.end - self.start
        left_num = self.pos - self.start
        right_num = self.end - self.pos
        return  self.impurity - (left_num / sample_num * impurity_left) - (right_num / sample_num * impurity_right)

    def proxy_impurity_gain(self):
        return self.impurity_gain()

class Gini(Criterion):
    def __init__(self):
        super(Gini, self).__init__()

    def node_impurity(self):
        y = self.y[self.indices[self.start:self.end]]
        if y.shape[0] == 0:
            return 0.0
        gini = 1.0
        unique_item, unique_counts = np.unique(y, return_counts=True)
        N = y.shape[0]
        gini -= np.sum(np.square(unique_counts)) / (N * N)
        return gini

    def children_impurity(self):
        left_gini = 0.0
        right_gini = 0.0

        y = self.y[self.indices[self.start:self.pos]]
        if y.shape[0] > 0:
            gini = 1.0
            unique_item, unique_counts = np.unique(y, return_counts=True)
            N = y.shape[0]
            gini -= np.sum(np.square(unique_counts)) / (N * N)
            left_gini = gini

        y = self.y[self.indices[self.pos:self.end]]
        if y.shape[0] > 0:
            gini = 1.0
            unique_item, unique_counts = np.unique(y, return_counts=True)
            N = y.shape[0]
            gini -= np.sum(np.square(unique_counts)) / (N * N)
            right_gini = gini

        return left_gini, right_gini

    def node_value(self):
        y = self.y[self.indices[self.start:self.end]]
        unique_item, unique_counts = np.unique(y, return_counts=True)
        return unique_item[np.argmax(unique_item)]

class MSE(Criterion):
    def __init__(self):
        super(MSE, self).__init__()

    def init(self, y, indices, start, end):
        # 调用父类方法
        super(MSE, self).init(y, indices, start, end)
        self.sum_left = 0
        self.sum_right = np.sum(np.square(y))

    def reset(self):
        self.pos = self.start
        #self.impurity = self.node_impurity()
        self.sum_left = 0
        self.sum_right = np.sum(self.y[self.indices[self.start:self.end]])

    # 更新pos
    def update(self, new_pos):
        for p in range(self.pos, new_pos):
            self.sum_left += self.y[self.indices[p]]
            self.sum_right -= self.y[self.indices[p]]
        self.pos = new_pos



    def node_impurity(self):
        y = self.y[self.indices[self.start:self.end]]
        if y.shape[0] == 0:
            return 0.0
        return np.mean(np.square(y - np.mean(y)))

    def node_value(self):
        y = self.y[self.indices[self.start:self.end]]
        return np.mean(y)

    def children_impurity(self):
        left_mse = 0.0
        right_mse = 0.0

        y = self.y[self.indices[self.start:self.pos]]
        if y.shape[0] > 0:
            left_mse = np.mean(np.square(y - np.mean(y)))

        y = self.y[self.indices[self.pos:self.end]]
        if y.shape[0] > 0:
            right_mse = np.mean(np.square(y - np.mean(y)))

        return left_mse, right_mse

    def proxy_impurity_gain(self):
        samples_left = self.pos - self.start
        samples_right = self.end - self.pos
        if samples_left != 0:
            proxy_impurity_left = self.sum_left * self.sum_left / samples_left
        else:
            proxy_impurity_left = 0.0

        if samples_right != 0:
            proxy_impurity_right = self.sum_right * self.sum_right / samples_right
        else:
            proxy_impurity_right = 0.0

        return proxy_impurity_left + proxy_impurity_right