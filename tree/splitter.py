import numpy as np
from abc import ABCMeta, abstractmethod
from tree.criterion import Criterion

FEATURE_THRESHOLD = 1e-7

class SplitRecord:
    impurity_gain = -np.inf  # impurity gain
    attr_index = -1 # 划分的属性
    split_value = 0.0 # 划分点（值）
    pos = 0 # 划分的位置，用于划分左右子树 左子树[start,pos),右子树[pos,end)


class Splitter(metaclass=ABCMeta):
    """
    Spliter类抽象出划分属性的过程
    """
    @abstractmethod
    def __init__(self, criterion, max_features,
                  min_samples_leaf):
        self.criterion = criterion
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.X = None
        self.y = None
        self.indices = None
        self.start = None
        self.end = None

    def init(self, X, y):
        self.X = X
        self.y = y
        self.indices = np.arange(0, X.shape[0])

    def node_reset(self, start, end):
        # 保存start和end
        self.start = start
        self.end = end
        # 重新初始化criterion
        self.criterion.init(self.y, self.indices, start, end)

    def node_split(self):
        pass

    def node_impurity(self):
        return self.criterion.node_impurity()

    def node_value(self):
        return self.criterion.node_value()


class BestSplitter(Splitter):
    def __init__(self, criterion, max_features,
                  min_samples_leaf):
        super(BestSplitter, self).__init__(
            criterion, max_features, min_samples_leaf
        )

    def node_split(self):
        # 处理一下变量名 省略‘self.’
        X,y = self.X,self.y
        sample_num, feature_num = X.shape
        indices = self.indices
        start = self.start
        end = self.end

        # 选择候选特征集
        selected_features = np.random.choice(feature_num, self.max_features, replace=False)
        # print(feature_num)
        best_split = SplitRecord()

        import time
        for j in selected_features:
            # t1 = time.time()
            current_attr_index = j
            # TODO 重置一下criterion
            self.criterion.reset()
            # TODO 使用候选特征对indices进行排序

            sorted_ind = np.argsort(X[indices[start:end], j])
            indices[start:end] = indices[start:end][sorted_ind]
            sorted_feature = np.append(X[indices[start:end], j],[np.inf])

            # TODO 移动游标，刷选出最好的分割点
            # 注意其实x>=-np.inf和x<=np.inf这个分割点本质上是一样的
            # 注意，分割点仍然是相邻的游标位置处的值相加求和除以2，但是不需要事先计算出来
            # 举个例子：
            # x==>[1,2,4,10,100]
            # 所有候选分割点是[1.5,3.0,55.0,inf] 判别左右子树用 ‘<=’
            # t2 = time.time()
            # for pos in range(start, end+1):
            pos = start
            while pos < end:
                # TODO 移动pos点到合适的位置，目的是过滤掉差不多相等的特征值
                while ( pos < end and sorted_feature[pos-start+1] <= sorted_feature[pos-start] + FEATURE_THRESHOLD):
                    pos = pos + 1
                # TODO 按pos点进行分割 并调用criterion进行计算
                self.criterion.update(pos)
                impurity_gain = self.criterion.proxy_impurity_gain()

                if impurity_gain >= best_split.impurity_gain:
                    best_split.impurity_gain = impurity_gain
                    best_split.attr_index = current_attr_index
                    best_split.split_value = (sorted_feature[pos-start] + sorted_feature[pos-start-1]) / 2.0
                    best_split.pos = pos
                pos = pos + 1
            # t3 = time.time()
            # print("1-2",t2-t1)
            # print("2-3",t3-t2)

        # TODO 需要重排啊
        if best_split.pos < end:
            sorted_ind = np.argsort(X[indices[start:end], best_split.attr_index])
            indices[start:end] = indices[start:end][sorted_ind]

        # end_time = time.time()
        # print(end_time-start_time)
        return best_split
