import  numpy as np

class Tree:
    def __init__(self, attr_index=-1, split_value=None,
                 results=None, left_sub_tree=None, right_sub_tree=None):
        self.attr_index = attr_index
        self.split_value = split_value
        self.results = results
        self.left_sub_tree = left_sub_tree
        self.right_sub_tree = right_sub_tree

    def predict(self, X):
        y_pred = []

        for i in range(X.shape[0]):
            tree = self
            while(True):
                if tree.results != None:
                    y_pred.append(tree.results)
                    break

                if X[i][tree.attr_index] <= tree.split_value:
                    tree = tree.left_sub_tree
                else:
                    tree = tree.right_sub_tree

        return np.array(y_pred)


class TreeBuilder:
    def __init__(self, splitter, max_depth, min_samples_split, min_samples_leaf):
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion_ = None
        self.max_features_ = None

    def build(self, X, y):
        self.splitter.init(X, y)
        sample_num = X.shape[0]
        return self._build(1, 0, sample_num)

    # 目前使用递归实现的，可以改用非递归实现，效率会更高
    def _build(self, depth, start, end ):

        # TODO 重置splitter
        self.splitter.node_reset(start, end)
        # 判断是否终止（叶节点）
        sample_num = end - start
        if (depth > self.max_depth) or (sample_num < self.min_samples_split):
            return Tree(results=self.splitter.node_value())


        # TODO 划分节点
        # import time
        # start_time = time.time()
        best_split = self.splitter.node_split()
        # end_time = time.time()
        # if depth == 1:
        #     print((end_time-start_time))

        # print(start, end, best_split.pos)
        if best_split.pos <= start or best_split.pos >= end:
            return Tree(results=self.splitter.node_value())

        depth += 1
        left_sub_tree = self._build(
            depth,
            start,
            best_split.pos)
        right_sub_tree = self._build(
            depth,
            best_split.pos,
            end)

        return Tree(
            attr_index=best_split.attr_index,
            split_value=best_split.split_value,
            results=None,
            left_sub_tree=left_sub_tree,
            right_sub_tree=right_sub_tree,
        )

