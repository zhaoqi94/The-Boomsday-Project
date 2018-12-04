


class Tree:
    def __init__(self, attr_index=-1, split_value=None,
                 results=None, left_sub_tree=None, right_sub_tree=None):
        self.attr_index = attr_index
        self.split_value = split_value
        self.results = results
        self.left_sub_tree = left_sub_tree
        self.right_sub_tree = right_sub_tree


class TreeBuilder:
    def __init__(self, splitter, max_depth, min_samples_split, min_samples_leaf):
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion_ = None
        self.max_features_ = None

    def build(self, X, y):
        return self._build(X, y, 1)

    # 目前使用递归实现的，可以改用非递归实现，效率会更高
    def _build(self, X, y, depth, ):

        self.splitter.node_reset(X, y)

        sample_num = X.shape[0]
        if (depth > self.max_depth) or (sample_num < self.min_samples_split):
            return Tree(results=self.splitter.node_value())

        # TODO 重置splitter
        self.splitter.node_reset(X, y)
        # TODO 划分节点
        best_split = self.splitter.node_split()

        if best_split["left_X"].shape[0] == 0 or best_split["right_X"].shape[0] == 0:
            return Tree(results=self.splitter.node_value())

        depth += 1
        left_sub_tree = self._build(
            best_split["left_X"],
            best_split["left_y"],
            depth)
        right_sub_tree = self._build(
            best_split["right_X"],
            best_split["right_y"],
            depth)

        return Tree(
            attr_index=best_split["best_attr_index"],
            split_value=best_split["best_split_value"],
            results=None,
            left_sub_tree=left_sub_tree,
            right_sub_tree=right_sub_tree,
        )

