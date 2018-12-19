# TODO KDTree实现
# 1. 简化版，不使用高级数据结构，且在选择划分轴时只使用 depth % n_features 且只用数据本身而非index(这样最后还要去找index,开销反而更大了，但是比较好实现)
# 2. 使用样本的indice而不是样本本身
# TODO 3. 限定叶节点的个数,而不是每个节点只有一个样本点
# TODO 3. 使用优先队列/堆，但是有点难

import numpy as np

class KDTree:
    class Node:
        def __init__(self,axis, point, left, right):
            self.parent = None
            self.axis = axis
            self.point = point
            self.left = left
            self.right = right

    def __init__(self, X, leaf_size=2):
        self.X = X
        self.X = X
        self.n_samples, self.n_features = X.shape
        # self.indices = np.arange(0, self.n_samples)
        self.leaf_size = leaf_size  # 暂时先不用
        self.root = self._build_tree(np.arange(0, self.n_samples), 0)

    # 就先不用样本索引值而直接使用样本
    def _build_tree(self, X_ind, depth):
        # 终止条件：如果X为空的话，返回一个None
        # 也可以进一步地使用leaf_size条件
        # print(X.shape)
        if X_ind.shape[0] == 0:
            return None
        # 确定划分轴
        axis = depth % self.n_features
        # 划分数据集X
        X_ind_f = self.X[:,axis][X_ind] # 先取列，再取行，开销小，因为在这里X_ind是不规则的
        sorted_ind = X_ind_f.argsort()
        sorted_X_ind = X_ind[sorted_ind]
        median = int(sorted_X_ind.shape[0] / 2)
        left_X_ind = sorted_X_ind[:median]
        right_X_ind = sorted_X_ind[median+1:]
        median_X_ind = sorted_X_ind[median]
        # 构建左子树
        left = self._build_tree(left_X_ind, depth + 1)
        # 构建右子树
        right = self._build_tree(right_X_ind, depth + 1)
        # 构建父节点
        node = self.Node(axis, median_X_ind, left, right)
        if left != None:
            left.parent = node
        if right != None:
            right.parent = node

        return node

    # 寻找距离最近的n个样本
    def kd_nearest_n(self, x, n):
        self.count = 0
        self.n_nearest_neighbor = []
        self.n_distances = []
        # choice = np.random.choice(self.n_samples, n)
        # for i in range(n):
        #     self.n_nearest_neighbor.append(choice[i])
        #     self.n_distances.append(np.linalg.norm(x-self.X[choice[i]]))
        self._kd_nearest_n(self.root, x, n)
        print(self.count)
        print(self.n_distances)
        return self.n_nearest_neighbor

    def _kd_nearest_n(self, node, x, n):
        if node==None:
            return
        # 使用节点的point来更新n_nearest_neighbor和n_distances
        cur_distance = np.linalg.norm(x-self.X[node.point])
        if len(self.n_nearest_neighbor) < n:
            self.n_nearest_neighbor.append(node.point)
            self.n_distances.append(cur_distance)
        else:

            max_distance_ind = int(np.argmax(self.n_distances))
            max_distance = self.n_distances[max_distance_ind]
            if max_distance > cur_distance:
                self.n_nearest_neighbor[max_distance_ind] = node.point
                self.n_distances[max_distance_ind] = cur_distance

        # 先比较x[axis]与node.point[axis] 确定首先搜索的子树
        axis = node.axis
        if(x[axis] < self.X[node.point][axis]):
            cur = node.left
            sibling = node.right
        else:
            cur = node.right
            sibling = node.left
        self._kd_nearest_n(cur,x,n)
        # 然后如果n_nearest_neighbor不是满的，或者另一颗子树的切分点的值满足半径约束，则继续搜索另一棵子树
        # if node.parent == None:
            # print("半径:",np.max(self.n_distances))
            # print("中心:",x[axis])
            # print("切分点",self.X[node.point][axis])
            # print("距离:",abs(x[axis]-self.X[node.point][axis]))
        if len(self.n_nearest_neighbor) < n or (np.max(self.n_distances) > abs(x[axis]-self.X[node.point][axis])):
            self.count += 1
            self._kd_nearest_n(sibling,x,n)
            print(self.count)
            print("半径:",np.max(self.n_distances))
            print("中心:",x[axis])
            print("切分点",self.X[node.point][axis])
            print("距离:",abs(x[axis]-self.X[node.point][axis]))




