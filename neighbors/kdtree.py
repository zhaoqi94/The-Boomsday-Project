# TODO KDTree实现
# 1. 简化版，不使用高级数据结构，且在选择划分轴时只使用 depth % n_features 且只用数据本身而非index(这样最后还要去找index,开销反而更大了，但是比较好实现)
# 2. 使用样本的indice而不是样本本身
# 3. 用bounding box计算距离来进行“剪枝” ，首先弄个“假”边界盒即可，即它只是对空间的划分，而不是根据样本收缩的边界 (确实剪枝的数量变多了)
# 4. “真”边界盒 每次要算样本的范围
# TODO 5. 用更高效的方法计算“真”bounding_box 可以把两个子节点的边界盒回溯上来，合并成新的边界盒 https://github.com/jmhodges/kdtree2/blob/master/src-c%2B%2B/kdtree2.cpp
# TODO 6. 限定叶节点的个数,而不是每个节点只有一个样本点
# TODO 7. 使用优先队列/堆，但是有点难（4和5好像必须合起来做？）
# TODO 8. 似乎比较好的实现，比如sklearn，在内部节点里是不放样本的！

import numpy as np
import copy

class KDTree:
    class Node:
        def __init__(self,axis, point, left, right, box):
            self.parent = None
            self.axis = axis
            self.point = point
            self.left = left
            self.right = right
            self.box = box

    class BoundingBox:
        def __init__(self, n_dim):
            self.n_dim = n_dim
            self.lower = np.zeros(self.n_dim) - np.inf
            self.upper = np.zeros(self.n_dim) + np.inf

        # 生成一份copy，并且重新设置一下axis轴的坐标范围
        def trimLeft(self, axis, upper):
            # new_box = KDTree.BoundingBox(self.n_dim)
            new_box = copy.deepcopy(self)
            new_box.upper[axis] = upper

            return new_box

        # 生成一份copy，并且重新设置一下axis轴的坐标范围
        def trimRight(self, axis, lower):
            # new_box = KDTree.BoundingBox(self.n_dim)
            new_box = copy.deepcopy(self)
            new_box.lower[axis] = lower

            return new_box


        # 判断是否与以x为中心，radius为半径的圆相交
        def intersect(self, x, radius):
            distance = self._compute_distance(x)
            return distance < radius

        def _compute_distance(self, x):
            distance = 0
            for i in range(self.n_dim):
                distance += np.square(self._1d_distance(x[i], self.lower[i], self.upper[i]))
            distance = np.sqrt(distance)
            return distance

        def _1d_distance(self, x_val, lower, upper):
            if x_val < lower:
                return x_val - lower
            elif x_val > upper:
                return upper - x_val
            else:
                return 0.0

    def __init__(self, X, leaf_size=2):
        self.X = X
        self.X = X
        self.n_samples, self.n_features = X.shape
        # self.indices = np.arange(0, self.n_samples)
        self.leaf_size = leaf_size  # 暂时先不用
        self.root = self._build_tree(np.arange(0, self.n_samples), 0, KDTree.BoundingBox(self.n_features))

    # 就先不用样本索引值而直接使用样本
    def _build_tree(self, X_ind, depth, box):
        # 终止条件：如果X为空的话，返回一个None
        # 也可以进一步地使用leaf_size条件
        if X_ind.shape[0] == 0:
            return None
        # 确定划分轴
        axis = depth % self.n_features
        # 划分数据集X
        X_ind_f = self.X[:,axis][X_ind] # 先取列，再取行，开销小，因为在这里X_ind是不规则的
        sorted_ind = X_ind_f.argsort()
        sorted_X_ind = X_ind[sorted_ind]
        median = int(sorted_X_ind.shape[0] / 2)
        median_X_ind = sorted_X_ind[median]
        left_X_ind = sorted_X_ind[:median]
        right_X_ind = sorted_X_ind[median+1:]


        '''
        # kdtree v3.0
        # “假”/“松”的边界盒
        cut_val = self.X[median_X_ind, axis]
        lower = upper = cut_val
        if median > 0:
            lower = self.X[sorted_X_ind[median-1], axis]
        if median+1 < len(X_ind):
            upper = self.X[sorted_X_ind[median+1], axis]
        # 构建左子树
        left = self._build_tree(left_X_ind, depth + 1, box.trimLeft(axis, lower))
        # 构建右子树
        right = self._build_tree(right_X_ind, depth + 1,  box.trimRight(axis, upper))
        '''

        # kdtree v4.0
        # "真"/"紧"的边界盒
        if len(left_X_ind) != 0:
            left_X = self.X[left_X_ind]
            left_box = KDTree.BoundingBox(self.n_features)
            left_box.lower = np.min(left_X, axis=0)
            left_box.upper = np.max(left_X, axis=0)
            # 构建左子树
            left = self._build_tree(left_X_ind, depth + 1, left_box)
        else:
            left = None

        if len(right_X_ind) != 0:
            right_X = self.X[right_X_ind]
            right_box = KDTree.BoundingBox(self.n_features)
            right_box.lower = np.min(right_X, axis=0)
            right_box.upper = np.max(right_X, axis=0)
            # 构建右子树
            right = self._build_tree(right_X_ind, depth + 1, right_box)
        else:
            right = None

        # 构建父节点
        node = self.Node(axis, median_X_ind, left, right, box)
        if left != None:
            left.parent = node
        if right != None:
            right.parent = node

        return node

    # 寻找距离最近的n个样本
    def kd_nearest_n(self, x, n):
        self.n_nearest_neighbors = []
        self.n_distances = []
        self._kd_nearest_n(self.root, x, n,1)
        return self.n_nearest_neighbors

    def _kd_nearest_n(self, node, x, n,depth):
        if node==None:
            return

        # 使用节点的point来更新n_nearest_neighbors和n_distances
        cur_distance = np.linalg.norm(x-self.X[node.point])
        # cur_distance = np.sum(np.square(x-self.X[node.point]))
        if len(self.n_nearest_neighbors) < n:
            self.n_nearest_neighbors.append(node.point)
            self.n_distances.append(cur_distance)
        else:
            max_distance_ind = int(np.argmax(self.n_distances))
            max_distance = self.n_distances[max_distance_ind]
            if max_distance > cur_distance:
                self.n_nearest_neighbors[max_distance_ind] = node.point
                self.n_distances[max_distance_ind] = cur_distance

        # 先比较x[axis]与node.point[axis] 确定首先搜索的子树
        axis = node.axis
        if(x[axis] < self.X[node.point,axis]):
            cur = node.left
            sibling = node.right
        else:
            cur = node.right
            sibling = node.left

        self._kd_nearest_n(cur,x,n,depth+1)

        '''
        # 然后如果n_nearest_neighbors不是满的，或者另一颗子树的切分点的值满足半径约束，则继续搜索另一棵子树
        if len(self.n_nearest_neighbors) < n or (np.max(self.n_distances) > abs(x[axis]-self.X[node.point,axis])):
            self._kd_nearest_n(sibling,x,n,depth+1)
        # 只判断分割超平面的“剪枝”方法不太行！因为“维数诅咒”，高维空间的欧式距离会远远大于某个坐标上的距离，导致过滤掉的点不多!!!
        '''

        '''使用边界盒进行判断会比较好'''
        if sibling!=None and (len(self.n_nearest_neighbors) < n or sibling.box.intersect(x, np.max(self.n_distances))):
            self._kd_nearest_n(sibling,x,n,depth+1)





