from collections import namedtuple
from operator import itemgetter
from pprint import pformat
from time import time
from random import randint
import numpy as np
import math

def compute_distance_with_list(X1, X2):
    """Calculate the Euclidean distance.
    Arguments:
        X1 {list}
        X2 {list}
    Returns:
        float
    """
    
    cdef float s = 0
    for i in range(len(X1)):
        s += (X1[i] - X2[i]) ** 2
    
    return math.sqrt(s)

def compute_distance_with_array(X1, X2):
    """Calculate the Euclidean distance.
    Arguments:
        X1 {ndarray}
        X2 {ndarray}
    Returns:
        float
    """
    
    cdef float s = 0
    for i in range(X1.shape[0]):
        s += (X1[i] - X2[i]) ** 2
    
    return math.sqrt(s)

class Node(namedtuple('Node', 'split_axis median_point left_node right_node')):
    # namedtuple(具名元组)构造一个带字段名的元组
    # 具名元组的实例和普通元组消耗的内存一样多，因为字段名都被存在对应的类里面
    # 这个类跟普通的对象实例比起来也要小一些，因为Python不会用__dict__来存放这些实例的属性
    # 使用方法collections.namedtuple(typename, field_names, verbose=False, rename=False)
    #   typename：元组名称 
    #   field_names: 元组中元素的名称
    #   rename: 如果元素名称中含有 python 的关键字，则必须设置为 rename=True 
    #   verbose: 保持默认即可 
    def __repr__(self):
        return pformat(tuple(self))

class KDTree(object):
    def __init__(self):
        print("KDTree object created.")

    def _get_variance(self, X, feature):
        """
        计算特征的方差
        D(X) = E(X^2)-[E(X)]^2
        """
        cdef int n = len(X)
        cdef int col_sum = 0
        cdef int col_sum_sqr = 0
        for i in range(n):
            xi = X[i][feature]
            col_sum += xi
            col_sum_sqr += xi ** 2
        return col_sum_sqr / n - (col_sum / n) ** 2
    
    def _choose_feature(self, X):
        """
        选择特征
        取方差最大的特征作为分割点特征
        """
        cdef int m = len(X[0])
        variances = map(lambda j: (
            j, self._get_variance(X, j)), range(m))
        return max(variances, key=lambda x: x[1])[0]
    
    def create(self, dataset, var_switch=False):
        # 是否根据方差选取特征
        self.var_switch = var_switch
        # 根节点
        self.root = self.createTree(dataset, depth=0)
    
    def createTree(self, point_list, depth=0):
        try:
            n = len(point_list[0]) # assumes all points have the same dimension
        except IndexError as e:
            return None
        # Select axis based on depth so that axis cycles through all valid values
        if self.var_switch:
            axis = self._choose_feature(point_list)
        else:
            axis = depth % n
    
        # Sort point list and choose median as pivot element
        point_list.sort(key=itemgetter(axis))
        median = len(point_list) // 2 # choose median
    
        # Create node and construct subtrees
        return Node(
            split_axis=axis,
            median_point=point_list[median],
            left_node=self.createTree(point_list[:median], depth + 1),
            right_node=self.createTree(point_list[median + 1:], depth + 1)
        )
    
    def search(self, tree, target, k):
        # 用于记录最近的K个节点
        self.k = k
        self.k_neighbour = []
        self.searchTree(tree.root, target)
    
    def searchTree(self, tree, target):
        if tree is None:
            return
        else:
            split_axis = tree[0]
            median_point = tree[1]
            if target[split_axis] <= median_point[split_axis]:
                self.searchTree(tree[2], target)
                dis = compute_distance_with_list(median_point, target)
                if len(self.k_neighbour) < self.k:
                    self.k_neighbour.append(dis)
                    self.k_neighbour.sort()
                else:
                    if dis < self.k_neighbour[-1]:
                        self.k_neighbour[-1] = dis
                        self.k_neighbour.sort()
                if len(self.k_neighbour) < self.k:
                    self.searchTree(tree[3], target)
                else:
                    split_dis = abs(target[split_axis] - median_point[split_axis])
                    if split_dis < self.k_neighbour[-1]:
                        self.searchTree(tree[3], target)
            else:
                self.searchTree(tree[3], target)
                dis = compute_distance_with_list(median_point, target)
                if len(self.k_neighbour) < self.k:
                    self.k_neighbour.append(dis)
                    self.k_neighbour.sort()
                else:
                    if dis < self.k_neighbour[-1]:
                        self.k_neighbour[-1] = dis
                        self.k_neighbour.sort()
                if len(self.k_neighbour) < self.k:
                    self.searchTree(tree[2], target)
                else:
                    split_dis = abs(target[split_axis] - median_point[split_axis])
                    if split_dis < self.k_neighbour[-1]:
                        self.searchTree(tree[2], target)
            return

