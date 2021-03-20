import numpy as np
from time import time
from random import randint
import heapq
import math

def gen_data(low, high, n_rows, n_cols=None):
    """Generate dataset randomly.
    Arguments:
        low {int} -- The minimum value of element generated.
        high {int} -- The maximum value of element generated.
        n_rows {int} -- Number of rows.
        n_cols {int} -- Number of columns.
    Returns:
        list -- 1d or 2d list with int
    """
    if n_cols is None:
        ret = [randint(low, high) for _ in range(n_rows)]
    else:
        ret = [[randint(low, high) for _ in range(n_cols)]
               for _ in range(n_rows)]
    return ret

def compute_distance_with_list(X1, X2):
    """Calculate the Euclidean distance.
    Arguments:
        X1 {list}
        X2 {list}
    Returns:
        float
    """
    
    s = 0
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
    
    s = 0
    for i in range(X1.shape[0]):
        s += (X1[i] - X2[i]) ** 2
    
    return math.sqrt(s)

class KDNode(object):
    def __init__(
            self, feature,
            left=None, right=None,
            axis_no=0, depth=0
    ):
        # 每个节点的特征值
        self.feature = feature
        # 节点的左孩子
        self.left = left
        # 节点的右孩子
        self.right = right
        # 划分维度编号
        self.axis_no = axis_no
        # 节点所在的深度
        self.depth = depth

class KDTree(object):
    def __init__(self, dimensions):
        # 根节点
        self.root = None
        # 记录维度
        self.dimensions = dimensions
        # 用于记录最近的K个节点
        self.k_neighbour = []

    def create(self, feature_dataset, depth):
        '''
        Function:
        ----------
        构造KD树
        
        Parameters
        ----------
        depth : int
                当前构造节点的深度
        dataset : numpy.ndarray
                只包含特征的矩阵
                
        Returns : KDNode
        -------
                当前树的根节点
        
        Notes
        -------
        1. 如果数据集中只有一条数据，则赋予空的叶子节点
        2. 如果不止一条数据，则进行如下操作：
            a. 根据构造树当前的深度，选定划分轴（根据哪个特征进行划分）
            b. 根据划分轴（该特征），对数据集按照该特征从小到大排序
            c. 选出中位数、排序特征中大于、小于中位数的子数据集
            d. 递归调用自身，构造KDTree
        '''
        
        num_samples = feature_dataset.shape[0]
        if num_samples < 1:
            return None
        if num_samples == 1:
            new_node = KDNode(
                feature_dataset[0],
                depth=depth
            )
        else:
            # 获取分隔坐标轴编号
            axis_no = depth % self.dimensions
            
            # 获取按axis_no列排序后的矩阵
            sort_index = np.argsort(feature_dataset, axis=0)
            sorted_dataset = feature_dataset[sort_index[:, axis_no]]
            
            # 获取第 axis_no 轴的中位数
            median_no = num_samples // 2
            # 获取需要设置在左子树的数据集及标签
            left_dataset = sorted_dataset[:median_no, :]
            # 获取需要设置在右子树的数据集及标签
            right_dataset = sorted_dataset[median_no + 1:, :]
            # 构造KDTree的节点
            new_node = KDNode(
                sorted_dataset[median_no, :],
                axis_no=axis_no,
                depth=depth
            )
            # 构造左子树与右子树
            new_node.left = self.create(
                left_dataset,
                depth + 1
            )
            new_node.right = self.create(
                right_dataset,
                depth + 1
            )
        return new_node

    def KDTree_NN(self, node: KDNode, target: np.ndarray, k: int):
        '''
        Function:
        ----------
        搜索KD树
    
        Parameters
        ----------
            node: 根节点
            target: 目标值
            
        Returns
        -------
            找到距离目标最近的K个值
        
        Notes
        -------
        '''
        
        if k < 1:
            raise ValueError("k must be greater than 0.")
        else:
            if node is None:
                raise ValueError("KDTree is None.")
            else:
                if target.shape[0] != self.dimensions:
                    raise ValueError("target node's dimension unmatched KDTree's dimension")
                else:
                    self.k_neighbour = []
                    self._KDTree_NN(node, target, k)

    def _KDTree_NN(self, node: KDNode, target: np.ndarray, k: int):
        if node is None:
            return
        else:
            if target[node.axis_no] <= node.feature[node.axis_no]:
                self._KDTree_NN(node.left, target, k)
                dis = compute_distance_with_array(node.feature, target)
                if len(self.k_neighbour) < k:
                    self.k_neighbour.append(dis)
                    self.k_neighbour.sort()
                else:
                    if dis < self.k_neighbour[-1]:
                        self.k_neighbour[-1] = dis
                        self.k_neighbour.sort()
                
                if len(self.k_neighbour) < k:
                    self._KDTree_NN(node.right, target, k)
                else:
                    split_dis = abs(target[node.axis_no] - node.feature[node.axis_no])
                    if split_dis < self.k_neighbour[-1]:
                        self._KDTree_NN(node.right, target, k)
            else:
                self._KDTree_NN(node.right, target, k)
                dis = compute_distance_with_array(node.feature, target)
                if len(self.k_neighbour) < k:
                    self.k_neighbour.append(dis)
                    self.k_neighbour.sort()
                else:
                    if dis < self.k_neighbour[-1]:
                        self.k_neighbour[-1] = dis
                        self.k_neighbour.sort()
                
                if len(self.k_neighbour) < k:
                    self._KDTree_NN(node.left, target, k)
                else:
                    split_dis = abs(target[node.axis_no] - node.feature[node.axis_no])
                    if split_dis < self.k_neighbour[-1]:
                        self._KDTree_NN(node.left, target, k)
            return

def exhausted_search(X, Xi):
    dist_best = float('inf')
    row_best = None
    for row in X:
        dist = get_euclidean_distance(Xi, row)
        if dist < dist_best:
            dist_best = dist
            row_best = row
    return row_best

if __name__ == '__main__':
    start_all = time()
    print("Testing KD Tree...")
    test_times = 1
    run_time_1 = run_time_2 = 0
    for _ in range(test_times):
        low = 0
        high = 100
        n_rows = 1000
        n_cols = 2
        X = gen_data(low, high, n_rows, n_cols)
        X_array = np.array(X)
        
        start = time()
        tree = KDTree(n_cols)
        tree.root = tree.create(X_array, 0)
        print("build_tree cost time:", time() - start)
        
        search_times = 1000
        for _ in range(search_times):
            Xi = gen_data(low, high, n_cols)
            Xi_array = np.array(Xi)
            start = time()
            tree.KDTree_NN(tree.root, Xi_array, 10)
            run_time_1 += time() - start
            ret1 = sum(tree.k_neighbour)
            start = time()
            pts_dis = np.zeros((X_array.shape[0]))
            for i in range(X_array.shape[1]):
                pts_dis += (X_array[:, i] - Xi_array[i]) ** 2
            pts_dis = np.sqrt(pts_dis)
            neighborhood_dis = np.sort(pts_dis)[:10]
            run_time_2 += time() - start
            ret2 = neighborhood_dis.sum()
            
            assert round(ret1, 5) == round(ret2, 5), print("\nX", X, "\nXi", Xi, "\nret1", ret1, "\nret2", ret2)
    
    print()
    print("%d tests passed!" % test_times)
    print("Total time %.3f s" % (time() - start_all))
    
    print("KD Tree Search %.3f s" % run_time_1)
    print("Exhausted search %.3F s" % run_time_2)


