from random import randint
from time import time
import numpy as np

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

def get_euclidean_distance(X1, X2):
    """Calculate the Euclidean distance.
    Arguments:
        X1 {list}
        X2 {list}
    Returns:
        list
    """
    
    #~ return ((arr1 - arr2) ** 2).sum() ** 0.5
    
    # Added by shangjie.
    return sum([(X1[i] - X2[i]) ** 2 for i in range(len(X1))])

def min_max_scale(X):
    """Scale the element of X into an interval [0, 1].
    Arguments:
        X {ndarray} -- 2d array object with int or float
    Returns:
        ndarray -- 2d array object with float
    """

    #~ X_max = X.max(axis=0)
    #~ X_min = X.min(axis=0)
    
    #~ return (X - X_min) / (X_max - X_min)
    
    # Added by shangjie.
    X_max = [- float("inf") for _ in range(len(X[0]))]
    for row in range(len(X)):
        for col in range(len(X[0])):
            if X[row][col] > X_max[col]:
                X_max[col] = X[row][col]
    
    X_min = [float("inf") for _ in range(len(X[0]))]
    for row in range(len(X)):
        for col in range(len(X[0])):
            if X[row][col] < X_min[col]:
                X_min[col] = X[row][col]
    
    X_scale = [[(X[row][col] - X_min[col]) / (X_max[col] - X_min[col]) for col in range(len(X[0]))] for row in range(len(X))]
    
    return X_scale
    
class Node(object):
    """创建Node类
    """
    
    # 初始化，存储父节点、左节点、右节点、特征及分割点
    def __init__(self):
        self.father = None
        self.left = None
        self.right = None
        self.feature = None
        self.split = None
    
    # 获取Node的各个属性
    def __str__(self):
        return "feature: %s, split: %s" % (str(self.feature), str(self.split))
    
    # 获取Node的兄弟节点
    def brother(self):
        if self.father is None:
            ret = None
        else:
            if self.father.left is self:
                ret = self.father.right
            else:
                ret = self.father.left
        return ret
    
class KDTree(object):
    """创建KDTree类
    """
    
    # 初始化，存储根节点
    def __init__(self):
        self.root = Node()
    
    # 获取KDTree属性
    def __str__(self):
        ret = []
        i = 0
        que = [(self.root, -1)]
        while que:
            nd, idx_father = que.pop(0)
            ret.append("%d -> %d: %s" % (idx_father, i, str(nd)))
            if nd.left is not None:
                que.append((nd.left, i))
            if nd.right is not None:
                que.append((nd.right, i))
            i += 1
        return "\n".join(ret)
    
    # 获取数组中位数的下标
    def _get_median_idx(self, X, idxs, feature):
        n = len(idxs)
        k = n // 2
        col = map(lambda i: (i, X[i][feature]), idxs)
        sorted_idxs = map(lambda x: x[0], sorted(col, key=lambda x: x[1]))
        median_idx = list(sorted_idxs)[k]
        return median_idx
    
    # 计算特征的方差
    # D(X) = E(X^2)-[E(X)]^2
    def _get_variance(self, X, idxs, feature):
        n = len(idxs)
        col_sum = col_sum_sqr = 0
        for idx in idxs:
            xi = X[idx][feature]
            col_sum += xi
            col_sum_sqr += xi ** 2
        return col_sum_sqr / n - (col_sum / n) ** 2
    
    # 选择特征
    # 取方差最大的特征作为分割点特征
    def _choose_feature(self, X, idxs):
        m = len(X[0])
        variances = map(lambda j: (
            j, self._get_variance(X, idxs, j)), range(m))
        return max(variances, key=lambda x: x[1])[0]
    
    # 分割特征
    # 把大于、小于中位数的元素分别放到两个列表中
    def _split_feature(self, X, idxs, feature, median_idx):
        idxs_split = [[], []]
        split_val = X[median_idx][feature]
        for idx in idxs:
            if idx == median_idx:
                continue
            xi = X[idx][feature]
            if xi < split_val:
                idxs_split[0].append(idx)
            else:
                idxs_split[1].append(idx)
        return idxs_split
    
    # 建立KDTree
    # 使用广度优先搜索的方式，注意要对X进行归一化
    def build_tree(self, X, y):
        X_scale = min_max_scale(X)
        nd = self.root
        idxs = range(len(X))
        que = [(nd, idxs)]
        while que:
            nd, idxs = que.pop(0)
            n = len(idxs)
            if n == 1:
                nd.split = (X[idxs[0]], y[idxs[0]])
                continue
            feature = self._choose_feature(X_scale, idxs)
            median_idx = self._get_median_idx(X, idxs, feature)
            idxs_left, idxs_right = self._split_feature(X, idxs, feature, median_idx)
            nd.feature = feature
            nd.split = (X[median_idx], y[median_idx])
            if idxs_left != []:
                nd.left = Node()
                nd.left.father = nd
                que.append((nd.left, idxs_left))
            if idxs_right != []:
                nd.right = Node()
                nd.right.father = nd
                que.append((nd.right, idxs_right))
    
    # 搜索辅助函数
    # 比较目标元素与当前结点的当前feature，访问对应的子节点
    # 反复执行上述过程，直到到达叶子节点
    def _search(self, Xi, nd):
        while nd.left or nd.right:
            if nd.left is None:
                nd = nd.right
            elif nd.right is None:
                nd = nd.left
            else:
                if Xi[nd.feature] < nd.split[0][nd.feature]:
                    nd = nd.left
                else:
                    nd = nd.right
        return nd
    
    # 欧氏距离
    # 计算目标元素与某个节点的欧氏距离
    # get_euclidean_distance没有进行开根号的操作，所以求出来的是欧氏距离的平方
    def _get_eu_dist(self, Xi, nd):
        X0 = nd.split[0]
        return get_euclidean_distance(Xi, X0)
    
    # 超平面距离
    # 计算目标元素与某个节点所在超平面的欧氏距离，为保持一致而加上平方
    def _get_hyper_plane_dist(self, Xi, nd):
        j = nd.feature
        X0 = nd.split[0]
        return (Xi[j] - X0[j]) ** 2
    
    # 搜索函数
    # 搜索KDTree中与目标元素距离最近的节点，使用广度优先搜索
    def nearest_neighbour_search(self, Xi):
        dist_best = float("inf")
        nd_best = self._search(Xi, self.root)
        que = [(self.root, nd_best)]
        while que:
            nd_root, nd_cur = que.pop(0)
            while 1:
                dist = self._get_eu_dist(Xi, nd_cur)
                if dist < dist_best:
                    dist_best = dist
                    nd_best = nd_cur
                if nd_cur is not nd_root:
                    nd_bro = nd_cur.brother()
                    if nd_bro is not None:
                        dist_hyper = self._get_hyper_plane_dist(
                            Xi, nd_cur.father)
                        if dist > dist_hyper:
                            _nd_best = self._search(Xi, nd_bro)
                            que.append((nd_bro, _nd_best))
                    nd_cur = nd_cur.father
                else:
                    break
        return nd_best
    
# 为便于对比，添加一个线型搜索函数
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
        y = gen_data(low, high, n_rows)
        
        tree = KDTree()
        start = time()
        tree.build_tree(X, y)
        print("build_tree cost time:", time() - start)
        
        search_times = 1000
        for _ in range(search_times):
            Xi = gen_data(low, high, n_cols)
            start = time()
            nd = tree.nearest_neighbour_search(Xi)
            run_time_1 += time() - start
            ret1 = get_euclidean_distance(Xi, nd.split[0])
            start = time()
            row = exhausted_search(X, Xi)
            run_time_2 += time() - start
            ret2 = get_euclidean_distance(Xi, row)
    
            assert ret1 == ret2, "target:%s\nrestult1:%s\nrestult2:%s\ntree:\n%s" \
                % (str(Xi), str(nd), str(row), str(tree))
    print()
    print("%d tests passed!" % test_times)
    print("Total time %.2f s" % (time() - start_all))
    
    print("KD Tree Search %.2f s" % run_time_1)
    print("Exhausted search %.2f s" % run_time_2)
    
