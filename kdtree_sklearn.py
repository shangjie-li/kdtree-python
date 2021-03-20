# -*- coding: UTF-8 -*-
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from sklearn.neighbors import KDTree
from time import time
from random import randint

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

#~ np.random.seed(0)
#~ points = np.random.random((100, 2))
#~ plt.scatter(points[:, 0], points[:, 1], c='b')
#~ print("points", points)
#~ tree = KDTree(points)
#~ point = points[0]
#~ print()

# 函数一：给定数量临近点搜索
#~ dists, indices = tree.query([point], k=10)
#~ print(dists, indices)

# 函数二：给定半径临近点搜索
#~ indices = tree.query_radius([point], r=0.2)
#~ indices = indices[0]
#~ print(indices)

#~ plt.scatter(points[indices, 0], points[indices, 1], c='r')
#~ plt.scatter(point[0], point[1], c='k')
#~ plt.show()

# 函数三：核密度估计
#~ density = tree.kernel_density([point], h=0.1)
#~ print(density)

def main():
    """Example usage"""
    
    start_all = time()
    print("Testing KD Tree...")
    test_times = 1
    run_time_1 = run_time_2 = run_time_3 = 0
    for _ in range(test_times):
        low = 0
        high = 100
        n_rows = 1000
        n_cols = 2
        X = gen_data(low, high, n_rows, n_cols)
        X_array = np.array(X)
        
        start = time()
        tree = KDTree(X_array)
        print("build_tree cost time:", time() - start)
        
        search_times = 1000
        for _ in range(search_times):
            Xi = gen_data(low, high, n_cols)
            Xi_array = np.array([Xi])
            start = time()
            dists, indices = tree.query(Xi_array, k=10)
            #~ indices = tree.query_radius(Xi_array, r=5)
            #~ density = tree.kernel_density(Xi_array, h=0.1)
            run_time_1 += time() - start
            ret1 = dists.sum()
            
            start = time()
            pts_dis = np.zeros((X_array.shape[0]))
            for i in range(X_array.shape[1]):
                pts_dis += (X_array[:, i] - Xi_array[0, i]) ** 2
            pts_dis = np.sqrt(pts_dis)
            neighborhood_dis = np.sort(pts_dis)[:10]
            run_time_2 += time() - start
            ret2 = neighborhood_dis.sum()
            
            assert round(ret1, 5) == round(ret2, 5), print("ret1", ret1, "ret2", ret2)
    print()
    print("%d tests passed!" % test_times)
    print("Total time %.3f s" % (time() - start_all))
    
    print("KD Tree Search %.3f s" % run_time_1)
    print("Array search %.3f s" % run_time_2)

if __name__ == '__main__':
    main()

