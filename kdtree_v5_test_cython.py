from time import time
from random import randint
import numpy as np
from kdtree_v5_cython import KDTree

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
        tree = KDTree()
        tree.create(X, var_switch=False)
        print("build_tree cost time:", time() - start)
        
        search_times = 1000
        for _ in range(search_times):
            Xi = gen_data(low, high, n_cols)
            Xi_array = np.array(Xi)
            start = time()
            tree.search(tree, Xi, 10)
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
            
            assert round(ret1, 5) == round(ret2, 5), print("ret1", ret1, "ret2", ret2)
    print()
    print("%d tests passed!" % test_times)
    print("Total time %.3f s" % (time() - start_all))
    
    print("KD Tree Search %.3f s" % run_time_1)
    print("Array search %.3f s" % run_time_2)
    
    #~ print()
    #~ print(tree.root)

if __name__ == '__main__':
    main()
