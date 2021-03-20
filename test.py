from time import time
import numpy as np
from scipy import spatial
import math

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

a = np.array([50, 45])
b = np.array([12, 24])
#~ a = [50, 45]
#~ b = [12, 24]

start = time()
for _ in range(10000):
    #~ c = ((a - b) ** 2).sum() ** 0.5
    #~ np.sqrt(np.sum((a - b) ** 2))
    #~ sum([(a[i] - b[i]) ** 2 for i in range(len(a))]) ** 0.5
    #~ c = math.sqrt(sum([(a[i] - b[i]) ** 2 for i in range(len(a))]))
    #~ sum(map(lambda x: (a[x] - b[x]) ** 2, range(len(a)))) ** 0.5
    #~ c = spatial.distance.euclidean(a, b)
    #~ c = np.linalg.norm(a - b)
    #~ c = compute_distance_with_array(a, b)
    #~ c = compute_distance_with_list(a, b)

print("cost time:", time() - start)
#~ print(c)
