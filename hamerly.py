# Python/Numba implementation of Hamerly's method for exact k-means acceleration using triangle inequality:
# G. Hamerly. Making k-means even faster. In SDM'10, SIAM International Conference on Data Mining, 2010, pp. 130-140.
# Original C++ source code: https://github.com/ghamerly/fast-kmeans
# Adopted for Python/Numba by Rustam Mussabayev (rmusab@gmail.com)
# 14 February 2023

import math
import numpy as np
from numba import njit

# INPUT PARAMETERS:
#
# points            - data points for clustering;
# centers           - initial cluster centers;
# max_iters         - maximum number of iterations. If max_iters <= 0 then there are 
#                     no any limitations on maximum number of iterations.
# use_inner_product - If True then advanced Euclidian distance re-casted in temms 
#                     of inner product is used which is more suitable for high dimensional 
#                     sparse data instead of simple one.
#                     the Euclidian distance re-casted in temms of inner product is used 
#
# OUTPUT: 
#
# centers           - final cluster centers;
# f                 - value of objective (the sum of the squared error);
# iterations        - total number of iterations;
# assignment        - the value of assignment[i] determines the index of assigned cluster for i-th point;
# numDistances      - total number of Euclidian distance calculations.

@njit
def hamerly_kmeans(points, centers, max_iters = -1, use_inner_product = True):   
    numDistances = 0
    def dist2(point1, point2):
        nonlocal numDistances
        numDistances += 1
        if use_inner_product:
            s1 = s2 = s3 = 0.0
            for i in range(point1.shape[0]):
                s1 += point1[i]*point1[i]
                s2 += point1[i]*point2[i]
                s3 += point2[i]*point2[i]             
            return s1 - 2*s2 + s3            
        else:
            d = 0.0
            for i in range(point1.shape[0]):
                d += (point1[i]-point2[i])**2
            return d            

    assert points.ndim == 2
    m, n = points.shape
    assert centers.ndim == 2 and centers.shape[1] == n
    k = centers.shape[0]
    assert m > 0 and n > 0 and k > 0       
    assignment = np.full(m, -1)
    sumNewCenters = np.zeros((k, n))
    clusterSize = np.zeros(k)
    s = np.zeros(k)
    centerMovement = np.full(k, np.inf)
    upper = np.full(m, np.inf)
    lower = np.zeros(m)        
    iterations = 0
    for i in range(m):
        closest = -1
        d = np.inf
        for j in range(k):
            d2 = dist2(points[i], centers[j])
            if d2 < d:
                closest = j
                d = d2
        assignment[i] = closest
        clusterSize[closest] += 1.0
        sumNewCenters[closest] += points[i]
    iterations += 1    
    if (iterations < max_iters) or (max_iters <= 0):
        for j in range(k):    
            if clusterSize[j] > 0.0:
                centers[j] = sumNewCenters[j] / clusterSize[j]
            else:
                centers[j] = np.full(n, np.nan)    
    while (iterations < max_iters) or (max_iters < 0):
        for i in range(k):
            s[i] = np.inf 
            for j in range(k):
                if i != j:
                    d2 = dist2(centers[i], centers[j])
                    if d2 < s[i]: s[i] = d2
            s[i] = math.sqrt(s[i]) / 2.0
        n_changed = 0
        for i in range(m):
            closest = assignment[i]
            upper_comparison_bound = max(s[closest], lower[i])
            if upper[i] > upper_comparison_bound:
                u2 = dist2(points[i], centers[closest])
                upper[i] = math.sqrt(u2)
                if upper[i] > upper_comparison_bound:
                    l2 = np.inf
                    for j in range(k):
                        if j != closest:
                            d2 = dist2(points[i], centers[j])
                            if d2 < u2:
                                l2 = u2
                                u2 = d2
                                closest = j
                            elif d2 < l2:
                                l2 = d2
                    lower[i] = math.sqrt(l2)
                    if assignment[i] != closest:
                        upper[i] = math.sqrt(u2)
                        oldAssignment = assignment[i]
                        clusterSize[assignment[i]] -= 1.0
                        clusterSize[closest] += 1.0
                        assignment[i] = closest
                        sumNewCenters[oldAssignment] -= points[i]
                        sumNewCenters[closest] += points[i]
                        n_changed += 1                        
        iterations += 1
        if ((max_iters > 0) and (iterations >= max_iters)) or (n_changed == 0):
            break
        for j in range(k):
            if clusterSize[j] > 0.0:
                z = sumNewCenters[j] / clusterSize[j]
            else:
                z = np.full(n, np.nan)
            centerMovement[j] = math.sqrt(dist2(z, centers[j]))
            centers[j] = z                
        # update bounds
        longest = centerMovement[0]
        secondLongest = centerMovement[1] if 1 < k else centerMovement[0]
        furthestMovingCenter = 0
        if longest < secondLongest:
            furthestMovingCenter = 1
            longest, secondLongest = secondLongest, longest
        for j in range(2,k):
            if longest < centerMovement[j]:
                secondLongest = longest
                longest = centerMovement[j]
                furthestMovingCenter = j
            elif secondLongest < centerMovement[j]:
                secondLongest = centerMovement[j]
        for i in range(m):
            upper[i] += centerMovement[assignment[i]]
            lower[i] -= secondLongest if assignment[i] == furthestMovingCenter else longest                                
    f = 0.0
    for i in range(m):
        f += dist2(points[i], centers[assignment[i]])
    return f, iterations, assignment, numDistances


# "Naive K-means" algorithm similar in functionality to the listed above "Hamerly K-means" algorithm
# Not for practical usage. It is relevant only for comparison with other clustering algorithms.
@njit
def naive_kmeans(points, centers, max_iters = -1, use_inner_product = True):
    def dist2(point1, point2):
        if use_inner_product:
            s1 = s2 = s3 = 0.0
            for i in range(point1.shape[0]):
                s1 += point1[i]*point1[i]
                s2 += point1[i]*point2[i]
                s3 += point2[i]*point2[i]             
            return s1 - 2*s2 + s3            
        else:
            d = 0.0
            for i in range(point1.shape[0]):
                d += (point1[i]-point2[i])**2
            return d   
    assert points.ndim == 2
    m, n = points.shape  
    assert (centers.ndim == 2) and (centers.shape[1] == n)
    k = centers.shape[0]
    assignment = np.full(m, -1)
    center_sums = np.empty((k, n))
    center_counts = np.zeros(k)
    f = np.inf
    n_iters = 0
    if (m > 0) and (n > 0) and (k > 0):
        while True:            
            f = 0.0 # assignment step
            n_changed = 0
            for i in range(m):
                min_d = np.inf
                min_ind = -1
                for j in range(k):
                    d = dist2(points[i], centers[j])
                    if d < min_d:
                        min_d = d
                        min_ind = j
                if assignment[i] != min_ind:
                    n_changed += 1
                    assignment[i] = min_ind
                f += min_d
            n_iters += 1
            if ((max_iters > 0) and (n_iters >= max_iters)) or (n_changed == 0):
                break
            for i in range(k): # update step
                center_counts[i] = 0.0
                for j in range(n):
                    center_sums[i,j] = 0.0
                    centers[i,j] = np.nan
            for i in range(m):
                center_ind = assignment[i]
                if center_ind > -1:
                    for j in range(n):
                        center_sums[center_ind,j] += points[i,j]
                    center_counts[center_ind] += 1.0
            for i in range(k):
                if center_counts[i] > 0.0:
                    for j in range(n):
                        centers[i,j] = center_sums[i,j] / center_counts[i]
    return f, n_iters, assignment, n_iters*k*m