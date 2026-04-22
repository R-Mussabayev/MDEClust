# Parallel and non-parallel (standard) implementation of
# Memetic Differential Evolution (MDEClust) algorithm [1]
# for the Euclidean Minimum Sum-of-Squares Clustering (MSSC) problem
# Programmed by Rustam Mussabayev (rmusab@gmail.com)
# 20 August 2022

# Original article with nonparallel MDEClust algorithm description:
# [1] Pierluigi Mansueto, Fabio Schoen. Memetic differential evolution methods for clustering problems. Pattern Recognition, Volume 114, 2021, 107849
# https://doi.org/10.1016/j.patcog.2021.107849
#
# The parallel version of MDEClust algorithm is descrided in the following paper (If you use this code, please cite):
# [2] Mussabayev, R., Mussabayev, R., Krassovitskiy, A., Ualiyeva, I. (2026). Parallel Memetic Differential Evolution for Minimum Sum-of-Squares Clustering.
# In: Nguyen, N.T., et al. Advances in Computational Collective Intelligence. ICCCI 2025. Communications in Computer and Information Science, vol 2747.
# Springer, Cham. https://doi.org/10.1007/978-3-032-10202-7_7


from mdeclust import mdeclust_nonparallel, mdeclust_parallel
import math
import csv
import numpy as np
import numba as nb

def load_dataset(filename, delimiter, columns):
    with open(filename, newline='') as f1:
        reader = csv.reader(f1, delimiter=delimiter)
        raw = [row[columns] for row in reader]
    return np.array(raw, dtype=float)

points = load_dataset('liver_disorders.data', ',', slice(0, 6)) # Dataset to be clustered
n_clusters = 20 # Desired number of clusters

# The standard parameter settings for the MDEClust algorithm, as recommended in the source paper [1]:
nmax = 5000             # Maximum number of consecutive non-improving iterations
population_size = 150   # Population size
tol = 0.0001            # Diversity threshold for stopping criterion
matching_mode = 0       # Center matching method (0: Hungarian, 1: greedy)
mutation = False        # Enable/disable mutation operator
alpha = 0.5             # Mutation bias parameter (0: uniform, 1: greedy)
n_attempts = 3          # Number of attempts to select distinct parent solutions

nb.set_num_threads(nb.config.NUMBA_NUM_THREADS) # Set the number of threads for parallel execution to the maximum possible
#nb.set_num_threads(3) # Set the number of threads for parallel execution to the some value
#nb.set_num_threads(1)

# Best Known Solution (for comparison)
f_best = 80044.73


print("="*60)
print("MDEClust Demo: Sequential vs Parallel Implementation")
print("="*60)
print("Problem: Minimum Sum-of-Squares Clustering (MSSC)")
print("Dataset: Liver Disorders (n=345, d=6)")
print(f"Number of clusters: {n_clusters}")
print()
print("This demo compares the standard (sequential) and parallel")
print("implementations of the Memetic Differential Evolution clustering algorithm.")
print()
print("Note: The first step performs Numba JIT compilation, which may take")
print("a noticeable amount of time. This overhead occurs only once.")
print("="*60)
print()


print("STEP 1: JIT compilation warm-up")
print()
print("Compiling Numba functions... this one-time startup may take a while.")
print()

# Note: the first call includes Numba JIT compilation and may be noticeably slower.
# This startup cost is paid only once per function signature.

# Warm-up calls with the same argument types
_ = mdeclust_nonparallel(points, n_clusters, population_size, tol,
                         1, matching_mode, mutation, alpha, n_attempts, False)

_ = mdeclust_parallel(points, n_clusters, population_size, tol,
                      1, matching_mode, mutation, alpha, n_attempts, False)

print("Compilation finished. Running the actual experiments.")
print()

print("STEP 2: Actual execution")
print()
print('SEQUENTIAL MDECLUST:')
print()
objective, centers, assignment, n_dists, n_execs, best_time, best_n_execs, best_n_dists = mdeclust_nonparallel(points, n_clusters, population_size, tol, nmax, matching_mode, mutation, alpha, n_attempts, True)
print('Full Objective: ', objective)
objective_gap = round((objective - f_best) / objective * 100, 2)
print('Objective Gap: ', objective_gap, '%')
print(f"Time to Best Solution: {best_time:.4f} s")
print()


print("PARALLEL MDECLUST:")
print()
objective, centers, assignment, n_dists, n_execs, best_time, best_n_execs, best_n_dists = mdeclust_parallel(points, n_clusters, population_size, tol, nmax, matching_mode, mutation, alpha, n_attempts, True)
print('Full Objective: ', objective)
objective_gap = round((objective - f_best) / objective * 100, 2)
print('Objective Gap: ', objective_gap, '%')
print(f"Time to Best Solution: {best_time:.4f} s")
print()


