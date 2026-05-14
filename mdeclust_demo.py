"""
============================================================================
MDEClust Demo: Sequential and Parallel Clustering Experiment

Runs the Memetic Differential Evolution Clustering (MDEClust) algorithm
and compares its sequential implementation based on [2] and the parallel
implementation proposed in [1] for solving the Euclidean Minimum
Sum-of-Squares Clustering (MSSC) problem, also known as the K-means 
clustering problem.

Dataset: Liver Disorders dataset (345 samples, 6 features)

Best known MSSC objective value for the Liver Disorders dataset with 20 clusters: 80,044.73

Default parameters for MDEClust used in this demo experiment are chosen according to [2]:

k = 20                   Number of clusters
population_size = 150    Number of complete clustering solutions used to form the population
tol = 0.0001             Diversity threshold for the stopping criterion
nmax = 5000              Maximum number of consecutive non-improving iterations
matching_mode = 0        Center matching method: 0 = Hungarian, 1 = greedy
mutation = False         Enable/disable mutation operator
alpha = 0.5              Mutation bias parameter: 0 = uniform, 1 = greedy
n_attempts = 3           Number of attempts to select distinct parent solutions

Note: The first step performs Numba JIT compilation, which may take a noticeable 
amount of time. This overhead occurs only once per function signature at the start 
and does not affect the actual experiment timings.

Please cite the following paper if you use this code:

[1] Rustam Mussabayev, Ravil Mussabayev, Alexander Krassovitskiy, Irina Ualiyeva (2026). 
Parallel Memetic Differential Evolution for Minimum Sum-of-Squares Clustering.
In: Nguyen, N.T., et al. Advances in Computational Collective Intelligence. 
ICCCI 2025. Communications in Computer and Information Science, vol 2747.
Springer, Cham. https://doi.org/10.1007/978-3-032-10202-7_7

The MDEClust algorithm implemented here is based on the method described in:

[2] Pierluigi Mansueto, Fabio Schoen. Memetic differential evolution methods for 
clustering problems. Pattern Recognition, Volume 114, 2021, 107849
https://doi.org/10.1016/j.patcog.2021.107849

Programmed by Rustam Mussabayev (rmusab@gmail.com)
20 August 2022
============================================================================
"""

import csv
import time
from pathlib import Path

import numpy as np
import numba as nb

from mdeclust import mdeclust_sequential, mdeclust_parallel


def load_dataset(filename, delimiter, columns):
    with open(filename, newline='') as f1:
        reader = csv.reader(f1, delimiter=delimiter)
        raw = [row[columns] for row in reader]
    return np.array(raw, dtype=float)

    
def main():
       
    filename = Path(__file__).resolve().parent / 'liver_disorders.data'
    points = load_dataset(filename, ',', slice(0, 6)) # Load the sample UCI Liver Disorders dataset to be clustered
    
    n_clusters = 20 # Desired number of clusters
    
    # The standard parameter settings for the MDEClust algorithm, as recommended in the source paper [2]:
    nmax = 5000             # Maximum number of consecutive non-improving iterations
    population_size = 150   # Number of complete clustering solutions used to form the population
    tol = 0.0001            # Diversity threshold for stopping criterion
    matching_mode = 0       # Center matching method (0: Hungarian, 1: greedy)
    mutation = False        # Whether to enable the mutation operator
    alpha = 0.5             # Mutation bias parameter (0: uniform, 1: greedy)
    n_attempts = 3          # Number of attempts to select distinct parent solutions
    
    nb.set_num_threads(nb.config.NUMBA_NUM_THREADS) # Use the maximum available number of parallel threads
    # nb.set_num_threads(3) # Use a fixed number of parallel threads
    # nb.set_num_threads(1) # Run the parallel implementation with one thread (equivalent to sequential mode)

    n_threads = nb.get_num_threads()
    
    # Best Known Solution (for comparison)
    f_best = 80044.73

    print(__doc__)
    
    print("STEP 1: Numba JIT compilation warm-up")
    print("-------------------------------------")
    print("Compiling Numba functions. This one-time startup may take a while.")
    print()
    
    # Note: the first call includes Numba JIT compilation and may be noticeably slower.
    # This startup cost is paid only once per function signature.
    
    # Warm-up calls with the same argument types

    print("Compiling Sequential MDEClust...")
    mdeclust_sequential(points[:100], 3, 5, tol, 1, matching_mode, mutation, alpha, n_attempts, False)

    print("Compiling Parallel MDEClust...")
    mdeclust_parallel(points[:100], 3, 5, tol, 1, matching_mode, mutation, alpha, n_attempts, False)
    
    print()
    print("Warm-up compilation completed.")
    print()
    
    print("STEP 2: Actual execution of experiments")
    print("---------------------------------------")
    print()
    
    print('SEQUENTIAL MDECLUST:')
    print()
    start = time.perf_counter()
    
    objective, centers, assignment, n_dists, n_execs, best_time, best_n_execs, best_n_dists = mdeclust_sequential(points, n_clusters, population_size, tol, nmax, matching_mode, mutation, alpha, n_attempts, True)
    
    elapsed_time = time.perf_counter() - start
    print()
    print('Final Objective: ', objective)
    objective_gap = round((objective - f_best) / f_best * 100, 2)
    print('Objective Gap: ', objective_gap, '%')
    
    print('#K-means executions to Best Solution: ', best_n_execs)
    print('#Distance Calculations to Best Solution: ', best_n_dists)
    
    print(f"Time to Best Solution: {best_time:.4f} s")
    print(f"Total Running Time: {elapsed_time:.4f} s")
    
    print('#Total K-means executions: ', n_execs)
    print('#Total Distance Calculations: ', n_dists)
    print()
    print()
    print()
    
    
    print("PARALLEL MDECLUST:")
    print()
    print('Number of parallel threads used: ', n_threads)
    print()
    start = time.perf_counter()
    
    objective, centers, assignment, n_dists, n_execs, best_time, best_n_execs, best_n_dists = mdeclust_parallel(points, n_clusters, population_size, tol, nmax, matching_mode, mutation, alpha, n_attempts, True)
    
    elapsed_time = time.perf_counter() - start
    print()
    print('Final Objective: ', objective)
    objective_gap = round((objective - f_best) / f_best * 100, 2)
    print('Objective Gap: ', objective_gap, '%')
    
    print('#K-means executions to Best Solution: ', best_n_execs)
    print('#Distance Calculations to Best Solution: ', best_n_dists)
    
    print(f"Time to Best Solution: {best_time:.4f} s")
    print(f"Total Running Time: {elapsed_time:.4f} s")
    
    print('#Total K-means executions: ', n_execs)
    print('#Total Distance Calculations: ', n_dists)
    print()


if __name__ == "__main__":
    main()