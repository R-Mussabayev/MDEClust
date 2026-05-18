# MDEClust

The Memetic Differential Evolution Clustering (MDEClust) algorithm is an
effective metaheuristic approach for solving the Euclidean Minimum
Sum-of-Squares Clustering (MSSC) problem, capable of producing high-quality
clustering solutions for small and medium-scale datasets. The MSSC objective
corresponds to the classical K-means clustering criterion.

In addition to a sequential MDEClust implementation based on [2], this
repository includes a parallel implementation proposed in [1], which combines
Differential Evolution global search with K-means local search under a unified
parallel framework.

Experimental results reported in [1] demonstrate that the parallel
MDEClust implementation achieves substantial runtime reductions while
maintaining competitive clustering quality across benchmark datasets
compared with the sequential implementation.

This repository includes:

- Sequential MDEClust implementation based on [2]
- Parallel MDEClust implementation proposed in [1]
- Python/Numba implementation of the accelerated Hamerly K-means algorithm [3]
- Python/Numba implementation of the Hungarian (Kuhn-Munkres) algorithm based on [4]
- Example experiment script and sample dataset

Programmed by Rustam Mussabayev (rmusab@gmail.com)

Initial release: 20 August 2022

---

## Features

- Sequential and parallel MDEClust implementations
- Numba JIT acceleration
- Multi-threaded parallel execution
- Hamerly-accelerated exact K-means
- Pure-Numba Hungarian algorithm implementation
- Support for large-scale clustering experiments
- Detailed benchmarking statistics:
  - objective values
  - execution times
  - number of K-means executions
  - number of squared Euclidean distance calculations

---

## Main Idea

MDEClust is an advanced clustering framework designed to overcome one of the
main limitations of classical K-means clustering: its strong dependence on
initial cluster centers.

Like standard K-means, MDEClust aims to group similar data points together by
minimizing the total distance between data points and their assigned cluster
centers. However, while K-means usually relies on a single initialization and
can therefore converge to low-quality clustering results, MDEClust explores
many alternative clustering solutions simultaneously and continuously improves
them during the optimization process.

At its core, MDEClust acts as an intelligent optimization layer built around
the classical K-means algorithm. Instead of replacing K-means, it enhances it
with powerful global-search and evolutionary mechanisms that help discover
better clustering configurations. In many ways, MDEClust can be thought of as
an advanced evolutionary version of K-means — or, informally, "K-means on
steroids". The final clustering solutions remain fully compatible with standard
K-means and can also be used directly as high-quality initializations for
further refinement.

MDEClust combines several complementary algorithmic ideas:

- **Population-based search** maintains multiple clustering solutions at the
  same time instead of depending on a single initialization.
- **Differential Evolution recombination** generates new candidate solutions by
  combining information from several existing clustering solutions.
- **Cluster-center matching** aligns similar clusters between solutions before
  recombination, ensuring meaningful center combinations.
- **Mutation and repair mechanisms** preserve diversity and prevent invalid or
  degenerate clustering configurations.
- **K-means local refinement** improves promising candidate solutions and makes
  clusters more compact.
- **Hamerly's accelerated K-means** significantly reduces unnecessary distance
  calculations, making repeated refinement much more efficient.

Together, these components allow MDEClust to search broadly across many
different clustering possibilities while still retaining the fast local
optimization capability of K-means. This combination often produces clustering
results that are substantially more stable and higher quality than those
obtained from standard K-means alone.

---

## How MDEClust Works

MDEClust maintains a population of complete clustering solutions. Each solution
contains a full set of cluster centers and represents one possible way of
grouping the data.

The optimization process works as follows [2]:

1. Several different clustering solutions are first generated using random
   K-means initialization (Forgy-style initialization). Each solution is then
   refined using K-means to create the initial population.

2. For each solution in the population, three other clustering solutions are
   selected from the same population as parents for recombination. To preserve
   population diversity, solutions with identical cluster assignments are
   avoided whenever possible.

3. Before recombination, cluster centers from different parent solutions are
   aligned so that semantically similar clusters correspond to each other
   correctly. This prevents unrelated clusters from being combined and helps
   preserve meaningful clustering structure during evolutionary operations. The
   alignment is performed using either the exact Hungarian algorithm or a faster
   greedy matching strategy.


4. A new candidate clustering solution is generated using the Differential
   Evolution (DE) recombination rule:

   $$
   O = S_1 + F \cdot (S_2 - S_3)
   $$

   where $S_1$, $S_2$, and $S_3$ are aligned parent clustering solutions,
   $S_2 - S_3$ is the differential vector between two parent solutions,
   $F$ is a scaling factor controlling the strength of the variation,
   and $O$ is the newly generated offspring solution.

   After center alignment, corresponding cluster centers are treated as vectors
   and combined mathematically to generate new cluster-center positions. In this
   process, the differential vector $(S_2 - S_3)$ acts as a directional signal
   that guides the search toward new clustering configurations.

   Unlike purely random search, Differential Evolution transforms population
   diversity into an active optimization mechanism. Differences between existing
   clustering solutions define structured search directions, allowing MDEClust
   to explore the search space intelligently through controlled variation rather
   than random perturbation alone.

   As a result, each offspring solution inherits structural information from
   several parent clustering solutions simultaneously while also introducing new
   variations that may reveal better clustering structures.

5. Optionally, a mutation step slightly modifies one cluster center to preserve
   diversity and encourage exploration of new clustering configurations.

6. Invalid or empty clusters are automatically repaired when necessary to
   maintain valid clustering solutions.

7. The offspring solution is then refined using Hamerly's accelerated K-means
   algorithm, which improves clustering quality while significantly reducing
   unnecessary distance calculations and computational cost.

8. The refined offspring competes against the current population member. If the
   offspring achieves a better clustering objective value, it replaces the
   existing solution in the population.

9. The process continues iteratively until the population becomes sufficiently
   stable or no further improvement is observed for a predefined number of
   iterations.

Unlike standard K-means, which follows only a single optimization path from one
initialization, MDEClust continuously maintains, explores, and improves many
competing clustering alternatives simultaneously. Over time, the population
gradually evolves toward higher-quality and more stable clustering solutions.

---

## Parallel MDEClust

Parallel MDEClust [1] extends the original algorithm with efficient multi-threaded
execution designed for modern multi-core CPUs.

Many computationally expensive operations — including population initialization,
Differential Evolution recombination, center matching, mutation, repair, and
K-means refinement — can be processed concurrently across multiple CPU threads.

This parallel design is especially beneficial for larger datasets and demanding
clustering experiments, where repeated K-means refinement becomes computationally
expensive. By combining population-level parallelism with accelerated local
optimization, Parallel MDEClust substantially reduces running time while
preserving the core evolutionary structure of the original algorithm.

---

## Requirements

- Python 3.10+
- NumPy
- Numba

Install dependencies using:

```bash
pip install numpy numba
```

---

## Repository Structure

```text
mdeclust.py           Sequential and parallel MDEClust implementations
hamerly.py            Hamerly accelerated exact K-means implementation
munkres.py            Pure-Numba Hungarian algorithm implementation
mdeclust_demo.py      Example experiment script
liver_disorders.data  Sample dataset used by the demo script
requirements.txt      Python package dependencies
README.md             Project documentation
```

---

## Quick Start

```bash
git clone https://github.com/R-Mussabayev/MDEClust.git
cd MDEClust
pip install numpy numba
python mdeclust_demo.py
```

---

## Example Usage

```python
import numpy as np
from mdeclust import mdeclust_parallel

# Generate a synthetic dataset of uniformly distributed random points
rng = np.random.default_rng(42)
points = rng.random((300, 2))

# Run MDEClust
objective, centers, assignment, n_dists, n_execs, \
best_time, best_n_execs, best_n_dists = mdeclust_parallel(
    points,
    k=3,                   # Number of clusters
    population_size=20,    # Population size for Differential Evolution
    tol=0.0001,            # Population diversity threshold for stopping
    nmax=100,              # Maximum number of consecutive non-improving iterations
    matching_mode=0,       # Center matching method: 0 = Hungarian, 1 = greedy
    mutation=False,        # Enable or disable the mutation operator
    alpha=0.5,             # Mutation bias parameter: 0 = uniform, 1 = distance-biased
    n_attempts=3,          # Number of attempts to select distinct parent solutions
    printing=True          # Print progress information    
)

print("Final Objective:", objective)
print("Cluster Centers:")
print(centers)
print("Time to Best Solution:", best_time)
```

For a complete experiment using the included `liver_disorders.data`
dataset, run:

```bash
python mdeclust_demo.py
```

---

### Output

The algorithm returns:

- `objective` — final MSSC objective value
- `centers` — cluster centers
- `assignment` — cluster assignment for each point
- `n_dists` — number of squared Euclidean distance calculations
- `n_execs` — number of K-means executions
- `best_time` — time required to obtain the best solution
- `best_n_execs` — number of K-means executions performed before obtaining the best solution
- `best_n_dists` — number of Euclidean distance calculations performed before obtaining the best solution

---

## Notes

- The first execution performs Numba JIT compilation and may take noticeable time.
  This overhead occurs only once per function signature and does not affect
  subsequent execution performance.

- The MDEClust algorithm is stochastic. Different runs may produce different
  clustering solutions and objective values.

---

## References

### Parallel MDEClust implementation

[1] Rustam Mussabayev, Ravil Mussabayev, Alexander Krassovitskiy,
Irina Ualiyeva (2026).

*Parallel Memetic Differential Evolution for Minimum Sum-of-Squares Clustering.*

In: Nguyen, N.T., et al. *Advances in Computational Collective Intelligence.*
ICCCI 2025. Communications in Computer and Information Science, vol 2747.
Springer, Cham.

https://doi.org/10.1007/978-3-032-10202-7_7

### Original MDEClust algorithm

[2] Pierluigi Mansueto, Fabio Schoen.

*Memetic differential evolution methods for clustering problems.*

Pattern Recognition, Volume 114, 2021, 107849

https://doi.org/10.1016/j.patcog.2021.107849

### Original Hamerly K-means algorithm

[3] Greg Hamerly.

*Making k-means even faster.*

Proceedings of the 2010 SIAM International Conference on Data Mining,
pp. 130–140.

### Numba Hungarian algorithm implementation

[4] Brian M. Clapper et al.

*numba-munkres* project.

https://github.com/hudl/numba-munkres

---

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{mussabayev2026parallel,
  title="Parallel Memetic Differential Evolution for Minimum Sum-of-Squares Clustering",
  author="Mussabayev, Rustam and Mussabayev, Ravil and Krassovitskiy, Alexander and Ualiyeva, Irina",
  booktitle="Advances in Computational Collective Intelligence",
  volume={2747},
  year="2026",
  publisher="Springer",
  address="Cham",
  pages="90--106",
  doi={10.1007/978-3-032-10202-7_7}
}
```

---

## License

The original code developed for this repository is licensed under the MIT License.

The `munkres.py` module contains adapted code derived from the
`numba-munkres` project:

https://github.com/hudl/numba-munkres

which is licensed under the Apache License, Version 2.0.

Please refer to the corresponding license files for details.
