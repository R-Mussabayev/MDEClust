# MDEClust

The Memetic Differential Evolution Clustering (MDEClust) algorithm is an
effective metaheuristic approach for solving the Euclidean Minimum
Sum-of-Squares Clustering (MSSC) problem, often producing high-quality
clustering solutions for small and medium-scale datasets.

In addition to a sequential MDEClust implementation based on [2], this
repository includes a parallel implementation proposed in [1], which combines
Differential Evolution global search with K-means local search under a unified
parallel framework.

Experimental results reported in [1] demonstrate that the parallel
MDEClust implementation achieves substantial runtime reductions while
preserving or improving clustering quality across benchmark datasets
compared with the sequential implementation.

The repository includes:

- Sequential MDEClust implementation based on [2]
- Parallel MDEClust implementation proposed in [1]
- Python/Numba implementation of the accelerated Hamerly K-means algorithm [3]
- Pure-Numba implementation of the Hungarian (Kuhn-Munkres) algorithm based on [4]
- Example experiment scripts and benchmark configurations

Programmed by Rustam Mussabayev (rmusab@gmail.com)

Initial release: 20 August 2022

---

## Features

- Sequential and parallel MDEClust implementations
- Numba JIT acceleration
- Parallel execution using `prange`
- Hamerly-accelerated exact K-means
- Pure-Numba Hungarian algorithm implementation
- Support for large-scale clustering experiments
- Detailed benchmarking statistics:
  - objective values
  - execution times
  - number of K-means executions
  - number of Euclidean distance calculations

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
mdeclust.py      Sequential and parallel MDEClust implementations
hamerly.py       Hamerly accelerated exact K-means implementation
munkres.py       Pure-Numba Hungarian algorithm implementation
mdeclust_demo.py Example experiment script
README.md        Project documentation
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
from mdeclust import mdeclust_parallel

objective, centers, assignment, n_dists, n_execs, \
best_time, best_n_execs, best_n_dists = \
    mdeclust_parallel(
        points,
        k=20,
        population_size=150,
        tol=0.0001,
        nmax=5000,
        matching_mode=0,
        mutation=False,
        alpha=0.5,
        n_attempts=3,
        printing=True
    )

print("Final Objective:", objective)
print("Time to Best Solution:", best_time)
```

---

## Parallel Implementation

The repository includes a parallel implementation of MDEClust proposed in [1].

The parallelization strategy is based on parallel population processing using
Numba's `prange` functionality. Independent clustering solutions within the
population are processed concurrently, allowing efficient utilization of
multi-core CPU architectures.

The implementation is intended for large-scale clustering problems where the
computational cost of repeated K-means executions becomes significant.

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
