# Pure-Numba Kuhn-Munkres realization of Hungarian algorithm for parallel processing.
# Adopted by Rustam Mussabayev from the original numba-munkres project.

# Original project is acceptable at https://github.com/hudl/numba-munkres
# and licensed under the Apache License, Version 2.0.
# See https://github.com/hudl/numba-munkres/blob/master/LICENSE.md for details.
# Copyright © 2008 Brian M. Clapper.

from numba import njit
import numpy as np

@njit
def pad_matrix(matrix):
    r"""
    Pad a possibly non-square matrix to make it square.

    Parameters
    ----------
    matrix : np.ndarray
        A ``(n_rows, n_cols)`` matrix

    Returns
    -------
    padded : np.ndarray
        A padded ``(max(n_rows, n_cols), max(n_rows, n_cols))`` cost matrix
    """
    matrix = np.asarray(matrix)
    n_rows, n_cols = matrix.shape
    n_dim = max(n_rows, n_cols)
    padded = np.zeros((n_dim, n_dim))
    padded[:n_rows, :n_cols] = matrix
    return padded

@njit
def munkres(cost_matrix):
    """
    Compute the indexes for the lowest-cost pairings between rows and columns
    in the database. Returns a list of indexes where the row index is the list 
    position and column index is the list value. 

    Parameters
    ----------
    cost_matrix : np.ndarray
        The ``(n_rows, n_cols)`` cost matrix. If this cost matrix is not
        square, it will be padded with zeros, via a call to ``pad_matrix()``.

    Returns
    -------
    matches : np.ndarray
        The ``(max(n_rows, n_cols)`` vector of rows to columns assignments        
    """
    n_rows, n_cols = cost_matrix.shape
    C = pad_matrix(cost_matrix)
    n = C.shape[0]    
    row_covered = np.full(n, False)
    col_covered = np.full(n, False)    
    Z0_r = np.zeros(1)
    Z0_c = np.zeros(1)
    path = np.zeros((n*2, n*2))
    matches = np.zeros((n, n))
    args = (C, row_covered, col_covered, Z0_r, Z0_c, path, matches)
    done = False
    step = 1
    while not done:
        if 0 < step < 7:
            if step == 1:
                step = __step1(*args)
            elif step == 2:
                step = __step2(*args)
            elif step == 3:
                step = __step3(*args)
            elif step == 4:
                step = __step4(*args)
            elif step == 5:
                step = __step5(*args)
            elif step == 6:
                step = __step6(*args)            
        else:
            done = True
    mat = matches[:n_rows, :n_cols]
    col_ind = np.full(n_rows, -1)
    for i in range(n_rows):
        col_ind[i] = np.argmax(mat[i])        
    return col_ind


@njit
def __step1(*args):
    r"""
    For each row of the matrix, find the smallest element and
    subtract it from every element in its row. Go to Step 2.
    """
    C, row_covered, col_covered, Z0_r, Z0_c, path, matches = args
    n = C.shape[0]
    for i in range(n):
        minval = C[i].min()
        # Find the minimum value for this row and subtract that minimum
        # from every element in the row.
        for j in range(n):
            C[i, j] -= minval
    return 2


@njit
def __step2(*args):
    r"""
    Find a zero (Z) in the resulting matrix. If there is no starred
    zero in its row or column, star Z. Repeat for each element in the
    matrix. Go to Step 3.
    """
    C, row_covered, col_covered, Z0_r, Z0_c, path, matches = args
    n = C.shape[0]
    for i in range(n):
        for j in range(n):
            if (C[i, j] == 0) and not col_covered[j] and not row_covered[i]:
                matches[i, j] = 1
                col_covered[j] = True
                row_covered[i] = True
                break
    col_covered[:] = False
    row_covered[:] = False
    return 3


@njit
def __step3(*args):
    r"""
    Cover each column containing a starred zero. If K columns are
    covered, the starred zeros describe a complete set of unique
    assignments. In this case, Go to DONE, otherwise, Go to Step 4.
    """
    C, row_covered, col_covered, Z0_r, Z0_c, path, matches = args
    n = C.shape[0]
    count = 0
    for i in range(n):
        for j in range(n):
            if matches[i, j] == 1 and not col_covered[j]:
                col_covered[j] = True
                count += 1
    if count >= n:
        step = 7  # done
    else:
        step = 4
    return step


@njit
def __step4(*args):
    r"""
    Find a noncovered zero and prime it. If there is no starred zero
    in the row containing this primed zero, Go to Step 5. Otherwise,
    cover this row and uncover the column containing the starred
    zero. Continue in this manner until there are no uncovered zeros
    left. Save the smallest uncovered value and Go to Step 6.
    """
    C, row_covered, col_covered, Z0_r, Z0_c, path, matches = args
    step = 0
    done = False
    while not done:
        row, col = __find_a_zero(*args)
        if row < 0:
            done = True
            step = 6
        else:
            matches[row, col] = 2
            star_col = __find_star_in_row(row, *args)
            if star_col >= 0:
                col = star_col
                row_covered[row] = True
                col_covered[col] = False
            else:
                done = True
                Z0_r[0] = row
                Z0_c[0] = col
                step = 5
    return step


@njit
def __step5(*args):
    r"""
    Construct a series of alternating primed and starred zeros as
    follows. Let Z0 represent the uncovered primed zero found in Step 4.
    Let Z1 denote the starred zero in the column of Z0 (if any).
    Let Z2 denote the primed zero in the row of Z1 (there will always
    be one). Continue until the series terminates at a primed zero
    that has no starred zero in its column. Unstar each starred zero
    of the series, star each primed zero of the series, erase all
    primes and uncover every line in the matrix. Return to Step 3
    """
    C, row_covered, col_covered, Z0_r, Z0_c, path, matches = args
    count = 0
    path[count, 0] = Z0_r[0]
    path[count, 1] = Z0_c[0]
    done = False
    while not done:
        row = __find_star_in_col(path[count, 1], *args)
        if row >= 0:
            count += 1
            path[count, 0] = row
            path[count, 1] = path[count-1, 1]
        else:
            done = True
        if not done:
            col = __find_prime_in_row(path[count, 0], *args)
            count += 1
            path[count, 0] = path[count-1, 0]
            path[count, 1] = col
    __convert_path(count, *args)
    col_covered[:] = False
    row_covered[:] = False
    __erase_primes(*args)
    return 3


@njit
def __step6(*args):
    r"""
    Add the value found in Step 4 to every element of each covered
    row, and subtract it from every element of each uncovered column.
    Return to Step 4 without altering any stars, primes, or covered
    lines.
    """
    C, row_covered, col_covered, Z0_r, Z0_c, path, matches = args
    n = C.shape[0]
    minval = __find_smallest(*args)
    events = 0  # track actual changes to matrix
    for i in range(n):
        for j in range(n):
            if row_covered[i]:
                C[i, j] += minval
                events += 1
            if not col_covered[j]:
                C[i, j] -= minval
                events += 1
            if row_covered[i] and not col_covered[j]:
                events -= 2  # change reversed, no real difference
    if events == 0:
        raise ValueError('Matrix cannot be solved!')
    return 4


@njit
def __find_smallest(*args):
    r"""
    Find the smallest uncovered value in the matrix.
    """
    C, row_covered, col_covered, Z0_r, Z0_c, path, matches = args
    n = C.shape[0]
    minval = np.inf
    for i in range(n):
        for j in range(n):
            if not row_covered[i] and not col_covered[j]:
                if minval > C[i, j]:
                    minval = C[i, j]
    return minval


@njit
def __find_a_zero(*args):
    r"""
    Find the first uncovered element with value 0
    """
    C, row_covered, col_covered, Z0_r, Z0_c, path, matches = args
    n = C.shape[0]
    row = -1
    col = -1
    i = 0
    done = False
    while not done:
        j = 0
        while True:
            if C[i, j] == 0 and not row_covered[i] and not col_covered[j]:
                row = i
                col = j
                done = True
            j += 1
            if j >= n:
                break
        i += 1
        if i >= n:
            done = True
    return row, col


@njit
def __find_star_in_row(row, *args):
    r"""
    Find the first starred element in the specified row. Returns
    the column index, or -1 if no starred element was found.

    Parameters
    ----------

    """
    C, row_covered, col_covered, Z0_r, Z0_c, path, matches = args
    n = C.shape[0]
    col = -1
    for j in range(n):
        if matches[row, j] == 1:
            col = j
            break

    return col


@njit
def __find_star_in_col(col, *args):
    r"""
    Find the first starred element in the specified row. Returns
    the row index, or -1 if no starred element was found.

    Parameters
    ----------
    col : int
        The index of the column to look for the starred element in

    Returns
    -------
    row : int
        The starred element in `col`
    """
    C, row_covered, col_covered, Z0_r, Z0_c, path, matches = args
    n = C.shape[0]
    row = -1
    for i in range(n):
        if matches[i, int(col)] == 1:
            row = i
            break
    return row


@njit
def __find_prime_in_row(row, *args):
    r"""
    Find the first prime element in the specified row. Returns
    the column index, or -1 if no starred element was found.

    Parameters
    ----------
    row : int
        The index of the row to look for the prime element in

    Returns
    -------
    col : int
        The prime element in `row`
    """
    C, row_covered, col_covered, Z0_r, Z0_c, path, matches = args
    n = C.shape[0]
    col = -1
    for j in range(n):
        if matches[int(row), j] == 2:
            col = j
            break
    return col


@njit
def __convert_path(count, *args):
    r"""
    Reverse matches
    """
    C, row_covered, col_covered, Z0_r, Z0_c, path, matches = args
    for i in range(count+1):
        if matches[int(path[i, 0]), int(path[i, 1])] == 1:
            matches[int(path[i, 0]), int(path[i, 1])] = 0
        else:
            matches[int(path[i, 0]), int(path[i, 1])] = 1


@njit
def __erase_primes(*args):
    r"""
    Erase all prime markings
    """
    C, row_covered, col_covered, Z0_r, Z0_c, path, matches = args
    n = C.shape[0]
    for i in range(n):
        for j in range(n):
            if matches[i, j] == 2:
                matches[i, j] = 0
