import numpy as np
def perm_mat_from_aut(aut,n):
    perm_matrix = np.eye(n,dtype=int)
    for cycle in aut:
        # Rotate the elements in the cycle
        for i in range(len(cycle)):
            from_idx = cycle[i] - 1  # convert to 0-based index
            to_idx = cycle[(i + 1) % len(cycle)] - 1  # next element in the cycle
            perm_matrix[from_idx, from_idx] = 0
            perm_matrix[from_idx, to_idx] = 1

    return perm_matrix