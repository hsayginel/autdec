import igraph as ig
import numpy as np
from sympy.combinatorics import Permutation, PermutationGroup
import scipy.sparse as sparse

def sparse_pcm_to_tanner_graph(pcm):
    """
    Creates a Tanner graph in igraph format from a sparse parity check matrix.
    Args:
        pcm: A (sparse) parity check matrix.
    Returns:
        An igraph Graph object representing the Tanner graph.
    """
    rows, cols = pcm.shape
    g = ig.Graph()
    g.add_vertices(cols + rows)
    # g.vs["type"] = ["variable"] * cols + ["check"] * rows
    g.vs["color"] = [1] * cols + [2] * rows
    edges = []
    rows_indices, col_indices = pcm.nonzero()
    for i in range(len(rows_indices)):
        edges.append((col_indices[i], cols + rows_indices[i]))
    g.add_edges(edges)
    return g

def graph_aut_group(g, print_order=True):
    """
    Calculates and returns the automorphism group of an igraph graph as a sympy PermutationGroup.

    Args:
        g: An igraph Graph object.
        print_order: A boolean indicating whether to print the order of the automorphism group.
                     Defaults to True.

    Returns:
        A sympy PermutationGroup object representing the automorphism group of the graph,
        or Identity if no automorphism generators are found.
    """
    automorphism_generators = g.automorphism_group(color=g.vs["color"])
    if automorphism_generators:
        sympy_permutations = [Permutation(list(generator)) for generator in automorphism_generators]
        sympy_group = PermutationGroup(sympy_permutations)
        if print_order:
            print("Automorphism group order:",sympy_group.order())
        return sympy_group
    else:
        print("Cannot create sympy group, no automorphism generators found.")
        return PermutationGroup(Permutation(range(1)))

def permutation_matrix(perm,n):
    """
    Creates a sparse permutation matrix (CSR format) from a sympy Permutation object.

    Args:
        perm: A sympy Permutation object
        n: the number of indices

    Returns:
        A scipy.sparse.csr_matrix representing the permutation matrix.
    """
    row_indices = list(range(n))
    col_indices = list(perm)
    data = np.ones(n, dtype=int)  
    return sparse.csr_matrix((data, (row_indices, col_indices)), shape=(n, n), dtype=int)

def list_perm_group_elements(group,n):
    """
    creates list of permutation matrices of all elements from a sympy permutation group using Dimino's method. 
    Faster than Schreier-Sims when we do not care about finding the order first.

    Args:
        group: A sympy PermutationGroup object.

    Returns:
        A NumPy array of all permutation matrices of automorphism group elements.
    """
    elts=[]
    for element in group.generate_dimino():
        elts.append(permutation_matrix(element,n))
    return elts

def list_perm_group_elements_schreier(group,n):
    """
    creates list of permutation matrices of all elements from a sympy permutation group using Schreier-Sims method.
    Faster than Dimino when we find the order of the group first.

    Args:
        group: A sympy PermutationGroup object.

    Returns:
        A NumPy array of all permutation matrices of automorphism group elements.
    """
    elts=[]
    for element in group.generate_schreier_sims():
        elts.append(permutation_matrix(element,n))
    return elts

def graph_auts_from_bliss(pcm,print_order=True):
    m,n=pcm.shape
    tanner_graph=sparse_pcm_to_tanner_graph(pcm)
    autgroup=graph_aut_group(tanner_graph, print_order=print_order)
    if print_order:
        auts_list=list_perm_group_elements_schreier(autgroup,n=n+m)
    else:
        auts_list=list_perm_group_elements(autgroup,n=n+m)
    return auts_list

def vertex_graph_auts_from_bliss(pcm,print_order=True):
    m,n=pcm.shape
    tanner_graph=sparse_pcm_to_tanner_graph(pcm)
    autgroup=graph_aut_group(tanner_graph, print_order=print_order)
    row_perms = []
    col_perms = []
    if print_order: 
        for element in autgroup.generate_schreier_sims():
            PX=permutation_matrix(element,n+m)
            col_perm = PX[:n,:n]
            row_perm = PX[n:,n:]
            row_perms.append(row_perm)
            col_perms.append(col_perm)
    else:
        for element in autgroup.generate_dimino():
            PX=permutation_matrix(element,n+m)
            col_perm = PX[:n,:n]
            row_perm = PX[n:,n:]
            row_perms.append(row_perm)
            col_perms.append(col_perm)
    return col_perms, row_perms

