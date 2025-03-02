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
        if element!=Permutation(range(n)):
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
        if element!=Permutation(range(n)):
            elts.append(permutation_matrix(element,n))
    return elts

def graph_auts_from_bliss(pcm,print_order=True):
    """
    Computes the automorphism group of a Tanner graph derived from a parity-check matrix (pcm) using the BLISS algorithm.

    This function takes a parity-check matrix, constructs the corresponding Tanner graph,
    and then calculates the automorphism group of this graph. It returns a list of
    automorphism permutations on the graph vertices. 

    Args:
        pcm (numpy.ndarray or scipy.sparse.csr_matrix): The parity-check matrix.
        print_order (bool, optional): If True, prints the order of the automorphism group.
                                      Defaults to True.

    Returns:
        auts_list: A list of graph automorphism permutations, where each permutation
                               is represented as a NumPy array. The length of each permutation
                               is the sum of the number of check nodes and variable nodes.
    """
    m,n=pcm.shape
    tanner_graph=sparse_pcm_to_tanner_graph(pcm)
    autgroup=graph_aut_group(tanner_graph, print_order=print_order)
    if print_order:
        auts_list=list_perm_group_elements_schreier(autgroup,n=n+m)
    else:
        auts_list=list_perm_group_elements(autgroup,n=n+m)
    return auts_list

def vertex_graph_auts_from_bliss(pcm,print_order=True):
    """
    Computes the vertex-wise automorphism group of a Tanner graph derived from a parity-check matrix (PCM) 
    using the BLISS algorithm.

    This function takes a parity-check matrix, constructs the corresponding Tanner graph,
    and then calculates the automorphism group of this graph. It returns separate lists of
    permutation matrices representing the automorphisms acting on the column (variable) nodes
    and row (check) nodes of the Tanner graph. If we print the order of the automorphism group,
    sympy runs Schreier Sims on the background, so it's faster to enumerate elements with the same method.
    Otherwise, Dimino's enumeration method is faster for relatively small groups.

    Args:
        pcm (numpy.ndarray or scipy.sparse.csr_matrix): The parity-check matrix.
        print_order (bool, optional): If True, prints the order of the automorphism group.
                                      Defaults to True.

    Returns:
        tuple (list of numpy.ndarray, list of numpy.ndarray): A tuple containing two lists:
            - The first list contains permutation matrices representing automorphisms on the column nodes.
            - The second list contains permutation matrices representing automorphisms on the row nodes.
    """
    m,n=pcm.shape
    tanner_graph=sparse_pcm_to_tanner_graph(pcm)
    autgroup=graph_aut_group(tanner_graph, print_order=print_order)
    row_perms = []
    col_perms = []
    if print_order: 
        for element in autgroup.generate_schreier_sims():
            if element!=Permutation(range(n+m)):
                PX=permutation_matrix(element,n+m)
                col_perm = PX[:n,:n]
                row_perm = PX[n:,n:]
                row_perms.append(row_perm)
                col_perms.append(col_perm)
    else:
        for element in autgroup.generate_dimino():
            if element!=Permutation(range(n+m)):
                PX=permutation_matrix(element,n+m)
                col_perm = PX[:n,:n]
                row_perm = PX[n:,n:]
                row_perms.append(row_perm)
                col_perms.append(col_perm)
    return col_perms, row_perms

def random_permutation_group_elements(grp, k):
    """
    Generates k distinct random elements from a SymPy permutation group, excluding the identity.

    Args:
        grp (sympy.combinatorics.perm_groups.PermutationGroup): The permutation group.
        k (int): The number of distinct random elements to generate.

    Returns:
        list of sympy.combinatorics.permutations.Permutation: A list of k distinct random permutations.
    """
    # if k > group.order() - 1:
    #     raise ValueError("k cannot be greater than the order of the group minus 1 (excluding identity).")

    identity = Permutation(grp.degree)
    elements = set()
    while len(elements) < k:
        if grp.schreier_sims:
            random_element = grp.random()
        else:
            random_element = grp.random_element_dimino()
        if random_element != identity:
            elements.add(random_element)
    return list(elements)


def random_vertex_graph_auts_from_bliss(pcm, k, print_order=True):
    """
    Computes the vertex-wise automorphism group of a Tanner graph derived from a parity-check matrix (PCM) 
    using the BLISS algorithm and returns k random elements from it.

    This function takes a parity-check matrix, constructs the corresponding Tanner graph,
    and then calculates the automorphism group of this graph. It returns separate lists of
    permutation matrices representing the automorphisms acting on the column (variable) nodes
    and row (check) nodes of the Tanner graph. If we print the order of the automorphism group,
    sympy runs Schreier Sims on the background, so it's faster to enumerate elements with the same method.
    Otherwise, Dimino's enumeration method is faster for relatively small groups.

    Args:
        pcm: The parity-check matrix.
        k: Number of random permutations to take.
        print_order (optional): If True, prints the order of the automorphism group.
                                Defaults to True.

    Returns:
        tuple (list of numpy.ndarray, list of numpy.ndarray): A tuple containing two lists:
            - The first list contains permutation matrices representing automorphisms on the column nodes.
            - The second list contains permutation matrices representing automorphisms on the row nodes.
    """
    m,n=pcm.shape
    tanner_graph=sparse_pcm_to_tanner_graph(pcm)
    autgroup=graph_aut_group(tanner_graph, print_order=print_order)
    rand_elements=random_permutation_group_elements(autgroup,k)
    row_perms = []
    col_perms = [] 
    for element in rand_elements:
        if element!=Permutation(range(n+m)):
            PX=permutation_matrix(element,n+m)
            col_perm = PX[:n,:n]
            row_perm = PX[n:,n:]
            row_perms.append(row_perm)
            col_perms.append(col_perm)
    return col_perms, row_perms