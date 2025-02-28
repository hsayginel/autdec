import numpy as np
import numba as nb

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

@nb.jit(nb.types.Tuple((nb.int8[:,:],nb.int64[:]))(nb.int8[:,:],nb.int64,nb.int64,nb.int64,nb.int64))
def HowZ2(A,tB,nB,nC,r0):
    pivots = []
    B = A.copy()
    if np.sum(A) == 0:
        return B,np.array(pivots,dtype=nb.int64)
    m = len(B)
    r = r0
    for j in range(nC):
        for t in range(tB):
            ## c is the column of B we are currently looking at
            c = j + t * nB
            iList = [i for i in range(r,m) if B[i,c] > 0]
            if len(iList) > 0:
                i = iList.pop(0)
                pivots.append(c)
                ## found j: if j > r, swap row j with row r
                if i > r:  
                    ## swap using bitflips - more elegant than array indexing
                    B[r] = B[r] ^ B[i]
                    B[i] = B[r] ^ B[i]
                    B[r] = B[r] ^ B[i]
                ## eliminate non-zero entries in column c apart from row r
                for i in [i for i in range(r) if B[i,c] > 0] + iList:     
                    B[i] = B[i] ^ B[r]
                r +=1
    return B,np.array(pivots,dtype=nb.int64)
    
def get_CNOT_circ(A,tB,nB,nC,r0):
    qc = []
    pivots = []
    B = A.copy()
    if np.sum(A) == 0:
        return B,pivots,qc
    m = len(B)
    r = r0
    for j in range(nC):
        for t in range(tB):
            ## c is the column of B we are currently looking at
            c = j + t * nB
            iList = [i for i in range(r,m) if B[i,c] > 0]
            if len(iList) > 0:
                i = iList.pop(0)
                pivots.append(c)
                ## found j: if j > r, swap row j with row r
                if i > r:  
                    ## swap using bitflips - more elegant than array indexing
                    B[r] = B[r] ^ B[i]
                    B[i] = B[r] ^ B[i]
                    B[r] = B[r] ^ B[i]
                    qc.append(('SWAP',(i+1,r+1)))
                ## eliminate non-zero entries in column c apart from row r
                for i in [i for i in range(r) if B[i,c] > 0] + iList:     
                    B[i] = B[i] ^ B[r]
                    qc.append(('CNOT',(i+1,r+1)))
                r +=1
    return B,pivots,qc

# @nb.jit
def blockDims(n,nA=0,tB=1,nC=-1):
    nA = min(n,nA)
    nB = (n - nA) // tB
    if nC < 0 or nC > nB:
        nC = nB 
    return nA,nB,nC

def invRange(n,S):
    '''return list of elements of range(n) NOT in S'''
    return sorted(set(range(n)) - set(S))

def ix_to_perm_mat(ix):
    m = len(ix)
    P = np.zeros((m,m),dtype=int)
    for i in range(m):
        P[ix[i],i] = 1
    return P

def rref_mod2(A,CNOTs=False):
    '''Return Howell matrix form modulo N plus transformation matrix U such that H = U @ A mod N'''
    if A.ndim == 1:
        A = A.reshape(1, -1)
    m,n = A.shape
    A = np.array(A,dtype=np.int8)
    B = np.hstack([A,np.eye(m,dtype=np.int8)])
    nA=0
    tB=1
    nC=-1
    r0=0
    nA,nB,nC = blockDims(n,nA,tB,nC)

    if CNOTs == False:
        HU, pivots = HowZ2(B,tB,nB,nC,r0)
        ix = list(pivots) + invRange(n,pivots)

        H, U = HU[:,:n],HU[:,n:]
        H = H[:,ix]
        P = ix_to_perm_mat(ix)
        
        return H, pivots, U, P
    elif CNOTs: 
        HU, pivots, qc = get_CNOT_circ(B,tB,nB,nC,r0)
        ix = list(pivots) + invRange(n,pivots)
        H, U = HU[:,:n],HU[:,n:]
        H = H[:,ix]

        return qc, H

def inv_mod2(mat):
    _, _, U, P = rref_mod2(mat)
    return (P@U)%2

def stab_map(HX,HX_new):
    _,pivots,row_transform,_ = rref_mod2(HX)
    m = HX.shape[0]
    stab_map_mat = np.zeros((m,m),dtype=int)
    for row,s in enumerate(HX_new):
        for i,p in enumerate(pivots):
            if s[p] == 1:
                stab_map_mat[row,i] = 1
    return (stab_map_mat@row_transform)%2

def symp_prod(A,B,return_omega=False):
    """
    Computes the binary symplectic product between A and B.
    """
    A = np.array(A,dtype=int)
    B = np.array(B,dtype=int)

    if A.ndim == 1:
        A = A.reshape(1, -1)
    if B.ndim == 1:
        B = B.reshape(1, -1)
        
    m, n = A.shape
    m_b, n_b = B.shape

    assert n == n_b 

    omega = np.eye(n,dtype=int)
    nhalf = n//2
    omega[:,:nhalf], omega[:,nhalf:] = omega[:,nhalf:].copy(), omega[:,:nhalf].copy()

    if return_omega:
        return np.array((A @ omega @ B.T)%2,dtype=int), omega
    else: 
        return np.array((A @ omega @ B.T)%2,dtype=int)

def rank_mod2(mat):
    """Return rank of binary matrix."""
    _, pivots, _, _ = rref_mod2(mat)
    return len(pivots) 

def is_matrix_full_rank(mat):
    """Checks if matrix is full rank."""
    if mat.ndim == 1:
        return True
    m = mat.shape[0]
    k = rank_mod2(mat)
    return m == k

def compute_standard_form(G):
    """
    Returns the standard form of a stabilizer code. 
    See Nielsen & Chuang Section 10.5.7.
    """
    if is_matrix_full_rank(G) == False:
        raise AssertionError("Rows of G should be independent. Use a generating set of stabilizers.")
    n, m = G.shape[1] // 2, G.shape[0]
    k = n-m
    
    G1 = G[:, :n]
    G2 = G[:, n:]

    ## LOGICALS

    # Step 1: Gaussian Elimination on G1 & adjust G2 rows & cols accordingly
    G1_rref, r_pivots, G1_transform_rows, G1_transform_cols = rref_mod2(G1)
    r = len(r_pivots)
    G = np.hstack((G1_rref,(G1_transform_rows@G2@G1_transform_cols)%2))

    # Step 2: Swap columns r to n from X to Z block 
    G[:,r:n], G[:,n+r:2*n] = G[:,n+r:2*n].copy(), G[:,r:n].copy()
    E1 = G[:, :n]
    E2 = G[:, n:]
    E1_rref, e_pivots, E_transform_rows, E_transform_cols = rref_mod2(E1)
    e = len(e_pivots)
    s = e - r
    G = np.hstack((E1_rref,(E_transform_rows@E2@E_transform_cols)%2))
    G[:,r:n], G[:,n+r:2*n] = G[:,n+r:2*n].copy(), G[:,r:n].copy()

    # Step 3: Z Logicals
    A_2 = G[:r,n-k:n]
    C = G[:r,(2*n-k):]
    E = G[r:,(2*n-k):]

    r1 = A_2.T
    r2 = np.zeros((k,n-k-r))
    r3 = np.eye(k)
    right = np.hstack((np.hstack((r1,r2)),r3))
    Z_logicals = np.hstack((np.zeros((k,n)),right))
    Z_logicals = np.array(Z_logicals,dtype=int)
    for z in Z_logicals:
        assert len(z) == 2*n
        assert np.allclose(symp_prod(G,z),np.zeros((G.shape[0])))

    # Step 4: X Logicals
    l1 = np.zeros((k,r))
    l2 = E.T
    l3 = np.eye(k)
    left = np.hstack((np.hstack((l1,l2)),l3))
    r1 = C.T
    r2 = np.zeros((k,n-k-r))
    r3 = np.zeros((k,k))
    right = np.hstack((np.hstack((r1,r2)),r3))
    X_logicals = np.hstack((left,right))
    X_logicals = np.array(X_logicals,dtype=int)
    for x in X_logicals:
        assert len(x) == 2*n
        assert np.allclose(symp_prod(G,x),np.zeros((G.shape[0])))

    # Step 5: Move columns (but not rows) back to their original position
    inv_E_transform_cols = inv_mod2(E_transform_cols)
    inv_G1_transform_cols = inv_mod2(G1_transform_cols)
    inv_transform_cols = inv_E_transform_cols @ inv_G1_transform_cols
    ## STABILIZERS
    G_new = np.zeros_like(G)
    G_new[:,:n] = (G[:,:n] @ inv_transform_cols)%2
    G_new[:,n:] = (G[:,n:] @ inv_transform_cols)%2
    ## LOGICALS
    Z_logicals_og_basis = np.zeros_like(Z_logicals)
    X_logicals_og_basis = np.zeros_like(X_logicals)
    Z_logicals_og_basis[:,:n] =  (Z_logicals[:,:n] @ inv_transform_cols)%2
    Z_logicals_og_basis[:,n:] =  (Z_logicals[:,n:] @ inv_transform_cols)%2
    X_logicals_og_basis[:,:n] =  (X_logicals[:,:n] @ inv_transform_cols)%2
    X_logicals_og_basis[:,n:] =  (X_logicals[:,n:] @ inv_transform_cols)%2
    ## DESTABILIZERS 
    DX = np.hstack([np.zeros((r,n)),np.eye(r),np.zeros((r,n-r))])
    DZ = np.hstack([np.zeros((s,r)),np.eye(s),np.zeros((s,n+k))])
    D = np.array(np.vstack([DX,DZ]),dtype=int)
    D_og_basis = np.zeros_like(D)
    D_og_basis[:,:n] =  (D[:,:n] @ inv_transform_cols)%2
    D_og_basis[:,n:] =  (D[:,n:] @ inv_transform_cols)%2

    return G_new, X_logicals_og_basis, Z_logicals_og_basis, D_og_basis

def stabs_to_H_symp(stabs):
    m = len(stabs)
    n = len(stabs[0])
    H_symp = np.zeros((m,2*n),dtype=int)
    for i_row, s in enumerate(stabs): 
        for i_col, pauli in enumerate(s):
            if pauli == 'I':
                pass
            elif pauli == 'X':
                H_symp[i_row,i_col] = 1
            elif pauli == 'Y':
                H_symp[i_row,i_col] = 1
                H_symp[i_row,i_col+n] = 1
            elif pauli == 'Z':
                H_symp[i_row,i_col+n] = 1
            else: 
                raise TypeError('Unknown Pauli: ',pauli)
            
    return H_symp