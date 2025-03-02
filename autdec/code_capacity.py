# python
import numpy as np
from ldpc import BpDecoder, BpOsdDecoder
from ldpc.bplsd_decoder import BpLsdDecoder
from scipy.sparse import csr_matrix
import random

# AutDEC
from autdec.perm_utils import *

def AutDEC_code_capacity(HX,HZ,LX,LZ,error_rate,shots,base_decoder,decoder_parameters,A_X,A_Z):

    """Calculates the logical error rates for a base decoder and AutDEC.

  This function simulates the performance of a given base decoder and AutDEC 
  on a CSS code C=CSS(HX,HZ) composed of two classical codes with specified parity 
  check matrices and logical operators, subject to a given physical error rate.

  Args:
    HX: Parity check matrix for the X code.
    HZ: Parity check matrix for the Z code.
    LX: Logical operator matrix for the X code.
    LZ: Logical operator matrix for the Z code.
    error_rate: The physical error rate.
    shots: The number of Monte Carlo simulations to run.
    base_decoder: The base decoder to use ('BP', 'BPOSD', or 'BPLSD').
    decoder_parameters: Parameters for the base decoder (e.g., number of iterations).
    A_X: A list of permutation matrices representing the automorphisms of HX.
    A_Z: A list of permutation matrices representing the automorphisms of HZ.

  Returns:
    A tuple containing:
      - shots: The number of shots taken.
      - base_dec_errs: The number of logical errors for the base decoder.
      - AutDEC_errs: The number of logical errors for AutDEC.
  """
    
    mx, nx = HX.shape
    mz, nz = HZ.shape
    k = LX.shape[0]
    assert nx == nz
    n = nx
    L = np.vstack((np.hstack((LX,np.zeros((k,n),dtype=int))),np.hstack((np.zeros((k,n),dtype=int),LZ))))
    H = np.vstack((np.hstack((HX,np.zeros((mx,n),dtype=int))),np.hstack((np.zeros((mz,n),dtype=int),HZ))))

    if base_decoder in ['BP','bp','Bp']:
        decoder = BpDecoder
    elif base_decoder in ['BP+OSD','BPOSD','bposd','osd','BpOsd']:
        decoder = BpOsdDecoder
    elif base_decoder in ['BP+LSD','BPLSD','bplsd','lsd','BpLsd']:
        decoder = BpLsdDecoder
    else:
        raise ValueError("Available base decoders are: BP, BP+OSD and BP+LSD.")

    ensemble_X = []
    Ua_X_list = []
    ensemble_Z= []
    Ua_Z_list = []
    for i in range(len(A_X)):
        # Auts of HX
        HX_new = HX@A_X[i]
        if i == 0:
            Ua_X = np.eye(mx,dtype=int)
        else: 
            Ua_X = stab_map(HX,HX_new)
        Ua_X_list.append(Ua_X)
        ensemble_X.append(decoder(csr_matrix(HX_new),error_rate=error_rate,**decoder_parameters))
        # Auts of HZ
        HZ_new = HZ@A_Z[i]
        if i == 0:
            Ua_Z = np.eye(mz,dtype=int)
        else: 
            Ua_Z = stab_map(HZ,HZ_new)
        Ua_Z_list.append(Ua_Z)
        ensemble_Z.append(decoder(csr_matrix(HZ_new),error_rate=error_rate,**decoder_parameters))
    
    base_dec_errs = 0
    AutDEC_errs = 0
    print('Starting decoding.')
    print(f"{'Trial no':<12} {'BaseDec Errors':>15} {'AutDEC Errors':>15}") 
    checkpoint = 5000
    for trial in range(shots):
        error = depolarizing_channel(error_rate,n)
        # Z errors and corrections
        correction_ensemble_Z = []
        eZ = error[:n]
        sX = (HX@eZ)%2 
        # X errors and corrections
        correction_ensemble_X = []
        eX = error[n:]
        sZ = (HZ@eX)%2

        for i in range(len(ensemble_X)): 
            corr_AutDEC_X = ensemble_X[i].decode(Ua_X_list[i]@sX%2)
            if i==0:
                cX_base = corr_AutDEC_X.copy()
            if np.allclose(HX@corr_AutDEC_X%2,sX): # check if BP converged. 
                correction_ensemble_X.append(corr_AutDEC_X)

            corr_AutDEC_Z = ensemble_Z[i].decode(Ua_Z_list[i]@sZ%2)
            if i==0:
                cZ_base = corr_AutDEC_Z.copy()
            if np.allclose(HZ@corr_AutDEC_Z%2,sZ): # check if BP converged. 
                correction_ensemble_Z.append(corr_AutDEC_Z)
                    

        if not correction_ensemble_X:
            cX_AutDEC=cX_base.copy()
        else:
            weights_cX = np.array(correction_ensemble_X).sum(axis=1)
            cX_indices = np.arange(len(correction_ensemble_X),dtype=int)
            weights_cX_sorted = sorted(zip(weights_cX, cX_indices))
            cX_AutDEC = correction_ensemble_X[weights_cX_sorted[0][1]]
        if not correction_ensemble_Z:
            cZ_AutDEC=cZ_base.copy()
        else:
            weights_cZ = np.array(correction_ensemble_Z).sum(axis=1)
            cZ_indices = np.arange(len(correction_ensemble_Z),dtype=int)
            weights_cZ_sorted = sorted(zip(weights_cZ, cZ_indices))
            cZ_AutDEC = correction_ensemble_Z[weights_cZ_sorted[0][1]]

            
        full_corr_base = np.hstack((cX_base,cZ_base))
        full_corr_AutDEC = np.hstack((cX_AutDEC,cZ_AutDEC))
        
        ### Check full correction ###
        if np.allclose(full_corr_base,error):
            assert logical_error_check(full_corr_base,error,L,H)==0
        else: 
            if logical_error_check(full_corr_base,error,L,H) == 1:
                base_dec_errs += 1

        if np.allclose(full_corr_AutDEC,error):
            assert logical_error_check(full_corr_AutDEC,error,L,H)==0
        else: 
            if logical_error_check(full_corr_AutDEC,error,L,H) == 1:
                AutDEC_errs += 1

        if (trial+1)%checkpoint == 0:
            print(f"{trial+1:<12} {base_dec_errs:>15} {AutDEC_errs:>15}") #align outputs
            
    return shots, base_dec_errs, AutDEC_errs
   

def depolarizing_channel(p,n): 
    prob_I = 1-p
    error_loc = np.zeros(n,dtype=int)
    error = np.zeros(2*n,dtype=int)
    for i in range(n):
        if np.random.random()>prob_I:
            error_loc[i]=1
    for q in np.where(error_loc==1)[0]:
        e_opts = ['X','Y','Z']
        e_type = random.choice(e_opts)
        if e_type == 'Z': 
            error[q]=1
        elif e_type=='X':
            error[q+n]=1
        else:
            error[q]=1
            error[q+n]=1
    return error

def logical_error_check(correction,error,logicals,check_matrix):
    residual_error=(correction+error)%2
    logical_error = 0
    m = check_matrix.shape[0]
    k = logicals.shape[0]

    # Check 1: If the residual error is in the codespace, ie commutes with stab gens
    if np.allclose(check_matrix@residual_error%2,np.zeros(m)) == False:
        logical_error += 1
    # Check 2: If the residual error has caused a logical error, ie anticommutes with opposite logicals
    elif np.allclose(logicals@residual_error%2,np.zeros(k)) == False:
        logical_error += 1
    return logical_error    