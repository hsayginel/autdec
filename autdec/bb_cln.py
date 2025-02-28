import stim
print('Stim version:',stim.__version__)

# PYTHON
import numpy as np
from scipy.sparse import coo_matrix
import time

# LDPC 
from ldpc import BpDecoder, BpOsdDecoder
from ldpc.bplsd_decoder import BpLsdDecoder

# AUTDECODE 
from autdec.dem_utils import *

def autdecode_bb_cln(bb_code_name,error_rate, num_shots, DEM_col_perms, DEM_row_perms, base_decoder,decoder_hyperparams,basis='Z'):
    if bb_code_name == 'BB72' or bb_code_name == 'bb72':
        # [[72,12,6]]
        code, A_list, B_list = create_bivariate_bicycle_codes(6, 6, [3], [1,2], [1,2], [3])
        d = 6
    elif bb_code_name == 'BB90' or bb_code_name == 'bb90':
        # [[90,8,10]]
        code, A_list, B_list = create_bivariate_bicycle_codes(15, 3, [9], [1,2], [2,7], [0])
        d = 10
    elif bb_code_name == 'BB108' or bb_code_name == 'bb108':
        # [[108,8,10]]
        code, A_list, B_list = create_bivariate_bicycle_codes(9, 6, [3], [1,2], [1,2], [3])
        d = 10
    elif bb_code_name == 'BB144' or bb_code_name == 'bb144':
        # [[144,12,12]]
        code, A_list, B_list = create_bivariate_bicycle_codes(12, 6, [3], [1,2], [1,2], [3])
        d = 12
    elif bb_code_name == 'BB288' or bb_code_name == 'bb288':
        # [[288,12,18]]
        code, A_list, B_list = create_bivariate_bicycle_codes(12, 12, [3], [2,7], [1,2], [3])
        d = 18
    elif bb_code_name == 'BB360' or bb_code_name == 'bb360':
        # [[360,12,<=24]]
        code, A_list, B_list = create_bivariate_bicycle_codes(30, 6, [9], [1,2], [25,26], [3])
        d = 24
    elif bb_code_name == 'BB756' or bb_code_name == 'bb756':
        # [[756,16,<=34]]
        code, A_list, B_list = create_bivariate_bicycle_codes(21,18, [3], [10,17], [3,19], [5])
        d = 34
    else: 
        raise ValueError('Not a valid BB code.')

    print(code.name)
    
    if base_decoder in ['BP','bp','Bp']:
        decoder = BpDecoder
    elif base_decoder in ['BP+OSD','BPOSD','bposd','osd','BpOsd']:
        decoder = BpOsdDecoder
    elif base_decoder in ['BP+LSD','BPLSD','bplsd','lsd','BpLsd']:
        decoder = BpLsdDecoder
    else:
        raise ValueError("Available base decoders are: BP, BP+OSD and BP+LSD.")

    if basis == 'Z':
        z_basis = True
    elif basis == 'X':
        z_basis = False

    
    ## DEM 
    circuit = build_circuit(code, A_list, B_list, 
                            p=error_rate, # physical error rate
                            num_repeat=d, # usually set to code distance
                            z_basis=z_basis,   # whether in the z-basis or x-basis
                            use_both=False, # whether use measurement results in both basis to decode one basis
                           )
    dem = circuit.detector_error_model()

    chk, obs, priors, col_dict = dem_to_check_matrices(dem, return_col_dict=True)

    start_time = time.perf_counter()
    dem_sampler: stim.CompiledDemSampler = dem.compile_sampler()
    det_data, obs_data, err_data = dem_sampler.sample(shots=num_shots, return_errors=False, bit_packed=False)
    end_time = time.perf_counter()
    print(f"Stim: noise sampling for {num_shots} shots, elapsed time:", end_time-start_time)
    
    DEM_priors_list = []
    ensemble = []
    no_of_auts = len(DEM_col_perms)
    for i in range(no_of_auts): 
        col_perms = DEM_col_perms[i]
        new_checks = chk@col_perms
        new_priors = priors @ col_perms 
        new_decoder = decoder(coo_matrix(new_checks), 
                        channel_probs=new_priors, 
                        **decoder_hyperparams
                        )
        DEM_priors_list.append(new_priors)
        ensemble.append(new_decoder)

 
    def likelihood(error):
        channel_probs = np.array(priors)
        likelihoods = np.where(error == 0, 1 - channel_probs, channel_probs)
        return np.prod(likelihoods)

    base_dec_errs = 0
    AutDEC_errs = 0

    print('Starting decoding.')
    print('Trial no \t BaseDec Errors \t AutDEC Errors')
    checkpoint = 50
    
    for trial in range(num_shots):
        correction_ensemble = []
        for i in range(len(ensemble)):
            corr_AutDEC = ensemble[i].decode(DEM_row_perms[i]@det_data[trial]%2)
            if i==0:
                correction_ensemble.append(corr_AutDEC) # base decoder correction added as first element
                corr_base = corr_AutDEC.copy()
            elif ensemble[i].converge: # check if BP converged. 
                correction_ensemble.append(corr_AutDEC)
                if base_decoder == ['BP','bp']: # if a BP path converged, exit the loop.
                    break

        correction_ensemble = sorted(correction_ensemble,key=likelihood,reverse=True)

        corr_AutDEC = correction_ensemble[0].copy() # final ensemble correction with highest likelihood in the list

        ans_base = (obs @ corr_base + obs_data[trial]) % 2
        ans_AutDEC = (obs @ corr_AutDEC + obs_data[trial]) % 2
        ec_result_base = ans_base.any()
        ec_result_AutDEC = ans_AutDEC.any()
        base_dec_errs += ec_result_base
        AutDEC_errs += ec_result_AutDEC

        
        if (trial+1)%checkpoint == 0:
            print(f'{trial+1} \t {base_dec_errs} \t {AutDEC_errs}')
    return num_shots, base_dec_errs, AutDEC_errs


def bb_dem_matrix(bb_code_name,basis='Z'):
    if bb_code_name == 'BB72' or bb_code_name == 'bb72':
        # [[72,12,6]]
        code, A_list, B_list = create_bivariate_bicycle_codes(6, 6, [3], [1,2], [1,2], [3])
        d = 6
    elif bb_code_name == 'BB90' or bb_code_name == 'bb90':
        # [[90,8,10]]
        code, A_list, B_list = create_bivariate_bicycle_codes(15, 3, [9], [1,2], [2,7], [0])
        d = 10
    elif bb_code_name == 'BB108' or bb_code_name == 'bb108':
        # [[108,8,10]]
        code, A_list, B_list = create_bivariate_bicycle_codes(9, 6, [3], [1,2], [1,2], [3])
        d = 10
    elif bb_code_name == 'BB144' or bb_code_name == 'bb144':
        # [[144,12,12]]
        code, A_list, B_list = create_bivariate_bicycle_codes(12, 6, [3], [1,2], [1,2], [3])
        d = 12
    elif bb_code_name == 'BB288' or bb_code_name == 'bb288':
        # [[288,12,18]]
        code, A_list, B_list = create_bivariate_bicycle_codes(12, 12, [3], [2,7], [1,2], [3])
        d = 18
    elif bb_code_name == 'BB360' or bb_code_name == 'bb360':
        # [[360,12,<=24]]
        code, A_list, B_list = create_bivariate_bicycle_codes(30, 6, [9], [1,2], [25,26], [3])
        d = 24
    elif bb_code_name == 'BB756' or bb_code_name == 'bb756':
        # [[756,16,<=34]]
        code, A_list, B_list = create_bivariate_bicycle_codes(21,18, [3], [10,17], [3,19], [5])
        d = 34
    else: 
        raise ValueError('Not a valid BB code.')
    
    if basis == 'Z':
        z_basis = True
    elif basis == 'X':
        z_basis = False
    ## DEM 
    circuit = build_circuit(code, A_list, B_list, 
                            p=0.001, # random error rate
                            num_repeat=d, # usually set to code distance
                            z_basis=z_basis,   # whether in the z-basis or x-basis
                            use_both=False, # whether use measurement results in both basis to decode one basis
                           )
    dem = circuit.detector_error_model()

    chk, obs, priors, col_dict = dem_to_check_matrices(dem, return_col_dict=True)

    return chk