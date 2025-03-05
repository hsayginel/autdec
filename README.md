# AutDEC: Quantum Automorphism Ensemble Decoder 
Python package for implementation of the Quantum Automorphism Ensemble Decoder [1], "AutDEC":

AutDEC is designed for parallel ensemble decoding of quantum error-correcting codes with large automorphism groups. It works by creating an ensemble of automorphism guided BP, BP+OSD or BP+LSD decoders. For Bivariate Bicycle Codes under circuit-level noise, using only a BP-based ensemble, we approach and match BP+OSD0 performance without postprocessing. On the other hand, using a BP+OSD0-based ensemble, we achieve similar decoding performance to BP+OSD10. It is inspired by the work of M. Geiselhart et al for classical codes [2]. 

Note: AutDEC is currently a work in progress and a serial implementation is provided with parallelization in future plans.

## Installation
PyPI: `pip install autdec`

Source code: Download this repository and run `pip install .`

## Overview
Below we have an overview of the AutDEC online decoding phase:

<img src="autdec.png" width="600">

The effectiveness of BP is limited by the presence of short-cycles on the Tanner graph. Automorphisms allow AutDEC to bypass this problem as seen below: 

<img src="rm_15_aut_e_colouredcycles.png" width="600">

We provide functions for finding the graph automorphism group of arbitrary codes. 

## Results
In [Code Capacity](./code_capacity/) we provide a notebook and simulation results for [[15,1,3]] punctured Reed-Muller code in depolarising code capacity noise model. 

For [Bivariate Bicycle Codes](./bivariate_cycle_codes) we provide a notebook and simulation results under circuit-level noise simulations. Currently, simulations can take a long time for large codes because of the serial implementation. 

## Citation 
Paper
```
@article{autdec_paper,
    author = "Koutsioumpas, Stergios and Sayginel, Hasan and Webster, Mark and Browne, Dan E.",
    title = "{Automorphism Ensemble Decoding of Quantum LDPC Codes}",
    eprint = "2503.01738",
    archivePrefix = "arXiv",
    journal = "arXiv:2503.01738",
    primaryClass = "quant-ph",
    month = "3",
    year = "2025"
}
```



Software
```
@misc{autdec_software,
author = {Sayginel, Hasan and Koutsioumpas, Stergios},
license = {MIT},
month = mar,
title = {{AutDEC}},
url = {https://github.com/hsayginel/autdec},
version = {1.0.0},
year = {2025}
}
```

## References
[1] S. Koutsioumpas*, H. Sayginel*, M. Webster, D. E Browne, Automorphism Ensemble Decoding of Quantum LDPC Codes, (2025), arXiv:2503.01738 [quant-ph].

[2] M. Geiselhart, A. Elkelesh, M. Ebada, S. Cammerer and S. t. Brink, "Automorphism Ensemble Decoding of Reed–Muller Codes", in IEEE Transactions on Communications, vol. 69, no. 10, pp. 6424-6438, Oct. 2021, doi: 10.1109/TCOMM.2021.3098798.
