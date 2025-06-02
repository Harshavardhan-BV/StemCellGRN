# MBFs

## Calculation codes
Brute-forces over all the possible boolean functions to identify number of monotone boolean functions giving a particular output. Outputs $v^n(k)$ (i.e the number of MBFs which output 1 with inputs having k 1s)
- [MBF_Sym_Net.py](./codes/MBF_Sym_Net.py): Parallelized reimplemetation of original code that can be run for upto T6. Still not memory efficient; consumes ~85GB of RAM parallelized on 32 threads.
- [MBF_CUDA.py](./codes/MBF_CUDA.py): GPU accelerated version on CUDA. Code ported by ChatGPT, so caution warranted. Much faster and memory efficient, and outputs match with CPU code.

## Analysis
- [process_out.py](./Analysis/process_out.py): Processing the output from either of the previous 2 codes to give number of MBF combinations that give steady states with k-high states ($\phi^n_k$)

## Misc
Some exploratory analyses (not part of manuscript).
- [EMBF_Sym_Net.py](./codes/EMBF_Sym_Net.py): Brute force search of Essential monotone boolean functions in the middle layer
- [Ising.py](./Analysis/Ising.py): Ising formulation of toggle-n networks correspondance to MBF/EMBF. 