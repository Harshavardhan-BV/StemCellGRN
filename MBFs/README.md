# MBFs

## Calculation codes
Brute-forces over all the possible boolean functions to identify number of monotone boolean functions giving a particular output. Outputs $v^k(j)$ (i.e the number of MBFs which output 1 with inputs having j 1s)
- [MBF_Sym_Net.py](./MBF_Sym_Net.py): Parallelized reimplemetation of original code that can be run for upto T6. Still not memory efficient; consumes ~85GB of RAM parallelized on 32 threads.
- [MBF_CUDA.py](./MBF_CUDA.py): GPU accelerated version on CUDA. Code ported by ChatGPT, so caution warranted. Much faster and memory efficient, and outputs match with CPU code.

## Analysis
- [process_out.py](./process_out.py): Processing the output from either of the previous 2 codes to give number of MBF combinations that give steady states with j high states ($n^j_i$)
