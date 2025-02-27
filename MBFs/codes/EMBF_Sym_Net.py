# %%
from itertools import product, combinations_with_replacement, permutations
import numpy as np
import pandas as pd
from multiprocessing import Pool, Value, Array
from MBF_Sym_Net import gen_inps, input_pairs, generate_rows, process_function, is_monotonic, add_func_if_monotonic
#%%
# Input
n = 4 # Number of nodes in Toggle-n network
#%%
def middle_layer(func, len_inputs, n):
    suum = func.sum()
    if n%2:
        return (suum == (len_inputs//2 + 1)) or (suum == len_inputs//2 - 1)
    else:
        return suum == len_inputs//2
    
def check_essential(df):
    # Check if it is essential
    # For each input column j, check if the output column depends on input j
    out = df.columns[-1]
    for j in df.columns[:-1]:
        # Groupby not including j
        not_j = df.columns[:-1].difference([j])
        if not_j.empty:
            df_un = df.nunique()
        else: 
            df_un = df.groupby(not_j.tolist()).nunique()
        # If any of the output column changes on change of input j (keeping other inputs fixed), then it is essential
        if not (df_un[out] == df_un[j]).any():
            return False
    return True

def is_essential_df(func, inputs):
    df = pd.DataFrame(np.append(inputs, func[:, None], axis=1))
    return check_essential(df)

def is_essential(func, inputs):
    for j in range(np.shape(inputs)[1]):
        not_j = np.delete(inputs, j, axis=1)
        # Sort the 2d array
        indx = np.lexsort(not_j.T)
        not_j = not_j[indx]
        vals, idx_start = np.unique(not_j, return_index=True, axis=0)
        # sort index again :(
        idx_start = np.sort(idx_start)

def return_func_if_embf(func_index):
    func = process_function(func_index, len(inputs))
    if not middle_layer(func, len(inputs), n):
        return
    if not is_monotonic(func, inp_pair):
        return
    if is_essential_df(func, inputs):
        return func
# %%
if __name__ == '__main__':
    inputs, inp_pair = gen_inps(n-1)
    rows = generate_rows(inputs)
    all_funcs = range(2**len(inputs))
    np.savetxt(f'../Output/EMBF_T{n}_inputs.txt', inputs, fmt='%d')
    #%%
    with Pool(32) as p:
        funcs = p.map(return_func_if_embf, all_funcs)
    # %%
    # Remove the None values from the list
    funcs = [x for x in funcs if x is not None]
    # %%
    df = pd.DataFrame(funcs)
    df.to_csv(f'../Output/EMBF_T{n}_funcs.txt', index=False, header=False, sep=' ')
# %%
for i in df.index:
    df_io = pd.DataFrame(np.hstack((inputs, df.loc[i,:].values[:, None])))
    df_io.groupby(list(range(n-2))).nunique()
# %%
